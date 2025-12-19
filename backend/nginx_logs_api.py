from flask import Flask, request, Response, jsonify
from collections import defaultdict, deque
from datetime import datetime
from urllib.parse import urlparse
import time
import os
import re

app = Flask(__name__)

# Read the NEW log
LOG_FILE = "/usr/local/nginx/logs/hls_access.log"

# New hls_ext format
LOG_PATTERN = re.compile(
    r'(?P<ip>\S+)\s+-\s+(?:(?P<remote_user>\S+)\s+)?\[(?P<time>[^\]]+)\]\s+'
    r'"(?P<method>\S+)\s+(?P<url>\S+)\s+HTTP/[^"]+"\s+'
    r'(?P<status>\d{3})\s+(?P<bytes>\d+)\s+'
    r'"(?P<referer>[^"]*)"\s+"(?P<ua>[^"]*)"\s+'
    r'rt=(?P<rt>[\d\.]+)\s+'
    r'xff="(?P<xff>[^"]*)"\s+'
    r'range="(?P<range>[^"]*)"\s+'
    r'accept="(?P<accept>[^"]*)"\s+'
    r'al="(?P<al>[^"]*)"\s+'
    r'origin="(?P<origin>[^"]*)"\s+'
    r'vid="(?P<vid>[^"]*)"'
)


TIME_FORMAT = "%d/%b/%Y:%H:%M:%S %z"

BOT_UA_RE = re.compile(r"(k6|curl|wget|python-requests|go-http-client|httpclient|scrapy|bot)", re.I)
INTERNAL_REFERER_RE = re.compile(r"(localhost:8501)", re.I)


def tail_lines(file_path: str, max_lines: int = 4000, block_size: int = 8192):
    """
    Efficiently read last N lines without scanning the whole file.
    """
    if not os.path.exists(file_path):
        return []

    lines = deque(maxlen=max_lines)
    with open(file_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        remaining = file_size
        buf = b""

        while remaining > 0 and len(lines) < max_lines:
            step = min(block_size, remaining)
            remaining -= step
            f.seek(remaining)
            data = f.read(step)
            buf = data + buf

            parts = buf.split(b"\n")
            # keep first part as it might be incomplete
            buf = parts[0]
            for p in parts[1:]:
                if p:
                    try:
                        lines.appendleft(p.decode("utf-8", errors="ignore"))
                    except Exception:
                        pass

        # last leftover
        if buf and len(lines) < max_lines:
            try:
                lines.appendleft(buf.decode("utf-8", errors="ignore"))
            except Exception:
                pass

    return list(lines)[-max_lines:]


def normalize_stream_key(s: str) -> str:
    """
    Accepts:
      - https://rtmp1.vodhosting.com/hls/NMT.m3u8
      - /hls/NMT.m3u8
      - /hls/NMT
      - hls/NMT.m3u8
    Returns a BASE prefix like: /hls/NMT
    (so it matches both /hls/NMT.m3u8 and /hls/NMT-123.ts)
    """
    s = (s or "").strip()
    if not s:
        return "/hls/NMT"

    p = urlparse(s)
    path = p.path if (p.scheme and p.netloc) else s
    if not path.startswith("/"):
        path = "/" + path

    # Remove query if any (defensive)
    path = path.split("?", 1)[0]

    # Strip .m3u8 if present
    if path.endswith(".m3u8"):
        path = path[:-5]  # remove ".m3u8"

    # Strip trailing slash
    path = path.rstrip("/")

    return path


def get_client_ip(ip: str, xff: str) -> str:
    # If you later enable real_ip, you can rely on remote_addr.
    # For now, if xff exists, use first IP in it.
    if xff and xff not in ("-", ""):
        first = xff.split(",")[0].strip()
        if first:
            return first
    return ip


def parse_line(line: str):
    m = LOG_PATTERN.search(line)
    if not m:
        return None

    try:
        ts_epoch = datetime.strptime(m.group("time"), TIME_FORMAT).timestamp()
        ip = m.group("ip")
        xff = m.group("xff")
        client_ip = get_client_ip(ip, xff)

        url = m.group("url").split("?", 1)[0]

        return {
            "ip": ip,
            "client_ip": client_ip,
            "xff": xff,
            "time_epoch": ts_epoch,
            "method": m.group("method"),
            "url": url,
            "status": int(m.group("status")),
            "bytes": int(m.group("bytes")),
            "referer": m.group("referer"),
            "ua": m.group("ua"),
            "rt": float(m.group("rt") or 0.0),
            "range": m.group("range"),
            "vid": m.group("vid"),
            "raw": line.rstrip("\n"),
        }
    except Exception:
        return None


@app.route("/api/nginx/viewers")
def get_realtime_viewers():
    stream_key_input = request.args.get("stream_key", "/hls/NMT.m3u8")
    base = normalize_stream_key(stream_key_input)  # e.g. /hls/NMT

    window_seconds = int(request.args.get("window_seconds", 30))
    window_seconds = max(5, min(window_seconds, 300))

    # how many lines to scan from end of file
    tail_n = int(request.args.get("tail_lines", 8000))
    tail_n = max(1000, min(tail_n, 200000))

    now_epoch = time.time()
    cutoff = now_epoch - window_seconds

    raw_lines = tail_lines(LOG_FILE, max_lines=tail_n)

    parsed_ok = 0
    matched = 0

    # sets of session IDs
    bot_sessions = set()
    real_sessions = set()

    for line in raw_lines:
        p = parse_line(line)
        if not p:
            continue
        parsed_ok += 1

        if p["time_epoch"] < cutoff:
            continue

        if not p["url"].startswith(base):
            continue

        matched += 1

        # session id: prefer vid
        vid = (p.get("vid") or "").strip()
        if vid and vid not in ("-", ""):
            sid = f"vid:{vid}"
        else:
            sid = f"{p['client_ip']}||{p['ua']}"

        ua = p.get("ua", "") or ""
        referer = p.get("referer", "") or ""
        status = p.get("status", 0)

        ua_bot = bool(BOT_UA_RE.search(ua))
        internal = bool(INTERNAL_REFERER_RE.search(referer))

        # classify bot
        if ua_bot:
            bot_sessions.add(sid)
            continue

        # classify real (very simple rule for now)
        if (200 <= status < 300) and (not internal):
            real_sessions.add(sid)

    def sid_to_id(sid: str) -> str:
        # return VUxx if it’s a vid session, else show the raw sid
        return sid.replace("vid:", "", 1)

    return jsonify({
        "stream_key": base,
        "window_seconds": window_seconds,
        "tail_lines": tail_n,

        # counts
        "approx_viewers": len(real_sessions),
        "bot_count": len(bot_sessions),

        # optional lists (helpful for verifying)
        "real_clients": sorted(sid_to_id(s) for s in real_sessions)[:200],
        "bot_clients": sorted(sid_to_id(s) for s in bot_sessions)[:200],

        "debug": {
            "log_file": LOG_FILE,
            "parsed_lines_in_tail": parsed_ok,
            "matched_lines_in_window": matched,
            "real_sessions_seen": len(real_sessions),
            "bot_sessions_seen": len(bot_sessions),
        }
    })


@app.route("/ping")
def ping():
    return "OK", 200


@app.route("/api/nginx/logs/text")
def get_logs_text():
    """
    Return last N lines of nginx log as plain text.
    Usage: /api/nginx/logs/text?lines=200
    """
    try:
        lines = int(request.args.get("lines", 200))
    except ValueError:
        lines = 200
    lines = max(1, min(lines, 2000))

    raw_lines = tail_lines(LOG_FILE, max_lines=lines)
    return Response("".join(raw_lines), mimetype="text/plain")


@app.route("/api/nginx/logs/json")
def get_logs_json():
    """
    Return last N lines as JSON (parsed if possible).
    Usage: /api/nginx/logs/json?lines=200&only_parsed=true
    """
    try:
        lines = int(request.args.get("lines", 200))
    except ValueError:
        lines = 200
    lines = max(1, min(lines, 2000))

    only_parsed = request.args.get("only_parsed", "false").lower() == "true"

    raw_lines = tail_lines(LOG_FILE, max_lines=lines)
    items = []
    for line in raw_lines:
        parsed = parse_line(line)
        if parsed:
            item = parsed.copy()
            # if you store epoch time
            if "time_epoch" in item:
                item["time_epoch"] = float(item["time_epoch"])
            items.append(item)
        elif not only_parsed:
            items.append({"raw": line.rstrip("\n")})

    return jsonify({"count": len(items), "lines": items})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
