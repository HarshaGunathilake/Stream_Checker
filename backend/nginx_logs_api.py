#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, Response, jsonify
from collections import defaultdict, deque
from datetime import datetime
from urllib.parse import urlparse
import time
import os
import re
import json
import math
from pathlib import Path
import traceback
import logging

# Optional ML deps (server may or may not have them installed)
try:
    import numpy as np
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    np = None
    XGBClassifier = None
    XGB_AVAILABLE = False

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.errorhandler(Exception)
def handle_all_errors(e):
    tb = traceback.format_exc()
    app.logger.error(tb)

    # Show full traceback only when debug=1 is passed
    if request.args.get("debug") == "1":
        return jsonify({"error": str(e), "trace": tb}), 500

    return jsonify({"error": "Internal Server Error"}), 500


# =========================
# NGINX LOG PARSING CONFIG
# =========================

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

# Fallback heuristic ONLY (used if model not available)
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
            buf = parts[0]  # might be incomplete
            for p in parts[1:]:
                if p:
                    try:
                        lines.appendleft(p.decode("utf-8", errors="ignore"))
                    except Exception:
                        pass

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

    path = path.split("?", 1)[0]
    if path.endswith(".m3u8"):
        path = path[:-5]  # remove ".m3u8"
    path = path.rstrip("/")
    return path


def get_client_ip(ip: str, xff: str) -> str:
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


# ==========================================
# XGBOOST: LOAD + AUTO-RELOAD ON FILE CHANGE
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "xgb_bot_model.json"
SCHEMA_PATH = MODEL_DIR / "feature_schema.json"

_XGB_MODEL = None
_XGB_SCHEMA = None
_XGB_LOAD_ERROR = None
_XGB_MODEL_MTIME = None
_XGB_SCHEMA_MTIME = None


def _mtime(p: Path):
    try:
        return p.stat().st_mtime
    except Exception:
        return None


def clear_xgb_cache():
    global _XGB_MODEL, _XGB_SCHEMA, _XGB_LOAD_ERROR, _XGB_MODEL_MTIME, _XGB_SCHEMA_MTIME
    _XGB_MODEL = None
    _XGB_SCHEMA = None
    _XGB_LOAD_ERROR = None
    _XGB_MODEL_MTIME = None
    _XGB_SCHEMA_MTIME = None


def load_xgb(force: bool = False):
    """
    Loads model+schema and caches them.
    Auto reloads if files changed.
    Returns: (model, schema, loaded_bool)
    """
    global _XGB_MODEL, _XGB_SCHEMA, _XGB_LOAD_ERROR, _XGB_MODEL_MTIME, _XGB_SCHEMA_MTIME

    if not XGB_AVAILABLE:
        _XGB_LOAD_ERROR = "xgboost/numpy not installed for this python"
        return None, None, False

    if not MODEL_PATH.exists() or not SCHEMA_PATH.exists():
        _XGB_LOAD_ERROR = f"missing files: {MODEL_PATH} or {SCHEMA_PATH}"
        return None, None, False

    cur_m = _mtime(MODEL_PATH)
    cur_s = _mtime(SCHEMA_PATH)

    # Return cached if valid and unchanged
    if (
        not force
        and _XGB_MODEL is not None
        and _XGB_SCHEMA is not None
        and _XGB_MODEL_MTIME == cur_m
        and _XGB_SCHEMA_MTIME == cur_s
    ):
        return _XGB_MODEL, _XGB_SCHEMA, True

    try:
        model = XGBClassifier()
        model.load_model(str(MODEL_PATH))
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

        if not isinstance(schema, list) or not schema:
            raise ValueError("feature_schema.json must be a non-empty JSON list")

        _XGB_MODEL = model
        _XGB_SCHEMA = schema
        _XGB_LOAD_ERROR = None
        _XGB_MODEL_MTIME = cur_m
        _XGB_SCHEMA_MTIME = cur_s
        return _XGB_MODEL, _XGB_SCHEMA, True

    except Exception as e:
        clear_xgb_cache()
        _XGB_LOAD_ERROR = f"failed to load model: {e}"
        app.logger.exception("XGB load failed")
        return None, None, False


def _safe_div(a, b):
    return a / b if b else 0.0


def _percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def build_features_for_session(events):
    """
    Build the SAME features you used for training (feature_schema.json order)
    """
    if not events:
        return {}

    ev = sorted(events, key=lambda x: x.get("time_epoch", 0.0))
    times = [float(e.get("time_epoch", 0.0)) for e in ev]
    duration_s = max(1.0, times[-1] - times[0]) if len(times) > 1 else 1.0

    urls = [e.get("url", "") or "" for e in ev]
    statuses = [int(e.get("status", 0) or 0) for e in ev]
    bytes_list = [int(e.get("bytes", 0) or 0) for e in ev]
    rt_list = [float(e.get("rt", 0.0) or 0.0) for e in ev]

    m3u8_count = sum(1 for u in urls if u.endswith(".m3u8"))
    ts_count = sum(1 for u in urls if u.endswith(".ts"))

    gaps = []
    for i in range(1, len(times)):
        gaps.append(max(0.0, times[i] - times[i - 1]))
    gaps_sorted = sorted(gaps)
    gap_mean = sum(gaps) / len(gaps) if gaps else duration_s
    gap_p50 = _percentile(gaps_sorted, 50)
    gap_p95 = _percentile(gaps_sorted, 95)

    s4xx = sum(400 <= s < 500 for s in statuses)
    s5xx = sum(500 <= s < 600 for s in statuses)

    rt_sorted = sorted(rt_list)
    rt_avg = _safe_div(sum(rt_list), len(rt_list))
    rt_p95 = _percentile(rt_sorted, 95)

    feats = {
        "req_count": float(len(ev)),
        "duration_s": float(duration_s),
        "rps": _safe_div(len(ev), duration_s),

        "m3u8_count": float(m3u8_count),
        "ts_count": float(ts_count),
        "m3u8_ts_ratio": _safe_div(m3u8_count, ts_count),

        "bytes_sum": float(sum(bytes_list)),
        "bytes_avg": _safe_div(sum(bytes_list), len(bytes_list)),

        "rt_avg": float(rt_avg),
        "rt_p95": float(rt_p95),

        "gap_mean": float(gap_mean),
        "gap_p50": float(gap_p50),
        "gap_p95": float(gap_p95),

        "err_rate": _safe_div(s4xx + s5xx, len(ev)),
        "status_4xx": float(s4xx),
        "status_5xx": float(s5xx),
    }
    return feats


def xgb_predict_proba_bot(feats: dict):
    """
    Returns P(bot) using schema-ordered vector with correct 2D shape (1, n_features).
    IMPORTANT: training used label=1 for bots -> we return predict_proba()[0,1] directly.
    """
    try:
        if feats is None:
            return None

        model, schema, ok = load_xgb(force=False)
        if not ok or model is None or schema is None:
            return None

        row = [float(feats.get(k, 0.0) or 0.0) for k in schema]
        X = np.array([row], dtype=np.float32)  # shape (1, n)

        return float(model.predict_proba(X)[0, 1])
    except Exception:
        app.logger.exception("xgb_predict_proba_bot failed")
        return None


# ==================
# BASIC ENDPOINTS
# ==================

@app.route("/ping")
def ping():
    return "OK", 200


@app.route("/api/nginx/model/status")
def model_status():
    _, _, loaded = load_xgb(force=False)
    return jsonify({
        "xgb_available": XGB_AVAILABLE,
        "model_loaded": loaded,
        "xgb_load_error": _XGB_LOAD_ERROR,
        "model_paths": {"model": str(MODEL_PATH), "schema": str(SCHEMA_PATH)},
        "mtimes": {"model": _mtime(MODEL_PATH), "schema": _mtime(SCHEMA_PATH)},
    })


@app.route("/api/nginx/model/reload")
def model_reload():
    clear_xgb_cache()
    _, _, loaded = load_xgb(force=True)
    return jsonify({
        "reloaded": True,
        "model_loaded": loaded,
        "xgb_load_error": _XGB_LOAD_ERROR,
        "xgb_available": XGB_AVAILABLE,
        "model_paths": {"model": str(MODEL_PATH), "schema": str(SCHEMA_PATH)},
    })


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
    return Response("\n".join(raw_lines) + ("\n" if raw_lines else ""), mimetype="text/plain")


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
            item["time_epoch"] = float(item.get("time_epoch", 0.0))
            items.append(item)
        elif not only_parsed:
            items.append({"raw": line.rstrip("\n")})

    return jsonify({"count": len(items), "lines": items})


# ===========================
# SIMPLE REALTIME VIEWER COUNT
# ===========================

@app.route("/api/nginx/viewers")
def get_realtime_viewers():
    stream_key_input = request.args.get("stream_key", "/hls/NMT.m3u8")
    base = normalize_stream_key(stream_key_input)

    window_seconds = int(request.args.get("window_seconds", 30))
    window_seconds = max(5, min(window_seconds, 300))

    tail_n = int(request.args.get("tail_lines", 8000))
    tail_n = max(1000, min(tail_n, 200000))

    now_epoch = time.time()
    cutoff = now_epoch - window_seconds

    raw_lines = tail_lines(LOG_FILE, max_lines=tail_n)

    parsed_ok = 0
    matched = 0

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

        vid = (p.get("vid") or "").strip()
        if vid and vid not in ("-", ""):
            sid = f"vid:{vid}"
        else:
            sid = f"{p['client_ip']}||{p.get('ua','')}"

        ua = p.get("ua", "") or ""
        referer = p.get("referer", "") or ""
        status = p.get("status", 0)

        ua_bot = bool(BOT_UA_RE.search(ua))
        internal = bool(INTERNAL_REFERER_RE.search(referer))

        if ua_bot:
            bot_sessions.add(sid)
            continue

        if (200 <= status < 300) and (not internal):
            real_sessions.add(sid)

    def sid_to_id(sid: str) -> str:
        return sid.replace("vid:", "", 1)

    return jsonify({
        "stream_key": base,
        "window_seconds": window_seconds,
        "tail_lines": tail_n,
        "approx_viewers": len(real_sessions),
        "bot_count": len(bot_sessions),
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


# ==========================
# XGB SCORING ENDPOINT
# ==========================

@app.route("/api/nginx/bots/score")
def score_bots_xgb():
    """
    Returns per-session bot probability for a stream in the last N seconds.

    Query:
      stream_key=/hls/NMT.m3u8
      window_seconds=30
      tail_lines=8000
      threshold=0.7
      limit=200

    Model files expected:
      ./model/xgb_bot_model.json
      ./model/feature_schema.json

    If model not present or fails to load, fallback uses UA regex.
    """
    stream_key_input = request.args.get("stream_key", "/hls/NMT.m3u8")
    base = normalize_stream_key(stream_key_input)

    window_seconds = int(request.args.get("window_seconds", 30))
    window_seconds = max(5, min(window_seconds, 300))

    tail_n = int(request.args.get("tail_lines", 8000))
    tail_n = max(1000, min(tail_n, 200000))

    threshold = float(request.args.get("threshold", 0.7))
    threshold = max(0.0, min(threshold, 1.0))

    limit = int(request.args.get("limit", 200))
    limit = max(1, min(limit, 1000))

    now_epoch = time.time()
    cutoff = now_epoch - window_seconds

    raw_lines = tail_lines(LOG_FILE, max_lines=tail_n)

    sessions = defaultdict(list)
    parsed_ok = 0
    matched = 0

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

        vid = (p.get("vid") or "").strip()
        if vid and vid not in ("-", ""):
            sid = f"vid:{vid}"
        else:
            sid = f"{p['client_ip']}||{p.get('ua','')}"
        sessions[sid].append(p)

    # attempt load (and auto-reload if file changed)
    _, _, model_loaded = load_xgb(force=False)

    scored = []
    for sid, events in sessions.items():
        feats = build_features_for_session(events)

        proba = xgb_predict_proba_bot(feats)

        # fallback if model isn't available
        if proba is None:
            ua = (events[-1].get("ua", "") or "")
            proba = 1.0 if BOT_UA_RE.search(ua) else 0.0

        last = events[-1]
        scored.append({
            "sid": sid.replace("vid:", "", 1),
            "client_ip": last.get("client_ip"),
            "ua": last.get("ua"),
            "bot_probability": float(proba),
            "features": feats,
        })

    scored.sort(key=lambda x: x["bot_probability"], reverse=True)
    scored = scored[:limit]

    likely_bots = sum(1 for s in scored if s["bot_probability"] >= threshold)
    likely_real = len(scored) - likely_bots

    return jsonify({
        "stream_key": base,
        "window_seconds": window_seconds,
        "tail_lines": tail_n,
        "threshold": threshold,
        "model_loaded": model_loaded,
        "xgb_load_error": _XGB_LOAD_ERROR,
        "xgb_available": XGB_AVAILABLE,
        "model_paths": {
            "model": str(MODEL_PATH),
            "schema": str(SCHEMA_PATH),
        },
        "summary": {
            "sessions_scored": len(scored),
            "likely_bots": likely_bots,
            "likely_real": likely_real,
        },
        "debug": {
            "log_file": LOG_FILE,
            "parsed_lines_in_tail": parsed_ok,
            "matched_lines_in_window": matched,
            "unique_sessions_in_window": len(sessions),
        },
        "sessions": scored
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
