import os
import json
import time
from datetime import datetime
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

# ---------- BASIC SETUP ----------
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

st.set_page_config(page_title="Live Viewer + Bot Monitor", layout="wide")

# Hide Streamlit main menu / deploy bar
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    div[data-testid="stToolbar"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Professional UI CSS
st.markdown(
"""
<style>
  div[data-testid="stVerticalBlockBorderWrapper"]{
    border: 1px solid rgba(148,163,184,0.16) !important;
    background: rgba(2,6,23,0.35) !important;
    border-radius: 18px !important;
    padding: 18px !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25) !important;
  }
  .muted { color: #94a3b8; font-size: 0.9rem; }
  .section-title { font-size: 1.05rem; font-weight: 650; margin-bottom: 10px; }
  .pill {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.18);
    background: rgba(15,23,42,0.55);
    font-size: 0.85rem;
    color: #cbd5e1;
  }
</style>
""",
    unsafe_allow_html=True,
)

# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "last_data" not in st.session_state:
    st.session_state.last_data = None
if "last_stream_key" not in st.session_state:
    st.session_state.last_stream_key = None
if "last_api_base" not in st.session_state:
    st.session_state.last_api_base = None
if "last_report_html" not in st.session_state:
    st.session_state.last_report_html = None
if "last_report_name" not in st.session_state:
    st.session_state.last_report_name = None


# ---------- HELPERS ----------
def extract_stream_key(url: str) -> str:
    """
    Accepts:
      - Full URL:  https://domain.com/hls/NMT.m3u8
      - Path:      /hls/NMT.m3u8
      - Relative:  hls/NMT.m3u8
    Returns:
      - Normalized path: /hls/NMT.m3u8
    """
    url = (url or "").strip()
    if not url:
        return "/"

    parsed = urlparse(url)
    if parsed.scheme and parsed.netloc:
        path = parsed.path or "/"
    else:
        path = url

    if not path.startswith("/"):
        path = "/" + path

    return path


def get_viewer_data(api_base: str, stream_key: str, window_seconds: int, bot_threshold: int):
    url = api_base.rstrip("/") + "/api/nginx/viewers"
    params = {
        "stream_key": stream_key,
        "window_seconds": window_seconds,
        "bot_threshold": bot_threshold,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def render_hls_player(m3u8_url: str, height: int = 420):
    """
    HLS player using hls.js (works for .m3u8 in most browsers).
    Note:
      - Autoplay may be blocked unless muted.
      - If your Streamlit site is HTTPS and stream is HTTP, the browser may block it (mixed content).
    """
    m3u8_url = (m3u8_url or "").strip()
    if not m3u8_url:
        components.html("<div class='muted'>No stream URL provided.</div>", height=60)
        return

    html = f"""
    <div style="width:100%; height:{height}px; border-radius:18px; overflow:hidden; border:1px solid rgba(148,163,184,0.16); background:rgba(2,6,23,0.35);">
      <video id="video" controls playsinline muted
        style="width:100%; height:100%; object-fit:contain; background:#000;"></video>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
    <script>
      const video = document.getElementById('video');
      const src = {json.dumps(m3u8_url)};

      function load() {{
        if (Hls.isSupported()) {{
          const hls = new Hls({{
            lowLatencyMode: true,
            enableWorker: true,
            backBufferLength: 30
          }});
          hls.loadSource(src);
          hls.attachMedia(video);
          hls.on(Hls.Events.MANIFEST_PARSED, function() {{
            video.play().catch(() => {{}});
          }});
        }} else if (video.canPlayType('application/vnd.apple.mpegurl')) {{
          video.src = src;
          video.addEventListener('loadedmetadata', function() {{
            video.play().catch(() => {{}});
          }});
        }} else {{
          video.outerHTML = "<div style='padding:14px;color:#cbd5e1;font-family:system-ui;'>HLS not supported in this browser.</div>";
        }}
      }}

      load();
    </script>
    """
    components.html(html, height=height + 30, scrolling=False)


# ---------- ADVANCED HTML REPORT ----------
def generate_html_report(df: pd.DataFrame, stream_key: str, api_base: str) -> str:
    if df.empty:
        return "<html><body><h2>No data available.</h2></body></html>"

    df = df.copy()
    df["viewers"] = df["viewers"].astype(int)
    df["bot_count"] = df["bot_count"].astype(int)
    df["total_unique_ips"] = df["total_unique_ips"].astype(int)

    latest = df.iloc[-1]

    max_viewers = int(df["viewers"].max())
    avg_viewers = float(df["viewers"].mean())
    max_bots = int(df["bot_count"].max())
    avg_bots = float(df["bot_count"].mean())

    latest_viewers = int(latest["viewers"])
    latest_bots = int(latest["bot_count"])
    latest_ips = int(latest["total_unique_ips"])

    bot_ratio_latest = (latest_bots / latest_ips * 100) if latest_ips > 0 else 0.0
    avg_ips = df["total_unique_ips"].mean()
    bot_ratio_avg = (avg_bots / avg_ips * 100) if avg_ips > 0 else 0.0

    window_seconds = int(latest["window_seconds"])
    bot_threshold = int(latest["bot_threshold"])

    labels = df["time"].tolist()
    viewers_series = df["viewers"].tolist()
    bots_series = df["bot_count"].tolist()

    labels_json = json.dumps(labels)
    viewers_json = json.dumps(viewers_series)
    bots_json = json.dumps(bots_series)

    rows_html = ""
    for _, row in df.iterrows():
        rows_html += f"""
        <tr>
            <td>{row['time']}</td>
            <td>{int(row['viewers'])}</td>
            <td>{int(row['bot_count'])}</td>
            <td>{int(row['total_unique_ips'])}</td>
            <td>{int(row['window_seconds'])}</td>
            <td>{int(row['bot_threshold'])}</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Viewer Report - {stream_key}</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{
                margin: 0;
                background: radial-gradient(circle at top left, #0b1120, #020617 55%);
                font-family: system-ui, sans-serif;
                color: #e5e7eb;
            }}
            .page {{
                max-width: 1200px;
                margin: auto;
                padding: 24px;
            }}
            .header {{
                display: flex;
                justify-content: space-between;
                padding: 16px;
                border-radius: 16px;
                border: 1px solid #1f2937;
                background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(15,23,42,0.8));
                margin-bottom: 24px;
            }}
            .grid-3 {{
                display: grid;
                grid-template-columns: repeat(3,1fr);
                gap: 16px;
                margin-bottom: 24px;
            }}
            .card {{
                padding: 16px;
                border-radius: 14px;
                border: 1px solid #1f2937;
                background: rgba(148,163,184,0.08);
            }}
            .kpi-label {{ font-size: 12px; color: #9ca3af; }}
            .kpi-value {{ font-size: 26px; font-weight: 600; }}
            .kpi-sub {{ color: #9ca3af; font-size: 12px; }}
            .flex {{ display: flex; gap: 16px; }}
            .panel {{
                flex: 2;
                padding: 16px;
                border-radius: 16px;
                border: 1px solid #1f2937;
                background: rgba(30,64,175,0.15);
            }}
            .panel-soft {{
                flex: 1;
                padding: 16px;
                border-radius: 16px;
                border: 1px solid #1f2937;
                background: rgba(148,163,184,0.08);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 16px;
            }}
            th, td {{
                border: 1px solid #1f2937;
                padding: 8px;
                font-size: 13px;
                text-align: center;
            }}
            th {{
                background: #0f172a;
                color: #9ca3af;
            }}
        </style>
    </head>

    <body>
    <div class="page">

        <div class="header">
            <div>
                <div style="font-size:24px;font-weight:600;">HLS Viewer & Bot Report</div>
                <div style="color:#9ca3af;">Stream: {stream_key}</div>
            </div>
            <div style="font-size:12px;color:#9ca3af;">Generated at: {latest['time']}</div>
        </div>

        <div class="grid-3">
            <div class="card">
                <div class="kpi-label">Latest viewers</div>
                <div class="kpi-value">{latest_viewers}</div>
                <div class="kpi-sub">Max: {max_viewers} · Avg: {avg_viewers:.1f}</div>
            </div>

            <div class="card">
                <div class="kpi-label">Bot activity</div>
                <div class="kpi-value">{latest_bots}</div>
                <div class="kpi-sub">Latest: {bot_ratio_latest:.1f}% · Avg: {bot_ratio_avg:.1f}%</div>
            </div>

            <div class="card">
                <div class="kpi-label">Unique IPs</div>
                <div class="kpi-value">{latest_ips}</div>
                <div class="kpi-sub">Window: {window_seconds}s · Th: {bot_threshold}</div>
            </div>
        </div>

        <div class="flex">
            <div class="panel">
                <canvas id="chart" height="120"></canvas>
            </div>

            <div class="panel-soft">
                <div style="font-size:16px;margin-bottom:10px;">Config</div>
                API Base: {api_base}<br>
                Stream Key: {stream_key}<br>
                Window: {window_seconds}s<br>
                Bot Threshold: {bot_threshold}<br>
                Checks: {len(df)}<br>
            </div>
        </div>

        <table>
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Viewers</th>
                    <th>Bots</th>
                    <th>IPs</th>
                    <th>Window</th>
                    <th>Threshold</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>

    </div>

    <script>
        const labels = {labels_json};
        const viewers = {viewers_json};
        const bots = {bots_json};

        new Chart(document.getElementById("chart"), {{
            type: "line",
            data: {{
                labels: labels,
                datasets: [
                    {{
                        label: "Viewers",
                        data: viewers,
                        borderColor: "rgba(59,130,246,1)",
                        backgroundColor: "rgba(59,130,246,0.15)",
                        borderWidth: 2,
                        tension: 0.25
                    }},
                    {{
                        label: "Bots",
                        data: bots,
                        borderColor: "rgba(249,115,22,1)",
                        backgroundColor: "rgba(249,115,22,0.15)",
                        borderWidth: 2,
                        tension: 0.25
                    }}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ labels: {{ color: "#e5e7eb" }} }} }},
                scales: {{
                    x: {{ ticks: {{ color: "#9ca3af" }} }},
                    y: {{ ticks: {{ color: "#9ca3af" }}, beginAtZero: true }}
                }}
            }}
        }});
    </script>

    </body>
    </html>
    """
    return html


# ---------- HEADER ----------
st.title("📺 Live Viewer + Bot Monitor")
st.caption("Manual viewer check with a user-controlled minimum run time before showing results.")

# ---------- TWO-PANE LAYOUT ----------
left, right = st.columns([0.95, 1.25], gap="large")

# ---------------- LEFT: SETTINGS ----------------
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>API Server</div>", unsafe_allow_html=True)

    api_mode = st.radio("Host type", ["IP address", "Domain"], horizontal=True)
    protocol = st.selectbox("Protocol", ["http", "https"])
    port = st.number_input("Port", 1, 65535, 8000)

    if api_mode == "IP address":
        host = st.text_input("Server IP", value="139.59.166.217")
    else:
        host = st.text_input("Domain", value="rtmp1.vodhosting.com")

    api_base = f"{protocol}://{host}:{port}"
    st.markdown(f"<span class='pill'>API Base: {api_base}</span>", unsafe_allow_html=True)

    st.markdown("<hr style='border:0;border-top:1px solid rgba(148,163,184,0.12);margin:16px 0;'>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Stream Settings</div>", unsafe_allow_html=True)
    stream_input = st.text_input("Stream URL or key", value="https://rtmp1.vodhosting.com/hls/NMT.m3u8")

    window_seconds = st.selectbox(
        "Time Window (how far back to count viewers)",
        [30, 60, 120, 300],
        index=0,
    )

    bot_threshold = st.number_input("Bot threshold (requests per window)", 5, 5000, 50)

    run_duration = st.number_input(
        "Minimum run time before showing results (seconds)",
        min_value=10,
        max_value=120,
        value=10,
        step=1,
    )

    st.markdown("<hr style='border:0;border-top:1px solid rgba(148,163,184,0.12);margin:16px 0;'>", unsafe_allow_html=True)

    run = st.button("Run Viewer Check", use_container_width=True)

    st.markdown(
        "<div class='muted'>Note: If Streamlit is HTTPS and the stream is HTTP, browser may block playback (mixed content).</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- RIGHT: PLAYER + RESULTS ----------------
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Live Preview</div>", unsafe_allow_html=True)

    render_hls_player(stream_input, height=420)

    # ---- Manual run only ----
    if run:
        stream_key = extract_stream_key(stream_input)

        # Running alert + progress (minimum run time)
        alert_box = st.warning(f"Running… collecting for at least {int(run_duration)} seconds", icon="⏳")
        progress = st.progress(0, text="Starting…")

        total = int(run_duration)
        for i in range(total):
            pct = int(((i + 1) / total) * 100)
            remaining = total - (i + 1)
            progress.progress(pct, text=f"Collecting… {remaining}s remaining")
            time.sleep(1)

        progress.empty()
        alert_box.empty()

        fetching_box = st.info("Fetching results from API…", icon="📡")
        try:
            data = get_viewer_data(api_base, stream_key, window_seconds, bot_threshold)
        except Exception as e:
            fetching_box.empty()
            st.error(f"API Error: {e}")
            st.stop()
        fetching_box.empty()

        st.success("Viewer check complete ✅", icon="✅")

        # Normalize fields (your API may return unique_ips list but not total_unique_ips)
        unique_ips = data.get("unique_ips", []) or []
        total_unique_ips = data.get("total_unique_ips", None)
        if total_unique_ips is None:
            total_unique_ips = len(unique_ips)

        # Persist last run
        st.session_state.last_data = data
        st.session_state.last_stream_key = stream_key
        st.session_state.last_api_base = api_base

        # Save entry to history
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "time": now,
            "stream_key": stream_key,
            "viewers": int(data.get("approx_viewers", 0)),
            "bot_count": int(data.get("bot_count", 0)),
            "total_unique_ips": int(total_unique_ips),
            "window_seconds": int(window_seconds),
            "bot_threshold": int(bot_threshold),
        }
        st.session_state.history.append(entry)

        # Generate report (stream-specific)
        df = pd.DataFrame(st.session_state.history)
        df_stream = df[df["stream_key"] == stream_key]

        html = generate_html_report(df_stream, stream_key, api_base)
        safe_key = stream_key.strip("/").replace("/", "_")
        filename = f"report_{safe_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = os.path.join(REPORTS_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        st.session_state.last_report_html = html
        st.session_state.last_report_name = filename

        st.success(f"Report saved: `{filepath}`")

    # Show last results (if any)
    if st.session_state.last_data:
        data = st.session_state.last_data
        unique_ips = data.get("unique_ips", []) or []
        total_unique_ips = data.get("total_unique_ips", None)
        if total_unique_ips is None:
            total_unique_ips = len(unique_ips)

        st.markdown("<hr style='border:0;border-top:1px solid rgba(148,163,184,0.12);margin:16px 0;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Results</div>", unsafe_allow_html=True)

        k1, k2, k3 = st.columns(3)
        k1.metric("Viewers", int(data.get("approx_viewers", 0)))
        k2.metric("Bots", int(data.get("bot_count", 0)))
        k3.metric("Unique IPs", int(total_unique_ips))

        with st.expander("Raw API Response", expanded=False):
            st.json(data)

        if st.session_state.last_report_html and st.session_state.last_report_name:
            st.download_button(
                "Download Report",
                data=st.session_state.last_report_html,
                file_name=st.session_state.last_report_name,
                mime="text/html",
                use_container_width=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- HISTORY TABLE ----------
st.subheader("Session History")
if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
else:
    st.info("No checks yet.")
