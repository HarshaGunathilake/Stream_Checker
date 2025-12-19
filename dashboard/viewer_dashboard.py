# dashboard.py
import time
from typing import Any, Dict, List, Tuple, Optional

import requests
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Nginx HLS Viewer + Bot Monitor",
    page_icon="📺",
    layout="wide",
)

# -----------------------------
# Small UI polish
# -----------------------------
st.markdown(
    """
    <style>
      .card {
        border: 1px solid rgba(148,163,184,0.18);
        background: rgba(2,6,23,0.35);
        padding: 16px 18px;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.22);
      }
      .muted { color: rgba(148,163,184,0.9); font-size: 0.95rem; }
      .big  { font-size: 1.25rem; font-weight: 700; }
      .pill {
        display:inline-block; padding: 4px 10px; border-radius: 999px;
        border: 1px solid rgba(148,163,184,0.22);
        background: rgba(15,23,42,0.55);
        font-size: 12px; margin-left: 6px;
      }
      .ok { color: #22c55e; }
      .bad { color: #ef4444; }
      .warn { color: #f59e0b; }
      .small { font-size: 12px; color: rgba(148,163,184,0.9); }
      hr { border: none; border-top: 1px solid rgba(148,163,184,0.14); margin: 12px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Helpers
# -----------------------------
def normalize_base_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return "http://127.0.0.1:8000"
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "http://" + url
    return url.rstrip("/")


def safe_json(resp: requests.Response) -> Dict[str, Any]:
    try:
        return resp.json()
    except Exception:
        return {"error": f"Non-JSON response (status={resp.status_code})", "text": resp.text[:500]}


def api_get(
    base_url: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: Tuple[int, int] = (5, 120),
    retries: int = 2,
    backoff: float = 1.2,
) -> Tuple[int, Dict[str, Any]]:
    """
    Returns (status_code, json_dict)
    """
    url = f"{normalize_base_url(base_url)}{path}"
    last_err = None
    for i in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            return r.status_code, safe_json(r)
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            if i < retries:
                time.sleep(backoff * (i + 1))
    return 0, {"error": f"Request failed: {last_err}", "url": url}


@st.cache_data(ttl=10, show_spinner=False)
def cached_api_get(
    base_url: str,
    path: str,
    params_items: Tuple[Tuple[str, Any], ...],
    timeout: Tuple[int, int],
    retries: int,
) -> Tuple[int, Dict[str, Any]]:
    params = dict(params_items)
    return api_get(base_url, path, params=params, timeout=timeout, retries=retries)


def flatten_sessions(sessions: List[Dict[str, Any]], threshold: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for s in sessions or []:
        feats = s.get("features") or {}
        ip = s.get("client_ip") or s.get("ip") or ""
        ua = s.get("ua") or ""
        prob = s.get("bot_probability")
        try:
            prob_f = float(prob) if prob is not None else None
        except Exception:
            prob_f = None

        row = {
            "ip": ip,
            "ua": ua,
            "bot_probability": prob_f,
        }
        # add feature columns
        for k, v in feats.items():
            row[k] = v
        # derived label
        if prob_f is None:
            row["label"] = "unknown"
        else:
            row["label"] = "bot" if prob_f >= threshold else "real"

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # nicer ordering if columns exist
    preferred = [
        "label", "bot_probability", "ip", "ua",
        "req_count", "m3u8_count", "ts_count",
        "m3u8_ts_ratio", "rps", "duration_s",
        "gap_mean", "gap_p50", "gap_p95",
        "err_rate", "status_4xx", "status_5xx",
        "rt_avg", "rt_p95", "bytes_avg", "bytes_sum"
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    # sorting
    if "bot_probability" in df.columns:
        df = df.sort_values(by="bot_probability", ascending=False, na_position="last")
    return df


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("⚙️ Settings")

base_url = st.sidebar.text_input(
    "API Base URL",
    value=st.session_state.get("base_url", "http://139.59.166.217:8000"),
    help="Example: http://139.59.166.217:8000 (make sure port 8000 is reachable from your PC)",
)
base_url = normalize_base_url(base_url)
st.session_state["base_url"] = base_url

stream_key = st.sidebar.text_input("stream_key", value="/hls/NMT.m3u8")
window_seconds = st.sidebar.number_input("window_seconds", min_value=30, max_value=86400, value=7200, step=30)
tail_lines = st.sidebar.number_input("tail_lines", min_value=1000, max_value=5000000, value=800000, step=10000)
limit = st.sidebar.number_input("limit", min_value=1, max_value=20000, value=5000, step=50)

st.sidebar.markdown("---")
threshold = st.sidebar.slider("Bot threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
connect_timeout = st.sidebar.number_input("Connect timeout (s)", min_value=1, max_value=30, value=5, step=1)
read_timeout = st.sidebar.number_input("Read timeout (s)", min_value=5, max_value=600, value=120, step=5)
retries = st.sidebar.number_input("Retries", min_value=0, max_value=5, value=2, step=1)

st.sidebar.markdown("---")
refresh = st.sidebar.button("🔄 Refresh now")

# Filters
st.sidebar.subheader("Filters")
label_filter = st.sidebar.multiselect("Label", ["bot", "real", "unknown"], default=["bot", "real", "unknown"])
min_prob = st.sidebar.slider("Min probability", 0.0, 1.0, 0.0, 0.01)
search_ip = st.sidebar.text_input("Search IP contains", value="")
search_ua = st.sidebar.text_input("Search UA contains", value="")

# -----------------------------
# Header
# -----------------------------
st.title("📺 Nginx HLS Viewer + Bot Monitor")
st.caption("Model status, bot scoring per session, and live session table (Streamlit + Requests + Pandas).")

# -----------------------------
# Model status
# -----------------------------
colA, colB = st.columns([1, 1])

with colA:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Model status**")

    status_progress = st.progress(0)
    status_msg = st.empty()

    with st.spinner("Checking model status..."):
        status_progress.progress(20)
        code, model_status = cached_api_get(
            base_url,
            "/api/nginx/model/status",
            tuple([]),
            (int(connect_timeout), int(read_timeout)),
            int(retries),
        )
        status_progress.progress(70)

    if code == 200 and isinstance(model_status, dict) and model_status.get("model_loaded"):
        status_msg.success("Model loaded ✅")
        st.json(model_status)
    else:
        err = model_status.get("error") if isinstance(model_status, dict) else None
        status_msg.error(err or f"Could not load model status (HTTP {code}).")
        st.json(model_status)

    status_progress.progress(100)
    time.sleep(0.1)
    status_progress.empty()
    st.markdown("</div>", unsafe_allow_html=True)

with colB:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Quick tips if you see timeout**")
    st.markdown(
        """
        - If your dashboard runs on your PC and API runs on the server, **port 8000 must be open** and allowed in firewall.
        - Try smaller values first: `tail_lines=200000` and `limit=500`.
        - Increase **Read timeout** (sidebar) if log parsing takes time.
        """
    )
    st.markdown(f"<span class='pill'>API: {base_url}</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# Bot scoring fetch
# -----------------------------
st.subheader("XGBoost bot scoring per session")

bots_params = {
    "stream_key": stream_key,
    "window_seconds": int(window_seconds),
    "tail_lines": int(tail_lines),
    "limit": int(limit),
}

# Use refresh button to bust cache
if refresh:
    st.cache_data.clear()

progress = st.progress(0)
msg = st.empty()

msg.info("Loading bot scoring data...")
progress.progress(15)

with st.spinner("Fetching /api/nginx/bots/score ..."):
    bots_code, bots = cached_api_get(
        base_url,
        "/api/nginx/bots/score",
        tuple(sorted(bots_params.items(), key=lambda x: x[0])),
        (int(connect_timeout), int(read_timeout)),
        int(retries),
    )

progress.progress(60)

if bots_code != 200 or not isinstance(bots, dict) or bots.get("error"):
    progress.empty()
    err_text = (bots or {}).get("error") if isinstance(bots, dict) else None
    msg.error(err_text or f"Failed to fetch bot scoring (HTTP {bots_code}).")
    st.code(str(bots), language="json")
    st.stop()

msg.success("Loaded ✅")
progress.progress(100)
time.sleep(0.1)
progress.empty()
msg.empty()

summary = bots.get("summary") or {}
sessions = bots.get("sessions") or []

# -----------------------------
# Summary metrics
# -----------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Sessions scored", summary.get("sessions_scored", len(sessions)))
m2.metric("Likely bots", summary.get("likely_bots", "—"))
m3.metric("Likely real", summary.get("likely_real", "—"))
m4.metric("Threshold", f"{threshold:.2f}")

# -----------------------------
# Sessions table
# -----------------------------
df = flatten_sessions(sessions, threshold=threshold)

if df.empty:
    st.warning("No sessions returned. Try increasing tail_lines / window_seconds or confirm traffic exists.")
    st.stop()

# Apply filters
df_view = df.copy()

if "label" in df_view.columns:
    df_view = df_view[df_view["label"].isin(label_filter)]

if "bot_probability" in df_view.columns:
    df_view = df_view[df_view["bot_probability"].fillna(-1) >= float(min_prob)]

if search_ip:
    df_view = df_view[df_view["ip"].astype(str).str.contains(search_ip, case=False, na=False)]

if search_ua and "ua" in df_view.columns:
    df_view = df_view[df_view["ua"].astype(str).str.contains(search_ua, case=False, na=False)]

# Compact display columns
display_cols = [c for c in ["label", "bot_probability", "ip", "ua", "req_count", "m3u8_count", "ts_count", "err_rate", "status_4xx", "gap_mean"] if c in df_view.columns]
rest_cols = [c for c in df_view.columns if c not in display_cols]
df_display = df_view[display_cols + rest_cols]

# Show table
st.dataframe(df_display, use_container_width=True, hide_index=True)

# -----------------------------
# Drill-down: select a session
# -----------------------------
st.markdown("### Session details")
left, right = st.columns([1, 2])

with left:
    ip_options = df_view["ip"].dropna().astype(str).unique().tolist()
    ip_options = sorted(ip_options)
    selected_ip = st.selectbox("Select IP", ip_options, index=0 if ip_options else None)

with right:
    if selected_ip:
        row = df_view[df_view["ip"].astype(str) == str(selected_ip)].head(1)
        if not row.empty:
            r = row.iloc[0].to_dict()
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"**IP:** `{r.get('ip','')}`")
            st.markdown(f"**Label:** `{r.get('label','')}`  |  **bot_probability:** `{r.get('bot_probability')}`")
            st.markdown(f"**UA:** `{(r.get('ua') or '')[:220]}`")
            st.markdown("<hr/>", unsafe_allow_html=True)

            # Show key features nicely
            key_feats = [
                "req_count", "m3u8_count", "ts_count", "m3u8_ts_ratio",
                "duration_s", "rps", "gap_mean", "gap_p50", "gap_p95",
                "err_rate", "status_4xx", "status_5xx",
                "rt_avg", "rt_p95", "bytes_avg", "bytes_sum",
            ]
            feat_show = {k: r.get(k) for k in key_feats if k in r}

            c1, c2, c3 = st.columns(3)
            c1.metric("req_count", feat_show.get("req_count", "—"))
            c2.metric("m3u8_count", feat_show.get("m3u8_count", "—"))
            c3.metric("ts_count", feat_show.get("ts_count", "—"))

            st.markdown("**All features (flattened)**")
            st.json({k: v for k, v in r.items() if k not in ["ip", "ua"]})
            st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Quick probability chart (no extra deps)
# -----------------------------
if "bot_probability" in df_view.columns and df_view["bot_probability"].notna().any():
    st.markdown("---")
    st.markdown("### Probability snapshot")
    # Take top 30 by prob for readability
    top = df_view.sort_values("bot_probability", ascending=False).head(30)
    chart_df = top[["ip", "bot_probability"]].set_index("ip")
    st.bar_chart(chart_df)

st.caption("If you still see timeouts: try smaller tail_lines/limit first, then increase Read timeout. Also ensure the server allows inbound connections to port 8000.")
