# dashboard.py
# deps: streamlit, requests, pandas

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Stream Monitor", page_icon="📡", layout="wide")

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
  background: radial-gradient(1200px 600px at 15% 0%, rgba(59,130,246,0.14), transparent 55%),
              radial-gradient(1000px 500px at 85% 15%, rgba(168,85,247,0.14), transparent 55%),
              linear-gradient(180deg, rgba(2,6,23,1) 0%, rgba(3,7,18,1) 100%);
}

h1, h2, h3, h4 { letter-spacing: -0.02em; }
.small-muted { color: rgba(148,163,184,0.9); font-size: 12px; }
.muted { color: rgba(148,163,184,0.9); }

.card {
  border: 1px solid rgba(148,163,184,0.16);
  background: rgba(15,23,42,0.45);
  padding: 16px 18px;
  border-radius: 18px;
  box-shadow: 0 14px 40px rgba(0,0,0,0.28);
}
.card-soft {
  border: 1px solid rgba(148,163,184,0.14);
  background: rgba(2,6,23,0.35);
  padding: 14px 16px;
  border-radius: 16px;
}

.pill {
  display:inline-flex; align-items:center; gap:8px;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(148,163,184,0.18);
  background: rgba(2,6,23,0.35);
  font-size: 12px;
  color: rgba(226,232,240,0.92);
}
.dot { width:8px; height:8px; border-radius:999px; display:inline-block; }
.dot-ok { background: rgba(34,197,94,1); }
.dot-warn { background: rgba(245,158,11,1); }
.dot-bad { background: rgba(239,68,68,1); }

[data-testid="stDataFrame"] {
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid rgba(148,163,184,0.14);
}

/* Buttons */
.stButton > button {
  border-radius: 12px !important;
  border: 1px solid rgba(148,163,184,0.18) !important;
  background: rgba(15,23,42,0.55) !important;
}
.stButton > button:hover {
  border: 1px solid rgba(148,163,184,0.26) !important;
}

/* Make ONLY the primary sidebar button (RUN/WORKING) bigger */
div[data-testid="stSidebar"] button[kind="primary"] {
  font-size: 16px !important;
  padding: 0.9rem 1rem !important;
  border-radius: 14px !important;
  border: 1px solid rgba(148,163,184,0.25) !important;
  background: rgba(59,130,246,0.18) !important;
}
div[data-testid="stSidebar"] button[kind="primary"]:hover {
  background: rgba(59,130,246,0.25) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# State init (prevents auto-run)
# -----------------------------
def init_state() -> None:
    defaults = {
        # run control
        "has_run": False,          # becomes True after first successful run
        "trigger_run": False,      # set True when RUN clicked (then executed once)
        "is_working": False,       # True while run in progress
        "last_error": "",

        # inputs
        "base_url": "http://139.59.166.217:8000",
        "stream_key": "/hls/NMT.m3u8",
        "threshold": 0.7,
        "window_seconds": 7200,
        "tail_lines": 800000,
        "limit": 5000,
        "connect_timeout": 5,
        "read_timeout": 120,
        "retries": 2,
        "auto_refresh": "Off",

        # filters
        "label_filter": ["bot", "real", "unknown"],
        "min_prob": 0.0,
        "search_ip": "",
        "search_ua": "",

        # results (saved from last run)
        "model_status": None,   # {"code": int, "data": dict}
        "model_url": "",
        "bots_data": None,      # dict
        "bots_url": "",
        "last_updated_utc": None,
        "timing_history": [],   # list of dicts
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# -----------------------------
# Utilities
# -----------------------------
def rerun() -> None:
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


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
        return {"error": f"Non-JSON response (status={resp.status_code})", "text": resp.text[:800]}


def api_get(
    base_url: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: Tuple[int, int] = (5, 120),
    retries: int = 2,
) -> Tuple[int, Dict[str, Any], str]:
    url = f"{normalize_base_url(base_url)}{path}"
    last_err = ""
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            return r.status_code, safe_json(r), url
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            if attempt < retries:
                time.sleep(0.8 * (attempt + 1))
    return 0, {"error": f"Request failed: {last_err}"}, url


@st.cache_data(ttl=10, show_spinner=False)
def cached_get(
    base_url: str,
    path: str,
    params_items: Tuple[Tuple[str, Any], ...],
    timeout: Tuple[int, int],
    retries: int,
) -> Tuple[int, Dict[str, Any], str]:
    return api_get(base_url, path, params=dict(params_items), timeout=timeout, retries=retries)


def badge(text: str, state: str = "ok") -> str:
    cls = "dot-ok" if state == "ok" else ("dot-warn" if state == "warn" else "dot-bad")
    return f'<span class="pill"><span class="dot {cls}"></span>{text}</span>'


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

        row: Dict[str, Any] = {
            "label": ("unknown" if prob_f is None else ("bot" if prob_f >= threshold else "real")),
            "bot_probability": prob_f,
            "ip": ip,
            "ua": ua,
        }
        for k, v in feats.items():
            row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    preferred = [
        "label", "bot_probability", "ip", "ua",
        "req_count", "m3u8_count", "ts_count",
        "m3u8_ts_ratio", "rps", "duration_s",
        "gap_mean", "gap_p50", "gap_p95",
        "err_rate", "status_4xx", "status_5xx",
        "rt_avg", "rt_p95", "bytes_avg", "bytes_sum",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    if "bot_probability" in df.columns:
        df = df.sort_values("bot_probability", ascending=False, na_position="last")

    return df


def style_sessions(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def _row_style(row: pd.Series) -> List[str]:
        label = str(row.get("label", "")).lower()
        if label == "bot":
            return ["background-color: rgba(239,68,68,0.08)"] * len(row)
        if label == "real":
            return ["background-color: rgba(34,197,94,0.07)"] * len(row)
        return [""] * len(row)

    sty = df.style.apply(_row_style, axis=1)
    if "bot_probability" in df.columns:
        sty = sty.format({"bot_probability": "{:.6f}"})

    for c in ["rps", "gap_mean", "err_rate", "m3u8_ts_ratio", "rt_avg", "rt_p95"]:
        if c in df.columns:
            sty = sty.format({c: "{:.3f}"})
    for c in ["bytes_avg", "bytes_sum"]:
        if c in df.columns:
            sty = sty.format({c: "{:,.0f}"})
    for c in ["req_count", "m3u8_count", "ts_count", "status_4xx", "status_5xx", "duration_s"]:
        if c in df.columns:
            sty = sty.format({c: "{:,.0f}"})
    return sty


def build_prob_hist(df: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    if df.empty or "bot_probability" not in df.columns:
        return pd.DataFrame()
    s = df["bot_probability"].dropna().clip(0.0, 1.0)
    if s.empty:
        return pd.DataFrame()
    cats = pd.cut(s, bins=bins, include_lowest=True)
    counts = cats.value_counts().sort_index()
    return pd.DataFrame({"bin": counts.index.astype(str), "count": counts.values})


def feature_importance_from_payload(payload: Any) -> Optional[pd.DataFrame]:
    """
    Only works if your API includes:
      - feature_importance: {"rps": 0.42, ...}
      or
      - feature_importance: [{"feature":"rps","importance":0.42}, ...]
    """
    if not isinstance(payload, dict):
        return None
    fi = payload.get("feature_importance") or payload.get("feature_importances")
    if fi is None:
        return None

    if isinstance(fi, dict):
        df = pd.DataFrame([{"feature": k, "importance": v} for k, v in fi.items()])
    elif isinstance(fi, list):
        rows = []
        for item in fi:
            if isinstance(item, dict):
                f = item.get("feature") or item.get("name")
                imp = item.get("importance") or item.get("gain") or item.get("weight")
                if f is not None and imp is not None:
                    rows.append({"feature": f, "importance": imp})
        df = pd.DataFrame(rows)
    else:
        return None

    if df.empty:
        return None

    df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
    df = df.dropna().sort_values("importance", ascending=False)
    return df if not df.empty else None


# -----------------------------
# HTML report helpers
# -----------------------------
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


def json_pretty(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)


def df_to_html_table(df: pd.DataFrame, max_rows: int = 500) -> str:
    if df is None or df.empty:
        return "<div class='muted'>No data.</div>"
    return df.head(max_rows).to_html(index=False, escape=True, classes="tbl")


def build_score_hist_rows(df: pd.DataFrame, bins: int = 20) -> List[Dict[str, Any]]:
    if df.empty or "bot_probability" not in df.columns:
        return []
    s = df["bot_probability"].dropna().clip(0.0, 1.0)
    if s.empty:
        return []
    cats = pd.cut(s, bins=bins, include_lowest=True)
    counts = cats.value_counts().sort_index()
    return [{"bin": str(k), "count": int(v)} for k, v in counts.items()]


def make_html_report(
    *,
    base_url: str,
    stream_key: str,
    params: Dict[str, Any],
    threshold: float,
    model_status_payload: Dict[str, Any],
    model_status_code: int,
    model_url: str,
    bots_payload: Dict[str, Any],
    bots_url: str,
    last_updated_utc: str,
    df_sessions: pd.DataFrame,
    max_rows: int = 500,
) -> str:
    summary = (bots_payload or {}).get("summary") or {}

    top_bots = pd.DataFrame()
    top_real = pd.DataFrame()
    if df_sessions is not None and not df_sessions.empty and "bot_probability" in df_sessions.columns:
        top_bots = (
            df_sessions[df_sessions["label"] == "bot"]
            .sort_values("bot_probability", ascending=False)
            .head(20)
        )
        top_real = (
            df_sessions[df_sessions["label"] == "real"]
            .sort_values("bot_probability", ascending=True)
            .head(20)
        )

    hist_rows = build_score_hist_rows(df_sessions, bins=20)
    max_count = max([r["count"] for r in hist_rows], default=1)

    if hist_rows:
        hist_html = "\n".join(
            f"""
            <div class="hist-row">
              <div class="hist-bin">{r["bin"]}</div>
              <div class="hist-bar-wrap">
                <div class="hist-bar" style="width:{(r["count"]/max_count)*100:.1f}%"></div>
              </div>
              <div class="hist-count">{r["count"]}</div>
            </div>
            """
            for r in hist_rows
        )
    else:
        hist_html = "<div class='muted'>No bot_probability values to build histogram.</div>"

    def pick_cols(df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in ["ip", "bot_probability", "label", "req_count", "m3u8_count", "ts_count", "rps", "err_rate"] if c in df.columns]
        return df[cols] if cols else df

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Stream Monitor Report</title>
  <style>
    body {{
      margin:0; padding:24px;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
      background: #020617;
      color: #e2e8f0;
    }}
    .muted {{ color: rgba(148,163,184,0.9); }}
    .wrap {{ max-width: 1200px; margin: 0 auto; }}
    .card {{
      border: 1px solid rgba(148,163,184,0.16);
      background: rgba(15,23,42,0.55);
      padding: 16px 18px;
      border-radius: 18px;
      box-shadow: 0 14px 40px rgba(0,0,0,0.28);
      margin-bottom: 14px;
    }}
    h1 {{ margin:0 0 6px 0; letter-spacing:-0.02em; }}
    h2 {{ margin:0 0 10px 0; letter-spacing:-0.02em; font-size:18px; }}
    .kpi {{
      display:flex; gap:10px; flex-wrap:wrap;
      margin-top: 10px;
    }}
    .pill {{
      display:inline-flex; align-items:center; gap:8px;
      padding: 6px 10px; border-radius: 999px;
      border: 1px solid rgba(148,163,184,0.18);
      background: rgba(2,6,23,0.35);
      font-size: 12px;
      color: rgba(226,232,240,0.92);
    }}
    .tbl {{
      width:100%;
      border-collapse: collapse;
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid rgba(148,163,184,0.14);
    }}
    .tbl th, .tbl td {{
      text-align:left;
      padding: 10px 10px;
      border-bottom: 1px solid rgba(148,163,184,0.12);
      vertical-align: top;
      font-size: 12px;
    }}
    .tbl th {{
      background: rgba(2,6,23,0.55);
      position: sticky;
      top: 0;
    }}
    pre {{
      background: rgba(2,6,23,0.55);
      border: 1px solid rgba(148,163,184,0.14);
      padding: 12px;
      border-radius: 14px;
      color: #e2e8f0;
      overflow:auto;
    }}
    .hist-row {{
      display:flex; align-items:center; gap:10px;
      margin: 8px 0;
    }}
    .hist-bin {{ width: 210px; font-size: 12px; color: rgba(148,163,184,0.9); }}
    .hist-bar-wrap {{
      flex: 1; height: 10px;
      background: rgba(148,163,184,0.10);
      border-radius: 999px; overflow:hidden;
    }}
    .hist-bar {{
      height: 10px;
      background: rgba(59,130,246,0.75);
      border-radius: 999px;
    }}
    .hist-count {{ width: 60px; text-align:right; font-size: 12px; }}
    .grid2 {{ display:grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    @media (max-width: 900px) {{
      .grid2 {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Stream Monitor Report</h1>
      <div class="muted">Generated: {last_updated_utc} UTC</div>
      <div class="kpi">
        <span class="pill">API: {base_url}</span>
        <span class="pill">stream_key: {stream_key}</span>
        <span class="pill">threshold: {threshold:.2f}</span>
      </div>
    </div>

    <div class="card">
      <h2>Run Parameters</h2>
      <pre>{json_pretty(params)}</pre>
    </div>

    <div class="card">
      <h2>Model Status</h2>
      <div class="muted">Status URL: {model_url} (HTTP {model_status_code})</div>
      <pre>{json_pretty(model_status_payload)}</pre>
    </div>

    <div class="card">
      <h2>Scoring Summary</h2>
      <div class="muted">Score URL: {bots_url}</div>
      <div class="kpi">
        <span class="pill">sessions: {len(df_sessions):,}</span>
        <span class="pill">likely_bots: {int(summary.get("likely_bots", 0)):,}</span>
        <span class="pill">likely_real: {int(summary.get("likely_real", 0)):,}</span>
        <span class="pill">sessions_scored: {int(summary.get("sessions_scored", len(df_sessions))):,}</span>
      </div>
    </div>

    <div class="card">
      <h2>Bot Score Distribution (bot_probability)</h2>
      {hist_html}
    </div>

    <div class="grid2">
      <div class="card">
        <h2>Top suspected bots</h2>
        {df_to_html_table(pick_cols(top_bots), max_rows=20)}
      </div>
      <div class="card">
        <h2>Top likely real</h2>
        {df_to_html_table(pick_cols(top_real), max_rows=20)}
      </div>
    </div>

    <div class="card">
      <h2>All Sessions (first {max_rows} rows)</h2>
      <div class="muted">Big max_rows makes big HTML.</div>
      {df_to_html_table(df_sessions, max_rows=max_rows)}
    </div>

    <div class="card">
      <h2>Raw JSON (bots/score)</h2>
      <pre>{json_pretty(bots_payload)}</pre>
    </div>

  </div>
</body>
</html>
"""
    return html


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Monitor")

base_url = st.sidebar.text_input("API Base URL", value=st.session_state["base_url"])
stream_key = st.sidebar.text_input("stream_key", value=st.session_state["stream_key"])

with st.sidebar.expander("Scoring Window", expanded=False):
    threshold = st.slider("Bot threshold", 0.0, 1.0, float(st.session_state["threshold"]), 0.05)
    window_seconds = st.number_input("window_seconds", 30, 86400, int(st.session_state["window_seconds"]), 30)
    tail_lines = st.number_input("tail_lines", 5000, 5000000, int(st.session_state["tail_lines"]), 5000)
    limit = st.number_input("limit", 1, 20000, int(st.session_state["limit"]), 50)

with st.sidebar.expander("Filters", expanded=False):
    label_filter = st.multiselect("Label", ["bot", "real", "unknown"], default=st.session_state["label_filter"])
    min_prob = st.slider("Min probability", 0.0, 1.0, float(st.session_state["min_prob"]), 0.01)
    search_ip = st.text_input("IP contains", value=st.session_state["search_ip"])
    search_ua = st.text_input("UA contains", value=st.session_state["search_ua"])

with st.sidebar.expander("Advanced", expanded=False):
    connect_timeout = st.number_input("Connect timeout (s)", 1, 30, int(st.session_state["connect_timeout"]), 1)
    read_timeout = st.number_input("Read timeout (s)", 5, 600, int(st.session_state["read_timeout"]), 5)
    retries = st.number_input("Retries", 0, 5, int(st.session_state["retries"]), 1)
    auto_refresh = st.selectbox(
        "Auto refresh",
        ["Off", "10s", "20s", "30s"],
        index=["Off", "10s", "20s", "30s"].index(st.session_state["auto_refresh"]),
    )

show_debug = st.sidebar.toggle("Debug", value=False)

# persist inputs
st.session_state.update(
    {
        "base_url": normalize_base_url(base_url),
        "stream_key": stream_key,
        "threshold": float(threshold),
        "window_seconds": int(window_seconds),
        "tail_lines": int(tail_lines),
        "limit": int(limit),
        "label_filter": label_filter,
        "min_prob": float(min_prob),
        "search_ip": search_ip,
        "search_ua": search_ua,
        "connect_timeout": int(connect_timeout),
        "read_timeout": int(read_timeout),
        "retries": int(retries),
        "auto_refresh": auto_refresh,
    }
)

# RUN button (NO API calls happen on first page load)
if st.session_state["is_working"]:
    st.sidebar.button("⏳ WORKING ON...", type="primary", use_container_width=True, disabled=True)
else:
    if st.sidebar.button("▶ RUN", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.session_state["trigger_run"] = True
        st.session_state["is_working"] = True
        st.session_state["last_error"] = ""
        rerun()

# -----------------------------
# Header (NO API calls here)
# -----------------------------
top_l, top_r = st.columns([0.72, 0.28])

with top_l:
    st.markdown(
        f"""
        <div class="card">
          <div style="font-size:28px; font-weight:800;">Stream Monitor</div>
          <div class="muted">HLS traffic + XGBoost bot scoring (per session)</div>
          <div style="margin-top:10px; display:flex; gap:10px; flex-wrap:wrap;">
            {badge(f"API: {st.session_state['base_url']}", "warn")}
            {badge(f"stream_key: {st.session_state['stream_key']}", "warn")}
            {badge(f"threshold: {st.session_state['threshold']:.2f}", "warn")}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_r:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Connection & Model**")

    if not st.session_state["has_run"]:
        st.markdown(badge("Not checked yet", "warn"), unsafe_allow_html=True)
        st.caption("Click **RUN** to load model status + scoring.")
    else:
        ms = st.session_state.get("model_status") or {}
        code_s = int(ms.get("code") or 0)
        model_payload = ms.get("data") or {}

        model_loaded = bool(isinstance(model_payload, dict) and model_payload.get("model_loaded"))
        xgb_ok = bool(isinstance(model_payload, dict) and model_payload.get("xgb_available"))

        if code_s == 200 and model_loaded and xgb_ok:
            st.markdown(badge("Model loaded (XGB OK)", "ok"), unsafe_allow_html=True)
        elif code_s == 200 and model_loaded:
            st.markdown(badge("Model loaded (XGB issue)", "warn"), unsafe_allow_html=True)
        else:
            st.markdown(badge("Model status error", "bad"), unsafe_allow_html=True)

        st.markdown(
            f"<div class='small-muted' style='margin-top:8px;'>Status URL: {st.session_state.get('model_url','')}</div>",
            unsafe_allow_html=True,
        )

        if show_debug:
            st.markdown("**Raw model status**")
            st.json(model_payload)

    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.get("last_error"):
    st.error(st.session_state["last_error"])

st.markdown("")

# -----------------------------
# Progress UI (shows only while running)
# -----------------------------
progress_wrap = st.container()
progress_ph = progress_wrap.empty()
progress_text_ph = progress_wrap.empty()


def do_run_once() -> None:
    """
    Single run:
      1) GET /api/nginx/model/status
      2) GET /api/nginx/bots/score
      3) Process sessions
    Saves everything to st.session_state.
    """
    base_url_ = st.session_state["base_url"]
    stream_key_ = st.session_state["stream_key"]
    threshold_ = st.session_state["threshold"]
    timeout_ = (int(st.session_state["connect_timeout"]), int(st.session_state["read_timeout"]))
    retries_ = int(st.session_state["retries"])

    params = {
        "stream_key": stream_key_,
        "window_seconds": int(st.session_state["window_seconds"]),
        "tail_lines": int(st.session_state["tail_lines"]),
        "limit": int(st.session_state["limit"]),
    }

    t0 = time.perf_counter()
    timings = {"model": 0.0, "score": 0.0, "process": 0.0, "total": 0.0}

    pb = progress_ph.progress(0)
    progress_text_ph.info("Starting...")

    try:
        # Step 1: model status
        pb.progress(15)
        progress_text_ph.info("1/3 Checking model status...")
        t1 = time.perf_counter()
        code_s, model_status, model_url = cached_get(base_url_, "/api/nginx/model/status", tuple([]), timeout_, retries_)
        timings["model"] = time.perf_counter() - t1
        st.session_state["model_status"] = {"code": code_s, "data": model_status}
        st.session_state["model_url"] = model_url

        # Step 2: scoring data
        pb.progress(45)
        progress_text_ph.info("2/3 Loading scoring data...")
        t2 = time.perf_counter()
        code_b, bots, bots_url = cached_get(
            base_url_,
            "/api/nginx/bots/score",
            tuple(sorted(params.items(), key=lambda x: x[0])),
            timeout_,
            retries_,
        )
        timings["score"] = time.perf_counter() - t2

        if code_b != 200 or not isinstance(bots, dict) or bots.get("error"):
            msg = f"Failed to load scoring data (HTTP {code_b})."
            st.session_state["last_error"] = msg
            pb.progress(100)
            progress_text_ph.error(msg)
            st.session_state["bots_data"] = None
            st.session_state["bots_url"] = bots_url
            return

        st.session_state["bots_data"] = bots
        st.session_state["bots_url"] = bots_url

        # Step 3: process
        pb.progress(80)
        progress_text_ph.info("3/3 Processing...")
        t3 = time.perf_counter()
        _ = flatten_sessions(bots.get("sessions") or [], threshold=threshold_)
        timings["process"] = time.perf_counter() - t3

        pb.progress(100)
        progress_text_ph.success("Done ✅")
        st.session_state["has_run"] = True
        st.session_state["last_updated_utc"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    except Exception as e:
        st.session_state["last_error"] = f"Run error: {e}"
    finally:
        timings["total"] = time.perf_counter() - t0
        hist = st.session_state.get("timing_history") or []
        hist.append(timings)
        st.session_state["timing_history"] = hist[-20:]

        time.sleep(0.15)
        progress_ph.empty()
        progress_text_ph.empty()


# -----------------------------
# Execute run only when RUN clicked
# -----------------------------
if st.session_state["trigger_run"]:
    st.session_state["trigger_run"] = False
    do_run_once()
    st.session_state["is_working"] = False
    rerun()

# -----------------------------
# Tabs (use saved results; no API calls here)
# -----------------------------
tabs = st.tabs(["📊 Overview", "🧾 Sessions", "🧠 XGBoost", "🛠️ Diagnostics"])


def require_data() -> Optional[Dict[str, Any]]:
    if not st.session_state["has_run"] or not st.session_state.get("bots_data"):
        st.info("Click **▶ RUN** in the sidebar to load data.")
        return None
    return st.session_state["bots_data"]


# -----------------------------
# Overview
# -----------------------------
with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Overview")

    bots = require_data()
    if not bots:
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    summary = bots.get("summary") or {}
    sessions = bots.get("sessions") or []
    df = flatten_sessions(sessions, threshold=st.session_state["threshold"])

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Sessions scored", int(summary.get("sessions_scored", len(sessions))))
    k2.metric("Likely bots", int(summary.get("likely_bots", int((df["label"] == "bot").sum()) if not df.empty else 0)))
    k3.metric("Likely real", int(summary.get("likely_real", int((df["label"] == "real").sum()) if not df.empty else 0)))
    k4.metric("Window (s)", int(st.session_state["window_seconds"]))
    k5.metric("Tail lines", int(st.session_state["tail_lines"]))

    last_u = st.session_state.get("last_updated_utc")
    if last_u:
        st.markdown(f"<div class='small-muted'>Last updated: {last_u} UTC</div>", unsafe_allow_html=True)

    if not df.empty and "bot_probability" in df.columns:
        top_bots = df[df["label"] == "bot"].head(10)
        top_real = df[df["label"] == "real"].sort_values("bot_probability", ascending=True).head(10)

        cA, cB = st.columns(2)
        with cA:
            st.markdown('<div class="card-soft">', unsafe_allow_html=True)
            st.markdown("**Top suspected bots**")
            if top_bots.empty:
                st.caption("No bots above threshold in this window.")
            else:
                cols = [c for c in ["ip", "bot_probability", "req_count", "m3u8_count", "ts_count"] if c in top_bots.columns]
                st.dataframe(top_bots[cols].reset_index(drop=True), use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with cB:
            st.markdown('<div class="card-soft">', unsafe_allow_html=True)
            st.markdown("**Top likely real viewers**")
            if top_real.empty:
                st.caption("No real viewers found in this window.")
            else:
                cols = [c for c in ["ip", "bot_probability", "req_count", "m3u8_count", "ts_count"] if c in top_real.columns]
                st.dataframe(top_real[cols].reset_index(drop=True), use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Sessions
# -----------------------------
with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Sessions")

    bots = require_data()
    if not bots:
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    df = flatten_sessions(bots.get("sessions") or [], threshold=st.session_state["threshold"])
    if df.empty:
        st.warning("No sessions returned. Increase tail_lines/window_seconds or confirm traffic exists.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    # filters
    df_view = df.copy()
    df_view = df_view[df_view["label"].isin(st.session_state["label_filter"])]
    if "bot_probability" in df_view.columns:
        df_view = df_view[df_view["bot_probability"].fillna(-1) >= float(st.session_state["min_prob"])]
    if st.session_state["search_ip"]:
        df_view = df_view[df_view["ip"].astype(str).str.contains(st.session_state["search_ip"], case=False, na=False)]
    if st.session_state["search_ua"] and "ua" in df_view.columns:
        df_view = df_view[df_view["ua"].astype(str).str.contains(st.session_state["search_ua"], case=False, na=False)]

    c1, c2, c3 = st.columns([0.5, 0.25, 0.25])
    with c1:
        st.caption(f"Showing {len(df_view):,} sessions (filtered) out of {len(df):,}")
    with c3:
        st.download_button(
            "⬇️ Download CSV",
            data=df_view.to_csv(index=False).encode("utf-8"),
            file_name="sessions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.dataframe(style_sessions(df_view), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Session details")
    left, right = st.columns([0.33, 0.67])

    with left:
        ip_list = sorted(df_view["ip"].dropna().astype(str).unique().tolist())
        selected_ip = st.selectbox("Select IP", ip_list, index=0 if ip_list else None)

    with right:
        if selected_ip:
            row = df_view[df_view["ip"].astype(str) == str(selected_ip)].head(1)
            if not row.empty:
                r = row.iloc[0].to_dict()
                state = "ok" if r.get("label") == "real" else ("bad" if r.get("label") == "bot" else "warn")
                st.markdown(
                    f"""
                    <div class="card-soft">
                      <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div style="font-weight:800; font-size:16px;">{selected_ip}</div>
                        {badge(f"label: {r.get('label','')}", state)}
                      </div>
                      <div class="small-muted" style="margin-top:8px;">UA: {str(r.get('ua',''))[:240]}</div>
                      <div class="small-muted" style="margin-top:8px;">
                        Decision: bot_probability ≥ {st.session_state['threshold']:.2f} → bot
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                mA, mB, mC, mD = st.columns(4)
                mA.metric("bot_probability", f"{(r.get('bot_probability') or 0):.6f}")
                mB.metric("req_count", int(r.get("req_count", 0) or 0))
                mC.metric("m3u8_count", int(r.get("m3u8_count", 0) or 0))
                mD.metric("ts_count", int(r.get("ts_count", 0) or 0))

                if show_debug:
                    st.markdown("**Raw session JSON**")
                    st.json(r)

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# XGBoost
# -----------------------------
with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧠 XGBoost details")

    bots = require_data()
    if not bots:
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    df = flatten_sessions(bots.get("sessions") or [], threshold=st.session_state["threshold"])

    st.markdown(
        """
- **bot_probability** is the XGBoost model output (0 → 1).
- Higher = more bot-like (based on your session feature patterns).
- Label rule:
  - `bot_probability ≥ threshold` → **bot**
  - otherwise → **real**
"""
    )

    ms = st.session_state.get("model_status") or {}
    code_s = int(ms.get("code") or 0)
    model_payload = ms.get("data") or {}

    model_loaded = bool(isinstance(model_payload, dict) and model_payload.get("model_loaded"))
    xgb_ok = bool(isinstance(model_payload, dict) and model_payload.get("xgb_available"))

    a, b = st.columns([0.55, 0.45])
    with a:
        st.markdown('<div class="card-soft">', unsafe_allow_html=True)
        st.markdown("**Model status (from last run)**")
        st.markdown(badge(f"status_code: {code_s}", "ok" if code_s == 200 else "bad"), unsafe_allow_html=True)
        st.markdown(badge(f"model_loaded: {model_loaded}", "ok" if model_loaded else "bad"), unsafe_allow_html=True)
        st.markdown(badge(f"xgb_available: {xgb_ok}", "ok" if xgb_ok else "warn"), unsafe_allow_html=True)
        st.caption(f"Status URL: {st.session_state.get('model_url','')}")
        if show_debug:
            st.json(model_payload)
        st.markdown("</div>", unsafe_allow_html=True)

    with b:
        st.markdown('<div class="card-soft">', unsafe_allow_html=True)
        st.markdown("**Feature importance**")
        fi = feature_importance_from_payload(model_payload)
        if fi is None:
            st.caption("Not provided by your API yet.")
            st.markdown(
                """
To show this, return from `/api/nginx/model/status` (or new endpoint):
- `feature_importance: {"rps": 0.42, "gap_mean": 0.19, ...}`
or
- `feature_importance: [{"feature":"rps","importance":0.42}, ...]`
"""
            )
        else:
            st.dataframe(fi.head(20), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")

    st.markdown('<div class="card-soft">', unsafe_allow_html=True)
    st.markdown("## Bot score summary")

    if df.empty or "bot_probability" not in df.columns:
        st.warning("No bot_probability values found.")
    else:
        s = df["bot_probability"].dropna()
        bots_n = int((df["label"] == "bot").sum())
        real_n = int((df["label"] == "real").sum())
        unk_n = int((df["label"] == "unknown").sum())

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Threshold", f"{st.session_state['threshold']:.2f}")
        k2.metric("Sessions", f"{len(df):,}")
        k3.metric("Bots", f"{bots_n:,}")
        k4.metric("Real", f"{real_n:,}")
        k5.metric("Unknown", f"{unk_n:,}")

        if not s.empty:
            p50 = float(s.quantile(0.50))
            p95 = float(s.quantile(0.95))
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Min", f"{float(s.min()):.4f}")
            m2.metric("Mean", f"{float(s.mean()):.4f}")
            m3.metric("P50", f"{p50:.4f}")
            m4.metric("P95", f"{p95:.4f}")
            m5.metric("Max", f"{float(s.max()):.4f}")

        st.markdown("**Score distribution**")
        hist_df = build_prob_hist(df, bins=20)
        if not hist_df.empty:
            st.bar_chart(hist_df.set_index("bin")["count"])
        else:
            st.caption("Not enough points to draw histogram.")

        st.markdown("")
        left, right = st.columns(2)
        with left:
            st.markdown("**Top suspected bots (highest scores)**")
            top_bots = df.sort_values("bot_probability", ascending=False).head(15)
            cols = [c for c in ["ip", "bot_probability", "label", "req_count", "m3u8_count", "ts_count", "rps", "err_rate"] if c in top_bots.columns]
            st.dataframe(top_bots[cols].reset_index(drop=True), use_container_width=True, hide_index=True)

        with right:
            st.markdown("**Top likely real (lowest scores)**")
            top_real = df.sort_values("bot_probability", ascending=True).head(15)
            cols = [c for c in ["ip", "bot_probability", "label", "req_count", "m3u8_count", "ts_count", "rps", "err_rate"] if c in top_real.columns]
            st.dataframe(top_real[cols].reset_index(drop=True), use_container_width=True, hide_index=True)

        st.markdown("")
        st.markdown("**Feature list (inferred from payload)**")
        inferred = [c for c in df.columns if c not in ["label", "bot_probability", "ip", "ua"]]
        if inferred:
            st.dataframe(pd.DataFrame({"feature": inferred}), use_container_width=True, hide_index=True)
        else:
            st.caption("No features found in this payload.")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Diagnostics + Export HTML
# -----------------------------
with tabs[3]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Diagnostics")

    st.markdown("**Current request**")
    st.code(
        f"GET {st.session_state['base_url']}/api/nginx/bots/score"
        f"?stream_key={st.session_state['stream_key']}"
        f"&window_seconds={st.session_state['window_seconds']}"
        f"&tail_lines={st.session_state['tail_lines']}"
        f"&limit={st.session_state['limit']}",
        language="text",
    )

    st.markdown("**Timeout help**")
    st.markdown(
        """
- Reduce: `tail_lines=200000`, `limit=500`
- Increase **Read timeout** to 240–300s
- Ensure server allows inbound port **8000**
"""
    )

    if st.session_state["has_run"]:
        st.markdown("**Last URLs**")
        st.caption(f"Model: {st.session_state.get('model_url','')}")
        st.caption(f"Score: {st.session_state.get('bots_url','')}")
        if show_debug:
            st.markdown("**Raw payloads**")
            st.json(
                {
                    "model_status": st.session_state.get("model_status"),
                    "bots_data": st.session_state.get("bots_data"),
                }
            )

    st.markdown("---")
    st.markdown("### Export HTML report")

    if not st.session_state["has_run"] or not st.session_state.get("bots_data"):
        st.caption("Run first to generate a report.")
    else:
        max_rows_report = st.number_input("Max rows in HTML (sessions table)", 100, 20000, 500, 100)
        if st.button("🧾 Build HTML report", use_container_width=True):
            bots_payload = st.session_state["bots_data"]
            ms = st.session_state.get("model_status") or {}
            model_payload = ms.get("data") or {}
            model_code = int(ms.get("code") or 0)

            df_all = flatten_sessions(bots_payload.get("sessions") or [], threshold=st.session_state["threshold"])

            report_html = make_html_report(
                base_url=st.session_state["base_url"],
                stream_key=st.session_state["stream_key"],
                params={
                    "stream_key": st.session_state["stream_key"],
                    "window_seconds": st.session_state["window_seconds"],
                    "tail_lines": st.session_state["tail_lines"],
                    "limit": st.session_state["limit"],
                    "threshold": st.session_state["threshold"],
                },
                threshold=st.session_state["threshold"],
                model_status_payload=model_payload if isinstance(model_payload, dict) else {"raw": str(model_payload)},
                model_status_code=model_code,
                model_url=st.session_state.get("model_url", ""),
                bots_payload=bots_payload if isinstance(bots_payload, dict) else {"raw": str(bots_payload)},
                bots_url=st.session_state.get("bots_url", ""),
                last_updated_utc=st.session_state.get("last_updated_utc") or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                df_sessions=df_all,
                max_rows=int(max_rows_report),
            )

            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"stream_report_{ts}.html"
            filepath = os.path.join(REPORTS_DIR, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report_html)

            st.success(f"Report generated: {filepath}")

            st.download_button(
                "⬇️ Download HTML report",
                data=report_html.encode("utf-8"),
                file_name=filename,
                mime="text/html",
                use_container_width=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Auto-refresh (ONLY after first manual run; never on first load)
# -----------------------------
if (
    st.session_state["auto_refresh"] != "Off"
    and st.session_state["has_run"]
    and not st.session_state["is_working"]
):
    seconds = int(st.session_state["auto_refresh"].replace("s", ""))
    time.sleep(seconds)
    st.cache_data.clear()
    st.session_state["trigger_run"] = True
    st.session_state["is_working"] = True
    rerun()
