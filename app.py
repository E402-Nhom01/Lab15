"""
Streamlit UI for the AI Eval Factory (Lab 14/15).

Layout:
    - Left:  chat with the V2 agent (SimpleRAG + rerank + GPT-4o-mini)
    - Right: red-accent monitor panel (system status, live pipeline trace,
             session metrics, benchmark aggregate from reports/summary.json)

Run:
    export OPENAI_API_KEY=...
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import altair as alt
import pandas as pd
import streamlit as st

# Bridge Streamlit Cloud secrets -> os.environ so downstream libs (openai, etc.)
# pick them up. Must run BEFORE importing MainAgent (which constructs the LLM
# client at import time via its dependencies).
for _k in ("OPENAI_API_KEY", "COST_PER_1K_TOKENS"):
    try:
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = str(st.secrets[_k])
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        # Local run without secrets.toml — fine, env vars will be used.
        break

from agent.main_agent import MainAgent
# main_agent inserts simple-rag into sys.path on import, so this works after it.
from rag.prompt import ANSWER_PROMPT  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = REPO_ROOT / "reports"
COST_PER_1K_TOKENS = float(os.getenv("COST_PER_1K_TOKENS", "0.002"))
MAX_QUERIES_PER_SESSION = int(os.getenv("MAX_QUERIES_PER_SESSION", "20"))


def _get_secret(key: str, default: str = "") -> str:
    """Read from st.secrets first, fall back to env var. Safe when no secrets file."""
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        pass
    return os.getenv(key, default)


def password_gate() -> None:
    """Block the app behind a shared password if APP_PASSWORD secret is set.

    If no password is configured, the gate is a no-op (open access).
    """
    expected = _get_secret("APP_PASSWORD")
    if not expected:
        return
    if st.session_state.get("_auth_ok"):
        return

    st.markdown(
        "<h2 style='text-align:center;margin-top:80px;'>🔒 AI Eval Factory</h2>"
        "<p style='text-align:center;color:#94a3b8;'>Nhập password để truy cập demo.</p>",
        unsafe_allow_html=True,
    )
    with st.form("auth_form", clear_on_submit=False):
        pwd = st.text_input("Password", type="password", label_visibility="collapsed")
        ok = st.form_submit_button("Vào")
    if ok:
        if pwd == expected:
            st.session_state["_auth_ok"] = True
            st.rerun()
        else:
            st.error("Sai password.")
    st.stop()


# ---------------------------------------------------------------------------
# Agent bootstrap (cached so PDF parse + rerank model load once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Khởi tạo Agent V2 (load PDF, BM25, rerank)...")
def load_agent() -> MainAgent:
    return MainAgent(version="v2")


# ---------------------------------------------------------------------------
# One-shot traced query: reruns the V2 pipeline inline so we can time each
# stage and push live updates into the monitor panel.
# ---------------------------------------------------------------------------
def run_traced(agent: MainAgent, question: str, on_step) -> Dict[str, Any]:
    stages: Dict[str, Dict[str, Any]] = {}

    on_step("retrieval", "running", {})
    t0 = time.perf_counter()
    retrieved = agent.retrieval.retrieve(question, top_k=agent.retrieval_top_k)
    dt_retrieval = time.perf_counter() - t0
    retrieved_ids = [r["chunk_id"] for r in retrieved]
    stages["retrieval"] = {
        "ms": int(dt_retrieval * 1000),
        "top_k": agent.retrieval_top_k,
        "chunks": retrieved_ids,
    }
    on_step("retrieval", "done", stages["retrieval"])

    docs = [r["chunk_text"] for r in retrieved]
    meta = [
        {"chunk_id": r["chunk_id"], "source_document": r["source_document"]}
        for r in retrieved
    ]
    final_docs, final_meta = docs, meta

    if agent.rerank and docs:
        on_step("rerank", "running", {})
        t0 = time.perf_counter()
        reranked_docs, _ = agent.rerank.rerank(
            question, docs, top_k=min(agent.rerank_top_k, len(docs)), metadata=meta
        )
        dt_rerank = time.perf_counter() - t0
        doc_to_meta = {d: m for d, m in zip(docs, meta)}
        final_docs = reranked_docs
        final_meta = [doc_to_meta[d] for d in reranked_docs if d in doc_to_meta]
        stages["rerank"] = {
            "ms": int(dt_rerank * 1000),
            "top_k": agent.rerank_top_k,
            "chunks": [m["chunk_id"] for m in final_meta],
        }
        on_step("rerank", "done", stages["rerank"])
    else:
        stages["rerank"] = {"ms": 0, "top_k": 0, "chunks": [], "skipped": True}
        on_step("rerank", "skipped", stages["rerank"])

    on_step("llm", "running", {})
    t0 = time.perf_counter()
    prompt = ANSWER_PROMPT.format(query=question, context="\n".join(final_docs))
    answer = agent.llm.generate(prompt)
    dt_llm = time.perf_counter() - t0
    est_tokens = max(1, int((len(prompt) + len(answer)) / 4))  # ~chars/4
    stages["llm"] = {
        "ms": int(dt_llm * 1000),
        "model": "gpt-4o-mini",
        "est_tokens": est_tokens,
    }
    on_step("llm", "done", stages["llm"])

    faithfulness = _faithfulness(answer, final_docs)
    stages["faithfulness"] = {"value": faithfulness}
    on_step("faithfulness", "done", stages["faithfulness"])

    total_ms = sum(s.get("ms", 0) for s in stages.values() if isinstance(s.get("ms"), int))
    return {
        "answer": answer,
        "chunks": [m["chunk_id"] for m in final_meta],
        "contexts": final_docs,
        "stages": stages,
        "total_ms": total_ms,
        "est_tokens": est_tokens,
        "est_cost_usd": (est_tokens / 1000) * COST_PER_1K_TOKENS,
    }


def _faithfulness(answer: str, contexts: List[str]) -> float:
    """Cheap lexical faithfulness: |ans ∩ ctx| / |ans|. Range 0..1."""

    def norm(t: str) -> set:
        t = t.lower()
        for ch in ".,!?:;()[]{}\"'":
            t = t.replace(ch, " ")
        return {tok for tok in t.split() if tok}

    atoks = norm(answer)
    if not atoks:
        return 0.0
    ctoks = norm(" ".join(contexts))
    return round(len(atoks & ctoks) / len(atoks), 3)


# ---------------------------------------------------------------------------
# Aggregate benchmark metrics from reports/summary.json (if run previously)
# ---------------------------------------------------------------------------
def load_summary() -> Dict[str, Any] | None:
    fp = REPORTS_DIR / "summary.json"
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("messages", [])  # [{role, content, trace?}]
    ss.setdefault(
        "session_metrics",
        {"queries": 0, "total_ms": 0, "total_tokens": 0, "total_cost": 0.0, "faith_sum": 0.0},
    )
    ss.setdefault("latency_history", [])  # list of ints (ms) per query
    ss.setdefault("faith_history", [])    # list of floats per query
    ss.setdefault("query_history", [])    # list of full records per query
    ss.setdefault("last_trace", None)


# ---------------------------------------------------------------------------
# Stats helpers (for the dashboard)
# ---------------------------------------------------------------------------
def _percentile(values: List[float], p: float) -> float:
    """Linear-interpolated percentile. p in [0,100]. Empty list → 0."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    if lo == hi:
        return float(s[lo])
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


# ---------------------------------------------------------------------------
# Altair dark theme + panel helper
# ---------------------------------------------------------------------------
CHART_BG = "#0f172a"
CHART_AXIS = "#475569"
CHART_LABEL = "#94a3b8"
CHART_GRID = "#1e293b"


def _base_chart_config(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_view(stroke=None, fill=CHART_BG)
        .configure_axis(
            labelColor=CHART_LABEL,
            titleColor=CHART_LABEL,
            domainColor=CHART_AXIS,
            tickColor=CHART_AXIS,
            gridColor=CHART_GRID,
            labelFontSize=10,
            titleFontSize=10,
        )
        .configure_legend(
            labelColor=CHART_LABEL,
            titleColor=CHART_LABEL,
            labelFontSize=10,
            symbolSize=80,
            orient="top",
        )
        .properties(background=CHART_BG)
    )


def render_panel_head(index: int, title: str, sub: str, badge: str, badge_kind: str = "") -> str:
    """HTML for the panel header (number + title + subtitle + SLO badge)."""
    badge_cls = f"panel-badge {badge_kind}".strip()
    return (
        f'<div class="panel-head">'
        f'  <div>'
        f'    <div class="panel-title">{index}. {title}</div>'
        f'    <div class="panel-sub">{sub}</div>'
        f'  </div>'
        f'  <div><span class="{badge_cls}">{badge}</span></div>'
        f'</div>'
    )


def panel_empty(msg: str = "Cần ít nhất 2 query để vẽ biểu đồ") -> str:
    return f'<div class="panel-empty">{msg}</div>'


def compute_timeseries(records: List[Dict]) -> Dict[str, List]:
    """
    Build per-query cumulative/rolling series used by dashboard panels.

    Returns columns aligned by index 0..N-1.
    """
    out: Dict[str, List] = {
        "q": [],                   # query #
        "latency_ms": [],          # instantaneous
        "p50": [], "p95": [], "p99": [],  # rolling percentiles up to q
        "cum_queries": [],         # 1,2,3,...
        "cum_errors": [],          # cumulative error count
        "error_pct": [],           # cum_errors / cum_queries * 100
        "cum_cost": [],            # cumulative USD
        "cum_tokens_in": [],       # cumulative input tokens (est.)
        "cum_tokens_out": [],      # cumulative output tokens (est.)
        "avg_faith": [],           # rolling average faithfulness
    }
    running_lat: List[float] = []
    sum_cost = 0.0
    sum_in = 0
    sum_out = 0
    err_count = 0
    faith_sum = 0.0
    for i, r in enumerate(records, start=1):
        running_lat.append(r["latency_ms"])
        sum_cost += r["cost_usd"]
        sum_in += r["tokens_in"]
        sum_out += r["tokens_out"]
        if r["error"]:
            err_count += 1
        faith_sum += r["faithfulness"]

        out["q"].append(i)
        out["latency_ms"].append(r["latency_ms"])
        out["p50"].append(_percentile(running_lat, 50))
        out["p95"].append(_percentile(running_lat, 95))
        out["p99"].append(_percentile(running_lat, 99))
        out["cum_queries"].append(i)
        out["cum_errors"].append(err_count)
        out["error_pct"].append((err_count / i) * 100)
        out["cum_cost"].append(sum_cost)
        out["cum_tokens_in"].append(sum_in)
        out["cum_tokens_out"].append(sum_out)
        out["avg_faith"].append(faith_sum / i)
    return out


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Eval Factory — Lab 14", layout="wide")
password_gate()

st.markdown(
    """
    <style>
    :root {
        --bg: #0a0f1f;
        --surface: #0f172a;
        --surface-2: #111a2e;
        --border: #1e293b;
        --border-strong: #334155;
        --accent: #ef4444;
        --accent-2: #f97316;
        --accent-soft: rgba(239, 68, 68, 0.10);
        --accent-border: rgba(239, 68, 68, 0.40);
        --ok: #22c55e;
        --warn: #f59e0b;
        --info: #60a5fa;
        --purple: #a855f7;
        --yellow: #facc15;
        --mute: #94a3b8;
        --text: #e2e8f0;
        --text-dim: #cbd5e1;
        --card-shadow: 0 1px 2px rgba(0,0,0,0.3), 0 4px 14px rgba(0,0,0,0.25);
    }
    /* Global dark background */
    .stApp { background: var(--bg) !important; }
    .block-container { padding-top: 2.0rem; max-width: 1400px; }
    body, [data-testid="stAppViewContainer"] { color: var(--text); }
    h1, h2, h3, h4, h5, h6, p, span, div, li, label { color: var(--text); }
    code, .stCodeBlock { background: var(--surface-2) !important; color: #fca5a5 !important; }

    /* Hero */
    .hero {
        display:flex; align-items:center; justify-content:space-between; gap:16px;
        padding: 16px 20px; margin-bottom: 18px;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(239,68,68,0.08) 0%, var(--surface) 55%, rgba(249,115,22,0.05) 100%);
        border: 1px solid var(--accent-border);
        box-shadow: var(--card-shadow);
    }
    .hero-title {
        font-size: 1.55rem; font-weight: 800; letter-spacing: -0.01em;
        background: linear-gradient(90deg, #f87171, #fb923c);
        -webkit-background-clip: text; background-clip: text; color: transparent;
        margin: 0;
    }
    .hero-sub { color: var(--mute); font-size:0.88rem; margin-top:4px; }
    .hero-pill {
        display:inline-flex; align-items:center; gap:6px;
        padding: 4px 10px; border-radius: 999px;
        background: var(--surface-2); border:1px solid var(--accent-border);
        font-size:0.78rem; font-weight:600; color:#fca5a5;
    }
    .hero-pill .dot {
        width:8px; height:8px; border-radius:50%; background:var(--ok);
        box-shadow: 0 0 0 3px rgba(34,197,94,0.25);
    }

    /* Section label */
    .sec-label {
        display:flex; align-items:center; gap:8px;
        font-size: 0.95rem; font-weight: 700; color: var(--text);
        margin: 6px 0 10px 0;
    }
    .sec-label .sec-dot {
        width:10px; height:10px; border-radius:50%; background:var(--accent);
        box-shadow: 0 0 0 4px rgba(239,68,68,0.18);
    }

    /* Monitor cards */
    .monitor-card {
        border: 1px solid var(--border);
        background: var(--surface);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: var(--card-shadow);
    }
    .monitor-title {
        color: var(--accent);
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 10px;
        display:flex; align-items:center; gap:6px;
    }
    .monitor-title::before {
        content:""; width:6px; height:6px; border-radius:50%;
        background: var(--accent);
    }
    .kv { font-size: 0.88rem; line-height: 1.55; }
    .kv .k { color: var(--mute); }
    .kv .v { color: var(--text); font-weight: 600; }

    .stage-row { display:flex; justify-content:space-between; font-size:0.88rem; padding:3px 0; }
    .stage-ok    { color: var(--ok); }
    .stage-run   { color: var(--warn); }
    .stage-skip  { color: var(--mute); }
    .stage-wait  { color: var(--mute); }

    .chip {
        display:inline-flex; align-items:center; gap:4px;
        padding:2px 9px; border-radius:999px;
        font-size:0.72rem; font-weight:600;
        background: rgba(239,68,68,0.15); color:#fca5a5;
        margin:2px 3px 0 0;
        border:1px solid rgba(239,68,68,0.3);
    }
    .chip .rank {
        font-size:0.62rem; color:#fff; background:#dc2626;
        padding:0 5px; border-radius:999px;
    }

    /* Pipeline timeline */
    .tl-wrap { margin: 2px 0 8px 0; }
    .tl-bar {
        display:flex; width:100%; height:14px;
        border-radius:7px; overflow:hidden;
        background: var(--surface-2); border:1px solid var(--border);
    }
    .tl-seg {
        height:100%; display:flex; align-items:center; justify-content:center;
        font-size:0.65rem; font-weight:700; color:#fff;
        transition: width 0.25s ease;
        min-width: 0; overflow:hidden;
    }
    .tl-seg.retrieval { background: linear-gradient(90deg, #60a5fa, #3b82f6); }
    .tl-seg.rerank    { background: linear-gradient(90deg, #f59e0b, #d97706); }
    .tl-seg.llm       { background: linear-gradient(90deg, #f87171, #dc2626); }
    .tl-seg.pending   { background: repeating-linear-gradient(45deg,#1e293b,#1e293b 6px,#0f172a 6px,#0f172a 12px); }
    .tl-legend {
        display:flex; justify-content:space-between; margin-top:6px;
        font-size:0.72rem; color: var(--mute);
    }
    .tl-legend .item { display:flex; align-items:center; gap:4px; }
    .tl-legend .sw { width:8px; height:8px; border-radius:2px; }
    .sw.retrieval { background:#3b82f6; }
    .sw.rerank    { background:#d97706; }
    .sw.llm       { background:#b91c1c; }

    /* Stage pills */
    .stage-grid { display:grid; grid-template-columns: 1fr; gap:4px; margin-top:8px; }
    .stage-pill {
        display:flex; align-items:center; justify-content:space-between;
        padding:6px 10px; border-radius:8px;
        background: var(--surface-2); border:1px solid var(--border);
        font-size:0.82rem;
    }
    .stage-pill .left { display:flex; align-items:center; gap:8px; color: var(--text-dim); }
    .stage-pill .idot {
        width:8px; height:8px; border-radius:50%; background:#475569;
    }
    .stage-pill.ok     .idot { background: var(--ok); box-shadow:0 0 0 3px rgba(34,197,94,0.25); }
    .stage-pill.run    .idot { background: var(--warn); box-shadow:0 0 0 3px rgba(245,158,11,0.25);
        animation: pulse 1.1s ease-in-out infinite; }
    .stage-pill.skip   .idot { background:#475569; }
    .stage-pill.ok     { border-color: rgba(34,197,94,0.35); }
    .stage-pill.run    { border-color: rgba(245,158,11,0.4); background: rgba(245,158,11,0.06); }
    .stage-pill .right { font-variant-numeric: tabular-nums; color: var(--text); font-weight:600; }
    @keyframes pulse {
        0%,100% { transform: scale(1); opacity: 1; }
        50%     { transform: scale(1.25); opacity: 0.75; }
    }

    /* Faithfulness gauge */
    .gauge-wrap { margin-top: 10px; }
    .gauge-head { display:flex; justify-content:space-between; align-items:center;
        font-size:0.8rem; margin-bottom:4px; }
    .gauge-head .label { color: var(--mute); font-weight:600; }
    .gauge-head .val   { font-weight:700; font-variant-numeric: tabular-nums; color: var(--text); }
    .gauge-track {
        height: 8px; background: var(--surface-2); border-radius:999px; overflow:hidden;
        border:1px solid var(--border);
    }
    .gauge-fill {
        height:100%; border-radius:999px;
        background: linear-gradient(90deg, #ef4444 0%, #f59e0b 45%, #16a34a 80%);
        transition: width 0.3s ease;
    }

    /* Chunk row */
    .chunk-row { margin-top: 10px; }
    .chunk-row .label {
        font-size: 0.72rem; color: var(--mute); text-transform: uppercase;
        letter-spacing: 0.05em; font-weight:700; margin-bottom:3px;
    }

    /* Metric grid */
    .metric-grid {
        display:grid; grid-template-columns: 1fr 1fr; gap: 8px;
    }
    .metric-card {
        background: var(--surface-2); border:1px solid var(--border); border-radius:10px;
        padding:10px 12px;
    }
    .metric-card .mk {
        font-size:0.68rem; color: var(--mute); text-transform:uppercase;
        letter-spacing:0.05em; font-weight:700;
    }
    .metric-card .mv {
        font-size:1.25rem; font-weight:800; color: var(--text); margin-top:2px;
        font-variant-numeric: tabular-nums;
    }
    .metric-card .ms { font-size:0.72rem; color: var(--mute); margin-top:1px; }
    .metric-card.accent { border-color: rgba(239,68,68,0.35); background: rgba(239,68,68,0.06); }

    /* Gate badge */
    .gate-badge {
        display:inline-flex; align-items:center; gap:8px;
        padding:8px 14px; border-radius:10px;
        font-size:0.95rem; font-weight:800; letter-spacing:0.04em;
    }
    .gate-badge .g-dot { width:10px; height:10px; border-radius:50%; }
    .gate-approve {
        background: linear-gradient(135deg, rgba(34,197,94,0.18), rgba(34,197,94,0.08));
        color:#86efac; border:1px solid rgba(34,197,94,0.5);
    }
    .gate-approve .g-dot { background:#22c55e; box-shadow:0 0 0 4px rgba(34,197,94,0.25); }
    .gate-block {
        background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(239,68,68,0.08));
        color:#fca5a5; border:1px solid rgba(239,68,68,0.55);
    }
    .gate-block .g-dot { background:#ef4444; box-shadow:0 0 0 4px rgba(239,68,68,0.25); }
    .gate-unknown {
        background: var(--surface-2); color: var(--mute); border:1px solid var(--border);
    }
    .gate-unknown .g-dot { background: var(--mute); }

    /* Empty-state panel */
    .empty-panel {
        text-align:center; padding: 14px 8px;
        color: var(--mute); font-size:0.85rem;
    }
    .empty-panel code {
        background: rgba(239,68,68,0.12); color:#fca5a5; padding:1px 6px; border-radius:4px;
        font-size:0.78rem;
    }

    /* Chat chrome */
    [data-testid="stChatMessage"] {
        padding: 10px 14px !important; border-radius: 12px !important;
        margin-bottom: 10px !important;
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
    }

    /* Chat input: tidy container + visible text */
    [data-testid="stChatInput"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    [data-testid="stChatInput"] > div {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 2px !important;
        box-shadow: var(--card-shadow);
    }
    [data-testid="stChatInput"] > div:focus-within {
        border-color: var(--accent-border) !important;
        box-shadow: 0 0 0 3px rgba(239,68,68,0.15);
    }
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInputTextArea"] {
        background: transparent !important;
        color: var(--text) !important;
        caret-color: var(--accent) !important;
        font-size: 0.92rem !important;
        min-height: 44px !important;
        padding: 10px 12px !important;
        border: none !important;
        box-shadow: none !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--mute) !important; opacity: 0.9;
    }
    /* Submit button */
    [data-testid="stChatInputSubmitButton"] {
        background: var(--accent) !important;
        border: none !important;
        border-radius: 8px !important;
        color: #fff !important;
        transition: background 0.15s ease;
    }
    [data-testid="stChatInputSubmitButton"]:hover:not(:disabled) {
        background: #dc2626 !important;
    }
    [data-testid="stChatInputSubmitButton"]:disabled {
        background: var(--surface-2) !important;
        opacity: 0.5;
    }
    [data-testid="stChatInputSubmitButton"] svg {
        fill: #fff !important; color: #fff !important;
    }
    /* Trace summary row */
    .trace-summary {
        display:flex; gap:8px; flex-wrap:wrap;
        font-size:0.75rem; margin-top:4px;
    }
    .trace-summary .tbadge {
        display:inline-flex; align-items:center; gap:4px;
        padding:2px 8px; border-radius:6px;
        background: var(--surface-2); color: var(--text-dim); font-weight:600;
        border:1px solid var(--border);
        font-variant-numeric: tabular-nums;
    }
    .trace-summary .tbadge.accent {
        background: rgba(239,68,68,0.12); color:#fca5a5; border-color: rgba(239,68,68,0.4);
    }
    /* Context preview block */
    .ctx-preview {
        background: var(--surface-2); border:1px solid var(--border); border-radius:8px;
        padding:10px 12px; margin-top:8px;
        font-size:0.82rem; color: var(--text-dim); line-height:1.5;
        white-space: pre-wrap;
    }
    .ctx-preview-head {
        display:flex; justify-content:space-between; align-items:center;
        font-size:0.72rem; text-transform:uppercase; letter-spacing:0.05em;
        color: var(--mute); font-weight:700; margin-bottom:4px;
    }
    .ctx-preview-head .src { color:#fca5a5; }

    /* Streamlit expander + container dark */
    details[data-testid="stExpander"] summary {
        color: var(--text-dim) !important;
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
    }
    details[data-testid="stExpander"] { background: var(--surface) !important; }
    [data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--border) !important; }

    /* Dashboard panels */
    .dash-divider {
        height: 1px; background: var(--border); margin: 24px 0 18px 0;
    }
    .dash-title {
        font-size: 1.1rem; font-weight: 800; color: var(--text);
        display:flex; align-items:center; gap:10px; margin-bottom: 12px;
    }
    .dash-title .sec-dot {
        width:10px; height:10px; border-radius:50%; background: var(--accent);
        box-shadow: 0 0 0 4px rgba(239,68,68,0.18);
    }
    .dash-title .sub {
        font-size:0.78rem; font-weight:500; color: var(--mute);
    }
    .panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px 16px 10px 16px;
        height: 100%;
        box-shadow: var(--card-shadow);
    }
    .panel-head {
        display:flex; align-items:flex-start; justify-content:space-between;
        gap: 10px; margin-bottom: 6px;
    }
    .panel-title {
        font-size: 0.95rem; font-weight: 700; color: var(--text);
        line-height: 1.25;
    }
    .panel-sub {
        font-size: 0.72rem; color: var(--mute); margin-top: 2px;
    }
    .panel-badge {
        font-size: 0.65rem; font-weight: 700;
        padding: 3px 8px; border-radius: 6px;
        background: var(--surface-2); color: var(--mute);
        border: 1px solid var(--border);
        white-space: nowrap; letter-spacing: 0.02em;
    }
    .panel-badge.ok   { color:#86efac; border-color: rgba(34,197,94,0.4); background: rgba(34,197,94,0.08); }
    .panel-badge.warn { color:#fcd34d; border-color: rgba(250,204,21,0.4); background: rgba(250,204,21,0.08); }
    .panel-badge.err  { color:#fca5a5; border-color: rgba(239,68,68,0.45); background: rgba(239,68,68,0.10); }

    .panel-big {
        font-size: 2.35rem; font-weight: 800;
        margin: 6px 0 2px 0;
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.02em;
    }
    .panel-big.blue   { color: #60a5fa; }
    .panel-big.red    { color: #f87171; }
    .panel-big.green  { color: #4ade80; }
    .panel-big.purple { color: #c084fc; }
    .panel-big.yellow { color: #facc15; }
    .panel-big.white  { color: var(--text); }
    .panel-legend {
        font-size: 0.75rem; color: var(--mute); margin-bottom: 4px;
    }
    .panel-legend b { color: var(--text); font-weight: 700; }
    .panel-empty {
        color: var(--mute); font-size: 0.82rem; text-align:center;
        padding: 22px 8px; font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_agent_cfg = "V2 · BM25 top_k=5 → CrossEncoder rerank top 3 → GPT-4o-mini"
_key_ok = bool(os.getenv("OPENAI_API_KEY"))
_pill_html = (
    '<span class="hero-pill"><span class="dot"></span>API key OK</span>'
    if _key_ok else
    '<span class="hero-pill" style="color:#b45309;border-color:#fcd34d;background:#fffbeb;">'
    '<span class="dot" style="background:#f59e0b;box-shadow:0 0 0 3px rgba(245,158,11,0.18);"></span>'
    'OPENAI_API_KEY chưa set</span>'
)
st.markdown(
    f"""
    <div class="hero">
      <div>
        <div class="hero-title">🧪 AI Eval Factory — Chat với Agent V2</div>
        <div class="hero-sub">{_agent_cfg}</div>
      </div>
      <div>{_pill_html}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

init_state()

col_chat, col_mon = st.columns([2, 1], gap="large")

# ---------------------------------------------------------------------------
# Monitor column (right)
# ---------------------------------------------------------------------------
with col_mon:
    st.markdown(
        '<div class="sec-label"><span class="sec-dot"></span>System Monitor</div>',
        unsafe_allow_html=True,
    )

    # --- Status card ---
    status_slot = st.empty()

    def render_status(state: str, detail: str = "") -> None:
        color = {
            "healthy": "#16a34a", "busy": "#f59e0b",
            "error": "#dc2626", "warn": "#f97316",
        }.get(state, "#9ca3af")
        label = {
            "healthy": "Healthy — sẵn sàng",
            "busy": "Đang xử lý truy vấn...",
            "error": "Lỗi hệ thống",
            "warn": "Cảnh báo cấu hình",
        }.get(state, "Unknown")
        pulse_css = "animation: pulse 1.1s ease-in-out infinite;" if state == "busy" else ""
        status_slot.markdown(
            f"""
            <div class="monitor-card">
              <div class="monitor-title">System Status</div>
              <div style="display:flex;align-items:center;gap:10px;">
                <span style="width:12px;height:12px;border-radius:50%;background:{color};
                             box-shadow:0 0 0 4px {color}22;{pulse_css}"></span>
                <span style="font-weight:700;color:#111827;font-size:0.95rem;">{label}</span>
              </div>
              <div class="kv" style="margin-top:6px;"><span class="k">{detail}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    has_key = bool(os.getenv("OPENAI_API_KEY"))
    if not has_key:
        render_status("warn", "OPENAI_API_KEY chưa được set — chat sẽ fail khi gọi LLM.")
    else:
        render_status("healthy", "V2: BM25 top_k=5 → rerank top 3 → GPT-4o-mini")

    # --- Pipeline trace card (placeholder, updated live during a query) ---
    trace_slot = st.empty()

    def render_trace(stages: Dict[str, Dict[str, Any]]) -> None:
        # --- 1. Timeline bar (stacked) ---
        ms_r = stages.get("retrieval", {}).get("ms", 0) if stages.get("retrieval", {}).get("state") == "done" else 0
        ms_k = stages.get("rerank", {}).get("ms", 0) if stages.get("rerank", {}).get("state") == "done" else 0
        ms_l = stages.get("llm", {}).get("ms", 0) if stages.get("llm", {}).get("state") == "done" else 0
        total_ms = ms_r + ms_k + ms_l

        if total_ms > 0:
            pr = ms_r / total_ms * 100
            pk = ms_k / total_ms * 100
            pl = ms_l / total_ms * 100
            timeline = (
                f'<div class="tl-bar">'
                f'  <div class="tl-seg retrieval" style="width:{pr:.1f}%" title="Retrieval {ms_r}ms">{f"{ms_r}" if pr>12 else ""}</div>'
                f'  <div class="tl-seg rerank"    style="width:{pk:.1f}%" title="Rerank {ms_k}ms">{f"{ms_k}" if pk>12 else ""}</div>'
                f'  <div class="tl-seg llm"       style="width:{pl:.1f}%" title="LLM {ms_l}ms">{f"{ms_l}" if pl>12 else ""}</div>'
                f'</div>'
                f'<div class="tl-legend">'
                f'  <div class="item"><span class="sw retrieval"></span>Retrieval {ms_r}ms</div>'
                f'  <div class="item"><span class="sw rerank"></span>Rerank {ms_k}ms</div>'
                f'  <div class="item"><span class="sw llm"></span>LLM {ms_l}ms</div>'
                f'  <div class="item" style="font-weight:700;color:#111827;">Σ {total_ms}ms</div>'
                f'</div>'
            )
        else:
            timeline = (
                '<div class="tl-bar"><div class="tl-seg pending" style="width:100%;"></div></div>'
                '<div class="tl-legend"><div class="item">Đang chờ truy vấn đầu tiên...</div></div>'
            )

        # --- 2. Stage pills ---
        def pill(name: str, label: str, extra: str = "") -> str:
            s = stages.get(name, {"state": "wait"})
            state = s.get("state", "wait")
            cls = {"done": "ok", "running": "run", "skipped": "skip", "wait": ""}.get(state, "")
            if state == "done" and "ms" in s:
                right = f"{s['ms']} ms"
            elif state == "skipped":
                right = "skipped"
            elif state == "running":
                right = "..."
            else:
                right = "—"
            return (
                f'<div class="stage-pill {cls}">'
                f'  <div class="left"><span class="idot"></span><span>{label}</span>'
                f'    <span style="color:#9ca3af;font-size:0.75rem;">{extra}</span></div>'
                f'  <div class="right">{right}</div>'
                f'</div>'
            )

        ret_extra = f"top_k={stages.get('retrieval',{}).get('top_k','—')}"
        rerank_extra = f"top_k={stages.get('rerank',{}).get('top_k','—')}"
        llm_extra = stages.get("llm", {}).get("model", "gpt-4o-mini")

        stage_grid = (
            '<div class="stage-grid">'
            f"{pill('retrieval', 'Retrieval · BM25', ret_extra)}"
            f"{pill('rerank', 'Rerank · CrossEncoder', rerank_extra)}"
            f"{pill('llm', 'LLM · ' + llm_extra)}"
            '</div>'
        )

        # --- 3. Faithfulness gauge ---
        faith = stages.get("faithfulness", {})
        faith_html = ""
        if faith.get("state") == "done":
            val = float(faith.get("value", 0.0))
            pct = max(0.0, min(1.0, val)) * 100
            faith_html = (
                f'<div class="gauge-wrap">'
                f'  <div class="gauge-head">'
                f'    <span class="label">Faithfulness (lexical)</span>'
                f'    <span class="val">{val:.2f}</span>'
                f'  </div>'
                f'  <div class="gauge-track"><div class="gauge-fill" style="width:{pct:.1f}%"></div></div>'
                f'</div>'
            )

        # --- 4. Chunk chips ---
        chunk_html = ""
        ret = stages.get("retrieval", {})
        if ret.get("state") == "done" and ret.get("chunks"):
            chips = "".join(
                f'<span class="chip"><span class="rank">{i+1}</span>{c}</span>'
                for i, c in enumerate(ret["chunks"])
            )
            chunk_html += (
                f'<div class="chunk-row"><div class="label">Retrieved</div>{chips}</div>'
            )
        rerank = stages.get("rerank", {})
        if rerank.get("state") == "done" and rerank.get("chunks"):
            chips = "".join(
                f'<span class="chip" style="background:#fef3c7;color:#92400e;border-color:rgba(146,64,14,0.2);">'
                f'<span class="rank" style="background:#92400e;">{i+1}</span>{c}</span>'
                for i, c in enumerate(rerank["chunks"])
            )
            chunk_html += (
                f'<div class="chunk-row"><div class="label">Reranked (top {rerank.get("top_k","—")})</div>{chips}</div>'
            )

        trace_slot.markdown(
            f"""
            <div class="monitor-card">
              <div class="monitor-title">Pipeline Trace</div>
              <div class="tl-wrap">{timeline}</div>
              {stage_grid}
              {faith_html}
              {chunk_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    initial_stages = {k: {"state": "wait"} for k in ("retrieval", "rerank", "llm", "faithfulness")}
    last = st.session_state.last_trace
    render_trace(last["stages_ui"] if last else initial_stages)

    # --- Session metrics card ---
    sm = st.session_state.session_metrics
    avg_ms = (sm["total_ms"] / sm["queries"]) if sm["queries"] else 0
    avg_faith = (sm["faith_sum"] / sm["queries"]) if sm["queries"] else 0.0

    st.markdown(
        f"""
        <div class="monitor-card">
          <div class="monitor-title">Session Metrics</div>
          <div class="metric-grid">
            <div class="metric-card accent">
              <div class="mk">Queries</div>
              <div class="mv">{sm['queries']}</div>
              <div class="ms">phiên hiện tại</div>
            </div>
            <div class="metric-card">
              <div class="mk">Avg latency</div>
              <div class="mv">{avg_ms:.0f}<span style="font-size:0.72rem;color:#9ca3af;font-weight:600;"> ms</span></div>
              <div class="ms">{(avg_ms/1000):.2f}s trung bình</div>
            </div>
            <div class="metric-card">
              <div class="mk">Avg faith</div>
              <div class="mv">{avg_faith:.2f}</div>
              <div class="ms">lexical overlap</div>
            </div>
            <div class="metric-card">
              <div class="mk">Est. cost</div>
              <div class="mv">${sm['total_cost']:.4f}</div>
              <div class="ms">{sm['total_tokens']:,} tok</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # (Latency trend chart moved to the full dashboard below.)

    # --- Benchmark aggregate (offline, from reports/summary.json) ---
    summary = load_summary()
    if summary and summary.get("metrics"):
        m = summary["metrics"]
        reg = summary.get("regression", {})
        decision = reg.get("decision", "—")
        deltas = reg.get("deltas", {})

        def dtag(key: str, fmt: str = ":+.2f") -> str:
            if not deltas or key not in deltas:
                return ""
            d = deltas[key]
            color = "#16a34a" if d >= 0 else "#dc2626"
            arrow = "▲" if d > 0 else ("▼" if d < 0 else "–")
            return (
                f'<span style="color:{color};font-size:0.7rem;font-weight:700;">'
                f'{arrow} {format(d, fmt.strip(":"))}</span>'
            )

        gate_cls = {"APPROVE": "gate-approve", "BLOCK": "gate-block"}.get(decision, "gate-unknown")
        gate_label = {"APPROVE": "APPROVE", "BLOCK": "BLOCK"}.get(decision, "UNKNOWN")

        st.markdown(
            f"""
            <div class="monitor-card">
              <div class="monitor-title">Benchmark Aggregate · reports/summary.json</div>
              <div style="margin-bottom:10px;">
                <span class="gate-badge {gate_cls}">
                  <span class="g-dot"></span>Release Gate: {gate_label}
                </span>
              </div>
              <div class="metric-grid">
                <div class="metric-card accent">
                  <div class="mk">Avg score</div>
                  <div class="mv">{m.get('avg_score',0):.2f}</div>
                  <div class="ms">{dtag('avg_score', ':+.2f')} vs V1</div>
                </div>
                <div class="metric-card">
                  <div class="mk">Hit rate</div>
                  <div class="mv">{m.get('hit_rate',0):.0%}</div>
                  <div class="ms">{dtag('hit_rate', ':+.1%')} vs V1</div>
                </div>
                <div class="metric-card">
                  <div class="mk">MRR</div>
                  <div class="mv">{m.get('mrr',0):.0%}</div>
                  <div class="ms">{dtag('mrr', ':+.1%')} vs V1</div>
                </div>
                <div class="metric-card">
                  <div class="mk">Faith</div>
                  <div class="mv">{m.get('faithfulness',0):.2f}</div>
                  <div class="ms">full benchmark</div>
                </div>
                <div class="metric-card">
                  <div class="mk">Avg latency</div>
                  <div class="mv">{m.get('avg_latency',0):.2f}<span style="font-size:0.72rem;color:#9ca3af;font-weight:600;"> s</span></div>
                  <div class="ms">{dtag('avg_latency', ':+.2f')}s vs V1</div>
                </div>
                <div class="metric-card">
                  <div class="mk">Eval cost</div>
                  <div class="mv">${m.get('estimated_cost_usd',0):.4f}</div>
                  <div class="ms">{m.get('total_tokens',0):,} tok</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="monitor-card">
              <div class="monitor-title">Benchmark Aggregate</div>
              <div class="empty-panel">
                <div style="font-size:1.6rem;opacity:0.5;">📊</div>
                Chưa có <code>reports/summary.json</code>.<br/>
                Chạy <code>python main.py</code> để sinh benchmark + Release Gate.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Chat column (left)
# ---------------------------------------------------------------------------
with col_chat:
    st.markdown(
        '<div class="sec-label"><span class="sec-dot"></span>Chat — hỏi về tài liệu tuyển sinh FUV</div>',
        unsafe_allow_html=True,
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            tr = msg.get("trace")
            if tr:
                faith = tr["stages"]["faithfulness"]["value"]
                summary_chips = (
                    f'<div class="trace-summary">'
                    f'<span class="tbadge accent">⏱ {tr["total_ms"]} ms</span>'
                    f'<span class="tbadge">🎯 faith {faith:.2f}</span>'
                    f'<span class="tbadge">🧮 ~{tr["est_tokens"]} tok</span>'
                    f'<span class="tbadge">💰 ${tr["est_cost_usd"]:.4f}</span>'
                    f'<span class="tbadge">📄 {len(tr["chunks"])} chunk</span>'
                    f'</div>'
                )
                st.markdown(summary_chips, unsafe_allow_html=True)

                with st.expander("🔎 Xem context + trace chi tiết", expanded=False):
                    for i, (cid, ctx) in enumerate(zip(tr["chunks"], tr["contexts"])):
                        preview = ctx[:600] + ("..." if len(ctx) > 600 else "")
                        st.markdown(
                            f"""
                            <div class="ctx-preview">
                              <div class="ctx-preview-head">
                                <span>#{i+1} · <span class="src">{cid}</span></span>
                                <span>{len(ctx):,} chars</span>
                              </div>
                              {preview}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with st.expander("🧬 Raw stages (JSON)", expanded=False):
                        st.json(tr["stages"])

    used = st.session_state.session_metrics["queries"]
    remaining = max(0, MAX_QUERIES_PER_SESSION - used)
    if remaining <= 3:
        st.caption(f"⚠️ Còn {remaining}/{MAX_QUERIES_PER_SESSION} câu hỏi cho session này.")

    question = st.chat_input("Hỏi về tuyển sinh, quy chế học vụ, hoặc hồ sơ nhập học...")
    if question:
        if remaining <= 0:
            st.error(
                f"Đã đạt giới hạn {MAX_QUERIES_PER_SESSION} câu hỏi/session. "
                "Reload trang để bắt đầu session mới."
            )
            st.stop()

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        if not has_key:
            with st.chat_message("assistant"):
                st.error("Chưa có OPENAI_API_KEY. Export key rồi reload trang.")
            st.stop()

        render_status("busy", f"Đang xử lý: {question[:60]}...")

        try:
            agent = load_agent()
        except Exception as e:
            render_status("error", f"Không init được agent: {e}")
            with st.chat_message("assistant"):
                st.error(f"Init agent failed: {e}")
            st.stop()

        live_stages: Dict[str, Dict[str, Any]] = {
            k: {"state": "wait"} for k in ("retrieval", "rerank", "llm", "faithfulness")
        }

        def on_step(name: str, state: str, payload: Dict[str, Any]) -> None:
            entry = {"state": state}
            entry.update(payload)
            live_stages[name] = entry
            render_trace(live_stages)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_Đang suy nghĩ..._")
            try:
                trace = run_traced(agent, question, on_step)
            except Exception as e:
                render_status("error", f"Pipeline lỗi: {e}")
                placeholder.error(f"Query failed: {e}")
                st.stop()

            placeholder.markdown(trace["answer"])

            trace_ui = {"stages_ui": live_stages, **trace}
            st.session_state.last_trace = trace_ui
            st.session_state.messages.append(
                {"role": "assistant", "content": trace["answer"], "trace": trace}
            )

            sm = st.session_state.session_metrics
            sm["queries"] += 1
            sm["total_ms"] += trace["total_ms"]
            sm["total_tokens"] += trace["est_tokens"]
            sm["total_cost"] += trace["est_cost_usd"]
            sm["faith_sum"] += trace["stages"]["faithfulness"]["value"]
            st.session_state.latency_history.append(trace["total_ms"])
            st.session_state.faith_history.append(trace["stages"]["faithfulness"]["value"])

            # Full per-query record for the dashboard timeseries.
            faith_val = float(trace["stages"]["faithfulness"]["value"])
            # Tokens in = prompt chars /4 (we lost the raw prompt but est_tokens
            # already ≈ (prompt + answer)/4). Split via answer length.
            est_out = max(1, int(len(trace["answer"]) / 4))
            est_in = max(1, trace["est_tokens"] - est_out)
            st.session_state.query_history.append(
                {
                    "ts": time.time(),
                    "latency_ms": int(trace["total_ms"]),
                    "cost_usd": float(trace["est_cost_usd"]),
                    "tokens_in": est_in,
                    "tokens_out": est_out,
                    "faithfulness": faith_val,
                    "error": faith_val < 0.30,  # low-faith = proxy for bad answer
                }
            )

        render_status("healthy", f"Xong — {trace['total_ms']} ms, {len(trace['chunks'])} chunks")
        st.rerun()


# ---------------------------------------------------------------------------
# Dashboard — full width, 3×2 grid of panels
# ---------------------------------------------------------------------------
st.markdown('<div class="dash-divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dash-title"><span class="sec-dot"></span>'
    'Live Dashboard <span class="sub">· 6 panel metric, cập nhật sau mỗi query</span>'
    '</div>',
    unsafe_allow_html=True,
)

_records = st.session_state.query_history
_ts = compute_timeseries(_records)
_has_data = len(_records) >= 2

dash_row1 = st.columns(3, gap="small")
dash_row2 = st.columns(3, gap="small")

# Lab-aligned thresholds (mirror main.py RELEASE_GATE defaults).
GATE_MAX_LATENCY_S = 3.0
GATE_MIN_FAITH = 0.10
GATE_MIN_SCORE = 2.5
GATE_MIN_HIT_RATE = 0.45
GATE_MIN_MRR = 0.40


def _open_panel(index: int, title: str, sub: str, badge: str, badge_kind: str = "") -> None:
    st.markdown(
        '<div class="panel">' + render_panel_head(index, title, sub, badge, badge_kind),
        unsafe_allow_html=True,
    )


def _close_panel() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


# ---- Panel 1: Latency per query + SLO (GATE_MAX_LATENCY) ----
with dash_row1[0]:
    _open_panel(
        1, "Latency (live session)",
        "milliseconds per query · red dashed = Release Gate budget",
        f"SLO ≤ {GATE_MAX_LATENCY_S:.1f}s", "ok",
    )
    if _records:
        last_ms = _records[-1]["latency_ms"]
        avg_ms = sum(r["latency_ms"] for r in _records) / len(_records)
        st.markdown(
            f'<div class="panel-big white">{last_ms}<span style="font-size:0.9rem;color:var(--mute);font-weight:600;"> ms</span></div>'
            f'<div class="panel-legend">avg <b>{avg_ms:.0f} ms</b> · last query</div>',
            unsafe_allow_html=True,
        )
    if _has_data:
        df = pd.DataFrame({"Query": _ts["q"], "ms": _ts["latency_ms"]})
        line = alt.Chart(df).mark_line(
            interpolate="monotone", color="#60a5fa", strokeWidth=2.2
        ).encode(x=alt.X("Query:Q", title=None), y=alt.Y("ms:Q", title="ms"))
        pts = alt.Chart(df).mark_point(
            filled=True, color="#60a5fa", size=45
        ).encode(x="Query:Q", y="ms:Q")
        slo = alt.Chart(pd.DataFrame({"y": [GATE_MAX_LATENCY_S * 1000]})).mark_rule(
            color="#ef4444", strokeDash=[4, 4], strokeWidth=1.5
        ).encode(y="y:Q")
        chart = _base_chart_config((line + pts + slo).properties(height=150))
        st.altair_chart(chart, use_container_width=True)
    else:
        st.markdown(panel_empty(), unsafe_allow_html=True)
    _close_panel()


# ---- Panel 2: Traffic (cumulative queries served) ----
with dash_row1[1]:
    total_q = len(_records)
    _open_panel(
        2, "Traffic", "queries served in this session · cumulative",
        "cumulative", "",
    )
    st.markdown(
        f'<div class="panel-big blue">{total_q}</div>',
        unsafe_allow_html=True,
    )
    if _has_data:
        df = pd.DataFrame({"Query": _ts["q"], "Requests": _ts["cum_queries"]})
        line = alt.Chart(df).mark_line(
            interpolate="step-after", color="#3b82f6", strokeWidth=2.2
        ).encode(x=alt.X("Query:Q", title=None), y=alt.Y("Requests:Q", title=None))
        pts = alt.Chart(df).mark_point(
            filled=True, color="#3b82f6", size=40
        ).encode(x="Query:Q", y="Requests:Q")
        chart = _base_chart_config((line + pts).properties(height=150))
        st.altair_chart(chart, use_container_width=True)
    else:
        st.markdown(panel_empty(), unsafe_allow_html=True)
    _close_panel()


# ---- Panel 3: Faithfulness rolling avg + SLO (GATE_MIN_FAITH) ----
with dash_row1[2]:
    _open_panel(
        3, "Faithfulness (rolling avg)",
        "lexical overlap answer↔context · yellow dashed = gate floor",
        f"SLO ≥ {GATE_MIN_FAITH:.2f}", "warn",
    )
    if _records:
        cur_avg = _ts["avg_faith"][-1]
        hallu = sum(1 for r in _records if r["faithfulness"] < 0.5)
        color_cls = "green" if cur_avg >= GATE_MIN_FAITH else "red"
        st.markdown(
            f'<div class="panel-big {color_cls}">{cur_avg:.2f}</div>'
            f'<div class="panel-legend">hallucination proxy (faith<0.5): <b>{hallu}</b> / {len(_records)}</div>',
            unsafe_allow_html=True,
        )
    if _has_data:
        df = pd.DataFrame({"Query": _ts["q"], "Faith": _ts["avg_faith"], "Per": _ts["q"]})
        # Per-query dots colored red if hallucination
        per_df = pd.DataFrame({
            "Query": _ts["q"],
            "Faith": [r["faithfulness"] for r in _records],
            "Hallu": ["yes" if r["faithfulness"] < 0.5 else "no" for r in _records],
        })
        avg_line = alt.Chart(df).mark_line(
            interpolate="monotone", color="#4ade80", strokeWidth=2.2
        ).encode(x=alt.X("Query:Q", title=None),
                 y=alt.Y("Faith:Q", title="faith", scale=alt.Scale(domain=[0, 1])))
        dots = alt.Chart(per_df).mark_point(filled=True, size=55).encode(
            x="Query:Q", y="Faith:Q",
            color=alt.Color("Hallu:N",
                scale=alt.Scale(domain=["no", "yes"], range=["#4ade80", "#ef4444"]),
                legend=None),
        )
        slo = alt.Chart(pd.DataFrame({"y": [GATE_MIN_FAITH]})).mark_rule(
            color="#facc15", strokeDash=[4, 4], strokeWidth=1.5
        ).encode(y="y:Q")
        chart = _base_chart_config((avg_line + dots + slo).properties(height=150))
        st.altair_chart(chart, use_container_width=True)
    else:
        st.markdown(panel_empty(), unsafe_allow_html=True)
    _close_panel()


# ---- Panel 4: Cost Over Time (cumulative USD) ----
with dash_row2[0]:
    _open_panel(
        4, "Cost Over Time", "USD · cumulative (est. tokens × unit price)",
        f"${COST_PER_1K_TOKENS}/1K tok", "",
    )
    if _records:
        total_cost = _ts["cum_cost"][-1] if _ts["cum_cost"] else 0.0
        st.markdown(
            f'<div class="panel-big purple">${total_cost:.4f}</div>',
            unsafe_allow_html=True,
        )
    if _has_data:
        df = pd.DataFrame({"Query": _ts["q"], "USD": _ts["cum_cost"]})
        area = alt.Chart(df).mark_area(
            interpolate="step-after",
            color=alt.Gradient(
                gradient="linear",
                stops=[alt.GradientStop(color="#a855f7", offset=0),
                       alt.GradientStop(color="#a855f700", offset=1)],
                x1=1, x2=1, y1=1, y2=0,
            ),
        ).encode(x=alt.X("Query:Q", title=None), y=alt.Y("USD:Q", title="USD"))
        line = alt.Chart(df).mark_line(
            interpolate="step-after", color="#c084fc", strokeWidth=2.2
        ).encode(x="Query:Q", y="USD:Q")
        pts = alt.Chart(df).mark_point(
            filled=True, color="#c084fc", size=40
        ).encode(x="Query:Q", y="USD:Q")
        chart = _base_chart_config((area + line + pts).properties(height=150))
        st.altair_chart(chart, use_container_width=True)
    else:
        st.markdown(panel_empty(), unsafe_allow_html=True)
    _close_panel()


# ---- Panel 5: Tokens In / Out (cumulative, dual line) ----
with dash_row2[1]:
    tot_in = _ts["cum_tokens_in"][-1] if _ts["cum_tokens_in"] else 0
    tot_out = _ts["cum_tokens_out"][-1] if _ts["cum_tokens_out"] else 0
    ratio = (tot_in / tot_out) if tot_out else 0.0
    _open_panel(
        5, "Tokens In / Out",
        "input vs output tokens · cumulative (estimated)",
        "estimated", "",
    )
    st.markdown(
        f'<div class="panel-legend">'
        f'In: <b style="color:#4ade80;">{tot_in:,}</b> &nbsp;·&nbsp; '
        f'Out: <b style="color:#facc15;">{tot_out:,}</b> &nbsp;·&nbsp; '
        f'Ratio: <b>{ratio:.2f}</b>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if _has_data:
        df = pd.DataFrame({
            "Query": _ts["q"] * 2,
            "Direction": (["Tokens In"] * len(_ts["q"])) + (["Tokens Out"] * len(_ts["q"])),
            "Tokens": _ts["cum_tokens_in"] + _ts["cum_tokens_out"],
        })
        color_scale = alt.Scale(
            domain=["Tokens In", "Tokens Out"],
            range=["#4ade80", "#facc15"],
        )
        line = alt.Chart(df).mark_line(
            interpolate="step-after", strokeWidth=2.2
        ).encode(
            x=alt.X("Query:Q", title=None),
            y=alt.Y("Tokens:Q", title="tokens"),
            color=alt.Color("Direction:N", scale=color_scale,
                legend=alt.Legend(title=None, orient="top")),
        )
        pts = alt.Chart(df).mark_point(filled=True, size=40).encode(
            x="Query:Q", y="Tokens:Q",
            color=alt.Color("Direction:N", scale=color_scale, legend=None),
        )
        chart = _base_chart_config((line + pts).properties(height=150))
        st.altair_chart(chart, use_container_width=True)
    else:
        st.markdown(panel_empty(), unsafe_allow_html=True)
    _close_panel()


# ---- Panel 6: Benchmark Snapshot (reports/summary.json) ----
with dash_row2[2]:
    _bench = summary  # already loaded above
    has_bench = bool(_bench and _bench.get("metrics"))
    badge_txt = "from summary.json" if has_bench else "not yet run"
    badge_kind = "ok" if has_bench else "warn"
    _open_panel(
        6, "Benchmark Snapshot",
        "retrieval + judge metrics from full golden-set run",
        badge_txt, badge_kind,
    )
    if has_bench:
        m = _bench["metrics"]
        decision = _bench.get("regression", {}).get("decision", "—")
        score = m.get("avg_score", 0)
        hit = m.get("hit_rate", 0)
        mrr = m.get("mrr", 0)
        agr = m.get("agreement_rate", 0)
        score_color = "green" if score >= GATE_MIN_SCORE else "red"

        st.markdown(
            f'<div class="panel-big {score_color}">{score:.2f}'
            f'<span style="font-size:0.8rem;color:var(--mute);font-weight:600;"> avg_score</span></div>',
            unsafe_allow_html=True,
        )
        # Horizontal stacked metric bars
        bar_df = pd.DataFrame({
            "Metric": ["hit_rate", "mrr", "agreement"],
            "Value":  [hit, mrr, agr],
            "SLO":    [GATE_MIN_HIT_RATE, GATE_MIN_MRR, 0.0],
        })
        base = alt.Chart(bar_df).encode(
            y=alt.Y("Metric:N", title=None, sort=["hit_rate", "mrr", "agreement"]),
        )
        bars = base.mark_bar(height=18).encode(
            x=alt.X("Value:Q", title=None, scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Metric:N",
                scale=alt.Scale(
                    domain=["hit_rate", "mrr", "agreement"],
                    range=["#60a5fa", "#a855f7", "#4ade80"]),
                legend=None),
        )
        labels = base.mark_text(
            align="left", dx=4, color="#e2e8f0", fontSize=10
        ).encode(x="Value:Q", text=alt.Text("Value:Q", format=".0%"))
        slo_rules = alt.Chart(bar_df[bar_df["SLO"] > 0]).mark_rule(
            color="#facc15", strokeDash=[3, 3], strokeWidth=1.5
        ).encode(x="SLO:Q", y="Metric:N")
        chart = _base_chart_config((bars + slo_rules + labels).properties(height=115))
        st.altair_chart(chart, use_container_width=True)

        gate_cls = {"APPROVE": "gate-approve", "BLOCK": "gate-block"}.get(decision, "gate-unknown")
        st.markdown(
            f'<div style="margin-top:4px;">'
            f'<span class="gate-badge {gate_cls}" style="padding:5px 10px;font-size:0.8rem;">'
            f'<span class="g-dot"></span>Gate: {decision}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            panel_empty("Chưa có reports/summary.json. Chạy: <code>python main.py</code>"),
            unsafe_allow_html=True,
        )
    _close_panel()
