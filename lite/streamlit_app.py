"""NEXUS Alpha v3, lite dashboard for Streamlit Community Cloud.

The full dashboard (dashboard/app_pro.py) loads 13 neural networks: 2.9 GB of weights and about 5 GB of memory, more than free
hosting allows. This lite version needs no PyTorch and no weights. It shows:

* real output of the full frozen models for four stocks (NVDA, V, F, BAC), computed with the full dashboard's own code from
  TradingView screenshots and saved in lite/examples/ (see lite/README.md for how they were made);
* the architecture views, the model routing and the Ask AI assistant of the full dashboard.

The look, the result cards, the study facts and the Gemini assistant are not copied: they are read from dashboard/app_pro.py at
start-up (only constants and helper functions are taken, nothing that loads a model), so both dashboards stay identical.
"""
import ast
import base64
import html
import io
import json
import math
import os
import random
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DASH = os.path.join(ROOT, "dashboard")
EXAMPLES = os.path.join(HERE, "examples")
FULL_URL = "https://buffalo-deposits-graph-legends.trycloudflare.com"
REPO_URL = "https://github.com/harshatangirala/nexus-alpha-v3-university-of-wollongong-in-dubai"
STOCKS = ["NVDA", "V", "F", "BAC"]
PALETTE_DARK = np.array([[19, 23, 34], [38, 166, 154], [239, 83, 80], [26, 110, 103], [160, 56, 54]], np.uint8)

st.set_page_config(page_title="NEXUS Alpha v3 · lite", page_icon=":material/candlestick_chart:", layout="wide", initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ shared parts of the full dashboard
KEEP_ASSIGN = {"THEMES", "STATIC_CSS", "CALL_META", "HORIZON_TXT", "COMPANIES", "PATTERN_INFO", "PLOT_CFG", "CHAT_RULES", "STUDY_FACTS",
               "PROJECT_FACTS", "GEMINI_BASE", "GEMINI_PREFERRED", "GEMINI_FALLBACK", "SUGGESTED"}
KEEP_DEF = {"rgba", "candle_uri", "build_css", "H", "_norm_tk", "company_name", "_yahoo_name", "pct", "chip", "pbar", "evidence_badge", "kpi",
            "row_at", "summary_points", "summary_html", "detect_patterns", "style_fig", "outlook_chart", "horizon_cards", "verdict_card",
            "risk_cards", "detail_table", "patterns_card", "sources_card", "study_context", "analysis_context", "gemini_models",
            "_gemini_stream", "_stream_with_fallback", "ask_gemini", "resolve_api_key"}


@st.cache_resource
def shared_code():
    with open(os.path.join(DASH, "app_pro.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in KEEP_DEF)
            or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in KEEP_ASSIGN for t in n.targets))]
    found = {n.name for n in keep if isinstance(n, ast.FunctionDef)} | {t.id for n in keep if isinstance(n, ast.Assign) for t in n.targets}
    missing = (KEEP_ASSIGN | KEEP_DEF) - found
    if missing:
        raise RuntimeError("dashboard/app_pro.py no longer defines: " + ", ".join(sorted(missing)))
    return compile(ast.Module(body=keep, type_ignores=[]), "dashboard/app_pro.py (shared parts)", "exec")


@st.cache_resource
def bundle():
    with open(os.path.join(DASH, "bundle.json"), encoding="utf-8") as f:
        return json.load(f)


B = bundle()
SK = {s["task"]: s for s in B.get("deploy_skill", [])}
X = dict(st=st, np=np, pd=pd, math=math, html=html, go=go, random=random, os=os, io=io, base64=base64, esc=html.escape, B=B, SK=SK)
exec(shared_code(), X)
H, esc = X["H"], html.escape
sys.path.insert(0, DASH)
import arch_view as AV  # noqa: E402  (plain HTML builder, no model code)


def load_example(tk):
    with open(os.path.join(EXAMPLES, f"{tk}.json"), encoding="utf-8") as f:
        d = json.load(f)
    r = dict(d)
    r["table"] = pd.DataFrame(d["table"])
    r["sim"] = {"quantiles": np.asarray(d["quantiles"], float), "paths": np.asarray(d["paths_sample"], float), "p_dd": float(d["p_dd_sim"])}
    D = pd.DataFrame(d["daily_bars"])
    D["date"] = pd.to_datetime(D["date"])
    r["D_screen"] = D
    r["how"] = "set from the screenshot set"
    r["used_text"] = False
    r["slot"] = tk
    return r


@st.cache_data(show_spinner=False)
def examples():
    return {tk: load_example(tk) for tk in STOCKS}


EX = examples()

# ------------------------------------------------------------------ header, theme
head = st.columns([6, 1.4], vertical_alignment="center")
with head[0]:
    H("""<div class="nx-brand">
    <div class="nx-logo"><i class="dn" style="height:13px"></i><i class="up" style="height:17px"></i><i class="up" style="height:14px"></i></div>
    <div><div class="nx-title">NEXUS <b>Alpha</b> v3<span>Lite dashboard</span></div>
    <div class="nx-sub">Chart images and SEC filings, read by 13 AI models: direction, volatility and drawdown risk</div></div></div>""")
with head[1]:
    light = st.toggle("Light mode", key="nx_light")
X["light"] = light
X["T"] = T = X["THEMES"]["light" if light else "dark"]
X["n_show"] = 30
st.html(X["build_css"](T, True))

H(f"""<div class="nx-card" style="border-left:4px solid var(--acc)">
<div class="nx-kicker">Lite version for public sharing</div>
<div class="nx-read" style="margin-top:6px">The full NEXUS Alpha dashboard runs 13 neural networks live (7 chart-vision networks and 6 language models,
2.9 GB of weights and about 5 GB of memory). That is more than Streamlit's free hosting allows, so the full version runs on the author's
machine and is online during demonstrations at <a href="{FULL_URL}" target="_blank">{FULL_URL}</a>. This lite version is always on. It shows the
real output of the full models for four stocks, computed from TradingView screenshots on 24 November 2026, plus the architecture, the model
routing and the Ask AI assistant. Code, notebook and results: <a href="{REPO_URL}" target="_blank">GitHub</a>.</div></div>""")

tab_e, tab_ai, tab_arch, tab_m, tab_g = st.tabs(["Example analyses", "Ask AI", "Architecture", "Models", "Guide"])

# ------------------------------------------------------------------ examples
with tab_e:
    labels = {tk: f"{tk} · {X['company_name'](tk)}" for tk in STOCKS}
    pick = st.segmented_control("Stock", STOCKS, default="NVDA", format_func=lambda t: labels[t], key="nx_pick", selection_mode="single") or "NVDA"
    r = EX[pick]
    D = r["D_screen"]
    pats = X["detect_patterns"](D)
    n_bars = len(D)
    r["_summary"] = X["summary_points"](r, pats, n_bars)
    r["_patterns"] = [{"date": str(p["date"].date()), "name": p["name"], "bias": p["bias"]} for p in pats if p["i"] >= n_bars - 15]
    H(X["verdict_card"](r))
    H('<div style="height:12px"></div>' + X["summary_html"](r["_summary"]))
    H(f'<div class="nx-note">Precomputed with the full dashboard\'s frozen models and its own code path (screenshot reader, 7 chart networks, '
      f'calibration, Monte Carlo with 4,000 paths). Data: {esc(r["data_how"])}. No filing text was added, so the outlook comes from the charts only.</div>')
    H(X["sources_card"](r).replace("your screenshots", "the screenshots").replace("your pictures", "the pictures"))
    H('<div class="nx-h">All four horizons</div>')
    H(X["horizon_cards"](r))
    H('<div class="nx-h">Risk and price range (20 sessions)</div>')
    H(X["risk_cards"](r))
    fig = X["outlook_chart"](r, D, pats, True)
    nm = X["company_name"](r["ticker"])
    fig.update_layout(title_text=f"{esc(r['ticker'])} ({esc(nm)}): recent candles and the next 20 sessions (4,000 simulated paths, 30 drawn)")
    st.plotly_chart(fig, width="stretch", config=X["PLOT_CFG"], theme=None, key=f"chart_{pick}_{int(light)}")
    st.caption("Drag to pan, scroll to zoom. Hover a marker to see the candlestick pattern. The cone is a Monte Carlo simulation matched to the models' "
               "direction and volatility; it is a range of what could happen, not a prediction of one price.")
    H(X["patterns_card"](pats, n_bars))
    with st.expander("What the models saw", icon=":material/visibility:"):
        st.caption("The candles, prices and volume were read from the screenshots and redrawn in the exact training format: weekly (52 candles), "
                   "daily (63) and hourly (70). The networks never see the broker's colours, fonts or indicators.")
        st.image(os.path.join(EXAMPLES, f"{pick}_model_input.png"), width="stretch")
    with st.expander("All numbers", icon=":material/table:"):
        H(X["detail_table"](r))

# ------------------------------------------------------------------ Ask AI
with tab_ai:
    H('<div class="nx-card"><div class="nx-kicker">Ask AI · powered by Google Gemini</div><div class="nx-read" style="margin-top:6px">Ask anything about the four '
      'example analyses or the study. The assistant answers from the study\'s verified results and the numbers on the Example analyses page.</div></div>')
    env_key = X["resolve_api_key"]()
    c_k, c_m = st.columns([2, 1])
    with c_k:
        if env_key:
            st.caption("Using the Gemini API key configured for this app.")
            api_key = env_key
        else:
            api_key = st.text_input("Your free Google Gemini API key", type="password", key="nx_api_key", placeholder="AIza...",
                                    help="Get a free key at aistudio.google.com/apikey. Kept only in this browser session's memory.").strip()
    with c_m:
        chat_model = st.selectbox("Model", X["gemini_models"](api_key) if api_key else X["GEMINI_FALLBACK"], key="nx_chat_model")
    st.caption("Context: the study's findings plus the four example analyses (NVDA, V, F, BAC). Your questions are sent to Google's Gemini API; "
               "do not paste anything private.")
    st.session_state.setdefault("nx_chat", [])

    def _queue(q_):
        st.session_state["nx_pending"] = q_

    sugg = ["Explain the NVDA result in simple words", "Compare the four stocks", "How reliable is the direction call?",
            "What does the drawdown risk mean?", "Which networks are used for volatility?", "What did the study prove, and what not?"]
    cols_q = st.columns(3)
    for i_, q_ in enumerate(sugg):
        cols_q[i_ % 3].button(q_, key=f"sq_{i_}", on_click=_queue, args=(q_,), width="stretch")
    for m_ in st.session_state["nx_chat"]:
        with st.chat_message(m_["role"], avatar=":material/person:" if m_["role"] == "user" else ":material/candlestick_chart:"):
            st.markdown(m_["content"])
    typed = st.chat_input("Ask about the examples or the study…", key="nx_chat_in")
    question = typed or st.session_state.pop("nx_pending", None)
    if question:
        if not api_key:
            st.warning("Paste a free Gemini API key above first (aistudio.google.com/apikey).")
        else:
            st.session_state["nx_chat"].append({"role": "user", "content": question})
            with st.chat_message("user", avatar=":material/person:"):
                st.markdown(question)
            hist = st.session_state["nx_chat"][-12:]
            while hist and hist[0]["role"] != "user":
                hist = hist[1:]
            with st.chat_message("assistant", avatar=":material/candlestick_chart:"):
                answer, ok_ = X["ask_gemini"](api_key, chat_model, hist, list(EX.values()))
                if ok_ and st.session_state.get("nx_chat_used"):
                    st.caption(f"Answered by {st.session_state['nx_chat_used']}")
            if ok_:
                st.session_state["nx_chat"].append({"role": "assistant", "content": answer})
    if st.session_state["nx_chat"]:
        st.button("Clear the conversation", key="nx_clear", on_click=lambda: st.session_state.update(nx_chat=[]))

# ------------------------------------------------------------------ architecture
with tab_arch:
    H('<div class="nx-card"><div class="nx-kicker">Architecture · how the whole project works</div><div class="nx-read" style="margin-top:6px">'
      'From the raw data to the prediction. <b>Hover over (or tap) any box</b> for a plain explanation of what it does and the numbers behind it.</div></div>')
    views = ["Research pipeline", "What happens when you click Analyse", "What it predicts"]
    view = st.segmented_control("View", views, default=views[0], key="nx_arch_view", selection_mode="single") or views[0]
    if view == views[0]:
        st.caption("How the models were built offline. This part ran once on a cloud GPU and was then frozen.")
        H(AV.research_html(B))
    elif view == views[1]:
        st.caption("The live path in the full dashboard: screenshots are read, the frozen networks run, and the result page is drawn. Nothing is retrained.")
        H(AV.live_html(B))
    else:
        st.caption("The 12 numbers the system produces for every stock: three kinds of prediction at four horizons.")
        H(AV.predicts_html(B))

# ------------------------------------------------------------------ models
with tab_m:
    H('<div class="nx-h">Which models make each call</div>')
    H('<div class="nx-note">For every task the three best models on 2022-2023 validation data are averaged. When a fresh filing text is supplied, '
      'the text models are blended in with the weight shown; a weight of 1 means charts only.</div>')
    lab_cv = {m: v["label"] for m, v in B["cv_models"].items()}
    lab_nl = {m: v["label"] for m, v in B["nlp_models"].items()}
    names = {"dir": "Direction", "vol": "Volatility", "dd": "Drawdown"}
    body = ""
    for t in B["tasks"]:
        kd, h = t.split("_")
        cvs = "".join(f'<span class="nx-pill">{esc(lab_cv.get(m, m))}</span>' for m in B["cv_members"].get(t, []))
        nls = "".join(f'<span class="nx-pill">{esc(lab_nl.get(m, m))}</span>' for m in B["nlp_members"].get(t, []))
        s = SK.get(t, {})
        ev = (f"{s['skill']:+.3f}" + (" ✓" if s.get("supported") else "")) if s else "-"
        body += f"<tr><td>{names[kd]} · {h} sessions</td><td>{cvs}</td><td>{nls}</td><td>{float(B['w_fresh'].get(t, 1.0)):.2f}</td><td>{ev}</td></tr>"
    H('<div class="nx-card nx-tablewrap"><table class="nx-table"><thead><tr><th>Task</th><th>Chart models (top 3)</th><th>Text models (top 3)</th>'
      f'<th>Weight on charts</th><th>Test skill (✓ = Holm-supported)</th></tr></thead><tbody>{body}</tbody></table></div>')
    H('<div class="nx-h">The full line-up</div>')
    H('<div class="nx-grid2"><div class="nx-card"><div class="nx-kicker">Chart-vision models (7)</div>' + "".join(f'<span class="nx-pill">{esc(v)}</span>' for v in lab_cv.values())
      + '</div><div class="nx-card"><div class="nx-kicker">Language models in the dashboard (6)</div>' + "".join(f'<span class="nx-pill">{esc(v)}</span>' for v in lab_nl.values()) + "</div></div>")

# ------------------------------------------------------------------ guide
with tab_g:
    H(f"""<div class="nx-grid2">
    <div class="nx-card"><div class="nx-kicker">Lite and full versions</div><div class="nx-read">The full dashboard lets you paste your own TradingView screenshots (1W, 1D, 1H)
    or type any ticker, add a filing summary, and runs all 13 networks live. It needs about 5 GB of memory, so it runs on the author's machine:
    <a href="{FULL_URL}" target="_blank">open the full dashboard</a> (available during demonstrations). This lite version shows saved results of those same models.</div></div>
    <div class="nx-card"><div class="nx-kicker">Why the charts are redrawn</div><div class="nx-read">The models were trained on charts drawn in one exact format. Broker screenshots
    differ in colours, fonts and zoom, so the dashboard reads the candles, prices and volume out of the screenshots and redraws the weekly, daily and hourly charts
    in the training format.</div></div>
    <div class="nx-card"><div class="nx-kicker">Reading the outlook</div><div class="nx-read"><b>UP / SIDEWAYS / DOWN</b> compares the price after the horizon with a band of
    0.5 x the stock's own volatility x √time, so SIDEWAYS means a small move for that stock. <b>Drawdown</b> means the price dips more than 10% below the last close
    at some point in the window.</div></div>
    <div class="nx-card"><div class="nx-kicker">The price cone</div><div class="nx-read">A Monte Carlo simulation of 4,000 paths, tuned so that its direction odds and volatility
    match the models. On the locked 2024-2026 test its nominal 80% and 90% ranges contained 82.6% and 90.8% of realised 20-session returns.</div></div></div>
    <div class="nx-note">Educational research, not financial advice. The models were trained on 50 large US stocks; nothing here accounts for costs, taxes or your own situation.</div>""")

H('<div class="nx-foot">NEXUS Alpha v3 · University of Wollongong in Dubai · lite dashboard · frozen recipe ' + esc(str(B.get("stamp", ""))) + "</div>")
