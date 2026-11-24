"""NEXUS Alpha v3 - Pro dashboard.

A presentation layer over the frozen A3b recipe: it calls exactly the same engine (nav3_engine.analyse) and the same
bundle.json as the notebook's own dashboard, so every probability is the parity-tested one. Only the look and the
explanations are new: dark / light themes, a drifting candlestick background, plain-English read-outs, candles joined to the
Monte Carlo cone, and an Architecture page that shows how the whole project works.

Run:  streamlit run app_pro.py   (in the folder that holds bundle.json, nav3_core.py, nav3_engine.py, screen_reader.py)
"""
import base64
import gc
import html
import io
import math
import os
import random
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nav3_engine as E
import screen_engine as SE
import flow_view as FV
import arch_view as AV
from nav3_core import PALETTE_DARK

# ------------------------------------------------------------------ low-memory mode
# The six language models need about 5 GB at full precision; on a PC with 16 GB of RAM they do not fit beside the vision models.
# In low-memory mode each language model is loaded, used and released before the next one (same weights, same numbers, slower).
# The engine itself is not changed (parity); the wrapper is installed once, so Streamlit reruns do not stack it.
if not hasattr(E, "_nlp_raw_orig"):
    E._nlp_raw_orig = E.nlp_raw
    E._LOWMEM = True

    def _nlp_raw_lowmem(B_, mid, texts):
        try:
            return E._nlp_raw_orig(B_, mid, texts)
        finally:
            if E._LOWMEM:
                E._CACHE.pop(("nlp", mid), None)
                gc.collect()
    E.nlp_raw = _nlp_raw_lowmem

st.set_page_config(page_title="NEXUS Alpha v3", page_icon=":material/candlestick_chart:", layout="wide", initial_sidebar_state="collapsed")


@st.cache_resource
def bundle():
    here = os.path.dirname(os.path.abspath(__file__))
    b = E.load_bundle(os.environ.get("NAV3_BUNDLE", here))
    for grp in ("cv_models", "nlp_models"):                 # model files that are not at the stored path are looked up in ./models (hosted copy)
        for m in b.get(grp, {}).values():
            p_ = m.get("path")
            if p_ and not os.path.exists(p_):
                alt = os.path.join(here, "models", os.path.basename(p_))
                if os.path.exists(alt):
                    m["path"] = alt
    return b


B = bundle()
SK = {s["task"]: s for s in B.get("deploy_skill", [])}
esc = html.escape

# ------------------------------------------------------------------ themes
THEMES = {
    "dark": dict(bg="#070c16", solid="#0f1729", panel="rgba(16,25,43,0.80)", panel2="rgba(148,163,184,0.09)", field="#0a1120",
                 line="rgba(148,163,184,0.18)", line2="rgba(148,163,184,0.34)",
                 text="#e6edf8", muted="#9aa9bf", acc="#38bdf8", acc2="#6366f1", up="#2dd4a3", down="#f4606b", mid="#f5b942",
                 glow1="rgba(56,189,248,0.09)", glow2="rgba(99,102,241,0.12)", shadow="0 8px 24px rgba(0,0,0,0.34)",
                 cop="0.15", plotly="plotly_dark", grid="rgba(148,163,184,0.14)"),
    "light": dict(bg="#f4f6fa", solid="#ffffff", panel="#ffffff", panel2="#f1f4f9", field="#ffffff",
                  line="#e3e8ef", line2="#c5cfdc",
                  text="#0f172a", muted="#586579", acc="#1d4ed8", acc2="#4338ca", up="#067647", down="#b42318", mid="#b54708",
                  glow1="rgba(29,78,216,0.05)", glow2="rgba(67,56,202,0.04)", shadow="0 1px 2px rgba(16,24,40,0.06), 0 6px 18px rgba(16,24,40,0.05)",
                  cop="0.07", plotly="plotly_white", grid="rgba(15,23,42,0.08)"),
}


def rgba(hexcol, a):
    h = hexcol.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


@st.cache_data(show_spinner=False)
def candle_uri(seed, up, down, grid, n, tile_w, h=900, thick=0.5):
    """One seamless tile of random-walk candlesticks as an SVG data URI. The increments sum to zero, so the last candle
    closes where the first one opened and the tile repeats without a seam."""
    rnd = random.Random(seed)
    inc = [rnd.gauss(0.0, 1.0) for _ in range(n)]
    mean = sum(inc) / n
    inc = [x - mean for x in inc]
    lv = [0.0]
    for x in inc:
        lv.append(lv[-1] + x)
    lo, hi = min(lv), max(lv)
    span = max(hi - lo, 1e-9)

    def y(v):
        return h * (0.80 - 0.58 * (v - lo) / span)

    step = tile_w / n
    bw = step * thick
    parts = [f"<line x1='0' x2='{tile_w}' y1='{h * k / 8:.0f}' y2='{h * k / 8:.0f}' stroke='{grid}' stroke-width='1' stroke-dasharray='2 10'/>" for k in range(1, 8)]
    for i in range(n):
        o, c = lv[i], lv[i + 1]
        top, bot = max(o, c) + abs(rnd.gauss(0, 0.55)), min(o, c) - abs(rnd.gauss(0, 0.55))
        col = up if c >= o else down
        cx = (i + 0.5) * step
        yo, yc = y(o), y(c)
        parts.append(f"<line x1='{cx:.1f}' x2='{cx:.1f}' y1='{y(top):.1f}' y2='{y(bot):.1f}' stroke='{col}' stroke-width='2.2'/>")
        parts.append(f"<rect x='{cx - bw / 2:.1f}' y='{min(yo, yc):.1f}' width='{bw:.1f}' height='{max(abs(yo - yc), 2.5):.1f}' rx='2' fill='{col}'/>")
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='{tile_w}' height='{h}' viewBox='0 0 {tile_w} {h}' preserveAspectRatio='none'>{''.join(parts)}</svg>"
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


STATIC_CSS = """
html,body{background:var(--bg)!important}
.stApp{background:radial-gradient(1100px 520px at 8% -8%,var(--glow1),transparent 62%),radial-gradient(900px 480px at 100% 0%,var(--glow2),transparent 58%),var(--bg)!important;color:var(--text)!important;font-family:Inter,"Segoe UI",system-ui,-apple-system,Roboto,sans-serif;-webkit-font-smoothing:antialiased}
[data-testid="stAppViewContainer"]{background:transparent!important;position:relative;z-index:1}
[data-testid="stHeader"]{background:transparent!important}
[data-testid="stToolbar"] *{color:var(--muted)!important}
.stDeployButton,.stAppDeployButton,footer{display:none!important}
.block-container,[data-testid="stMainBlockContainer"]{max-width:1240px;padding-top:1.1rem;padding-bottom:3rem}
.stApp::before,.stApp::after{content:"";position:fixed;top:0;left:0;height:100%;width:calc(100% + 1700px);pointer-events:none;z-index:0;background-repeat:repeat-x;-webkit-mask-image:linear-gradient(180deg,#000 0%,rgba(0,0,0,.6) 60%,rgba(0,0,0,.28) 100%);mask-image:linear-gradient(180deg,#000 0%,rgba(0,0,0,.6) 60%,rgba(0,0,0,.28) 100%)}
@keyframes nxA{from{transform:translateX(0)}to{transform:translateX(-1600px)}}
@keyframes nxB{from{transform:translateX(-1000px)}to{transform:translateX(0)}}
@keyframes nxGrow{from{transform:scaleX(0)}to{transform:scaleX(1)}}
@keyframes nxRise{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
@media (prefers-reduced-motion:reduce){.stApp::before,.stApp::after{animation:none!important}.nx-bar,.nx-card{animation:none!important}}
.element-container:has(.nx-css),[data-testid="stElementContainer"]:has(.nx-css),[data-testid="stHtml"]:has(.nx-css){display:none!important}

h1,h2,h3,h4,h5{color:var(--text)!important;letter-spacing:-.01em}
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li,[data-testid="stWidgetLabel"] p,[data-testid="stWidgetLabel"] label,label,summary p,summary span{color:var(--text)!important}
[data-testid="stWidgetLabel"] p{font-weight:600;font-size:.86rem}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] *{color:var(--muted)!important}
a{color:var(--acc)!important}
hr{border-color:var(--line)!important}

/* cards and layout */
.nx-card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px 18px;box-shadow:var(--shadow);-webkit-backdrop-filter:blur(10px);backdrop-filter:blur(10px);transition:border-color .2s ease;animation:nxRise .4s ease-out both}
.nx-card:hover{border-color:var(--line2)}
.nx-grid4{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:10px 0}
.nx-grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:10px 0}
.nx-grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:10px 0}
@media (max-width:980px){.nx-grid4{grid-template-columns:repeat(2,minmax(0,1fr))}.nx-grid3{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:600px){.nx-grid4,.nx-grid3,.nx-grid2{grid-template-columns:minmax(0,1fr)}.nx-verdict{grid-template-columns:minmax(0,1fr)!important}.nx-big{font-size:2.4rem!important}}
.nx-brand{display:flex;align-items:center;gap:14px}
.nx-logo{width:44px;height:44px;flex:0 0 44px;border-radius:11px;background:var(--panel);border:1px solid var(--line2);box-shadow:var(--shadow);display:flex;align-items:center;justify-content:center;gap:6px}
.nx-logo i{position:relative;display:block;width:7px;border-radius:2px}
.nx-logo i::before{content:"";position:absolute;left:50%;top:-6px;bottom:-6px;width:2px;margin-left:-1px;border-radius:2px;background:inherit}
.nx-logo i.up{background:var(--up)}.nx-logo i.dn{background:var(--down)}
.nx-title{font-size:1.65rem;font-weight:800;letter-spacing:-.02em;line-height:1.1;color:var(--text)}
.nx-title b{color:var(--acc)}
.nx-title span{font-weight:600;color:var(--muted);font-size:.95rem;margin-left:6px;letter-spacing:0}
.nx-sub{color:var(--muted);font-size:.9rem;margin-top:3px}
.nx-steps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:6px 0 14px}
@media (max-width:760px){.nx-steps{grid-template-columns:minmax(0,1fr)}}
.nx-step{display:flex;gap:12px;align-items:flex-start;padding:12px 14px;border-radius:12px;background:var(--panel);border:1px solid var(--line)}
.nx-step .n{flex:0 0 26px;height:26px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:.82rem;color:#fff;background:var(--acc)}
.nx-step .t{font-weight:700;color:var(--text)}
.nx-step .d{font-size:.84rem;color:var(--muted)}
.nx-kicker{font-size:.72rem;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);font-weight:700}
.nx-muted{color:var(--muted)}
.nx-h{font-size:.78rem;text-transform:uppercase;letter-spacing:.1em;font-weight:800;color:var(--muted);margin:22px 0 6px;display:flex;align-items:center;gap:10px}
.nx-h::after{content:"";flex:1;height:1px;background:var(--line)}
.nx-company{font-size:1.05rem;font-weight:700;color:var(--text);margin-top:6px}
.nx-hint{font-size:.8rem;color:var(--muted);margin:-6px 0 8px}
.nx-hint b{color:var(--text)}

/* verdict, bars, chips */
.nx-verdict{display:grid;grid-template-columns:minmax(230px,1fr) 2fr;gap:26px;align-items:center}
.nx-big{font-size:3rem;font-weight:800;line-height:1;letter-spacing:-.03em;font-variant-numeric:tabular-nums}
.nx-up{color:var(--up)}.nx-down{color:var(--down)}.nx-mid{color:var(--mid)}.nx-none{color:var(--muted)}
.nx-bar{display:flex;height:12px;border-radius:99px;overflow:hidden;background:var(--panel2);transform-origin:left;animation:nxGrow .8s ease-out both}
.nx-bar i{display:block;height:100%}
.b-down{background:var(--down)}.b-mid{background:var(--mid)}.b-up{background:var(--up)}
.nx-legend{display:flex;justify-content:space-between;font-weight:700;font-size:.84rem;margin-top:6px;font-variant-numeric:tabular-nums}
.nx-legend .dn{color:var(--down)}.nx-legend .md{color:var(--mid)}.nx-legend .upc{color:var(--up)}
.nx-chip{display:inline-flex;align-items:center;gap:6px;padding:3px 11px;border-radius:99px;font-weight:800;font-size:.76rem;letter-spacing:.05em}
.nx-ev{display:inline-block;margin-top:8px;padding:2px 9px;border-radius:99px;font-size:.73rem;font-weight:600;border:1px solid var(--line2);color:var(--muted)}
.nx-ev.ok{color:var(--up);border-color:var(--up)}
.nx-kpi .v{font-size:1.65rem;font-weight:800;font-variant-numeric:tabular-nums;color:var(--text)}
.nx-kpi .l{font-size:.72rem;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);font-weight:700}
.nx-kpi .s{font-size:.82rem;color:var(--muted);margin-top:2px}
.nx-read{font-size:1rem;line-height:1.6;color:var(--text)}
.nx-read b{color:var(--acc)}
.nx-table{width:100%;border-collapse:collapse;font-size:.9rem;font-variant-numeric:tabular-nums}
.nx-table th{color:var(--muted);text-align:left;font-weight:700;font-size:.72rem;text-transform:uppercase;letter-spacing:.07em;padding:8px 10px;border-bottom:1px solid var(--line2)}
.nx-table td{padding:9px 10px;border-bottom:1px solid var(--line);color:var(--text)}
.nx-table tbody tr:hover td{background:var(--panel2)}
.nx-table tr:last-child td{border-bottom:0}
.nx-tablewrap{overflow-x:auto}
.nx-pill{display:inline-block;padding:2px 10px;border-radius:99px;font-size:.78rem;font-weight:600;background:var(--panel2);border:1px solid var(--line);margin:2px 4px 2px 0;color:var(--text)}
.nx-note{font-size:.84rem;color:var(--muted);border-left:3px solid var(--acc);background:var(--panel2);border-radius:0 8px 8px 0;padding:8px 12px;margin:10px 0}
.nx-foot{text-align:center;color:var(--muted);font-size:.78rem;margin-top:30px;padding-top:14px;border-top:1px solid var(--line)}

/* ---- Streamlit widgets (Streamlit 1.67 markup; written for light and dark alike) ---- */
[data-testid="stTextInputRootElement"],[data-testid="stTextAreaRootElement"],[data-testid="stNumberInputContainer"],[data-testid="stSelectbox"] [role="group"]{background:var(--field)!important;border:1px solid var(--line2)!important;border-radius:10px!important;box-shadow:none!important;transition:border-color .15s,box-shadow .15s}
[data-testid="stTextInputRootElement"] [role="group"]{background:transparent!important;border:0!important}
[data-testid="stTextInputRootElement"]:focus-within,[data-testid="stTextAreaRootElement"]:focus-within,[data-testid="stNumberInputContainer"]:focus-within,[data-testid="stSelectbox"] [role="group"]:focus-within{border-color:var(--acc)!important;box-shadow:0 0 0 3px color-mix(in srgb,var(--acc) 22%,transparent)!important}
[data-testid="stTextInputRootElement"] input,[data-testid="stTextAreaRootElement"] textarea,[data-testid="stNumberInputContainer"] input,[data-testid="stSelectbox"] input{color:var(--text)!important;-webkit-text-fill-color:var(--text)!important;caret-color:var(--acc)}
[data-testid="stSelectbox"] [role="group"] svg,[data-testid="stNumberInputContainer"] svg{color:var(--muted)!important;fill:var(--muted)!important}
[data-testid="stSelectbox"] [role="group"] [class*="value"],[data-testid="stSelectbox"] [role="group"] div{color:var(--text)!important}
::placeholder{color:var(--muted)!important;-webkit-text-fill-color:var(--muted)!important;opacity:.85!important}
[data-testid="stSelectboxVirtualDropdown"],[data-testid="stPopoverBody"],[role="listbox"]{background:var(--solid)!important;color:var(--text)!important;border:1px solid var(--line2)!important;border-radius:10px!important;box-shadow:0 12px 32px rgba(0,0,0,.22)!important}
[role="option"]{color:var(--text)!important}
[role="option"]:hover,[role="option"][aria-selected="true"],[role="option"][data-highlighted],[role="option"][data-focused]{background:var(--panel2)!important}
[data-testid="stRadioOption"] > div > div:first-child{background:var(--field)!important;border:1px solid var(--line2)!important}
[data-testid="stRadioOption"]:has(input:checked) > div > div:first-child{background:var(--acc)!important;border-color:var(--acc)!important}
label:has(input[role="switch"]) > div:nth-of-type(1){background:var(--line2)!important}
label:has(input[role="switch"]:checked) > div:nth-of-type(1){background:var(--acc)!important}
label:has(input[role="switch"]) > div:nth-of-type(1) > div{background:#fff!important;box-shadow:0 1px 2px rgba(0,0,0,.3)}
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] *,[data-testid="stSlider"] [data-testid="stSliderThumbValue"]{color:var(--acc)!important}
[data-testid="stSlider"] [data-testid="stSliderTickBar"] *{color:var(--muted)!important}
[data-testid="stSlider"] [role="slider"],[data-testid="stSlider"] [role="group"] > div > div:nth-child(2){background-color:var(--acc)!important}
[data-testid="stFileUploaderDropzone"]{background:var(--field)!important;border:1.5px dashed var(--line2)!important;border-radius:12px!important}
[data-testid="stFileUploaderDropzone"] *{color:var(--muted)!important}
[data-testid="stFileUploaderDropzone"] button{background:var(--panel2)!important;color:var(--text)!important;border:1px solid var(--line2)!important}
[data-testid="stFileUploaderDropzone"] button *{color:var(--text)!important}
[data-testid="stFileUploaderFile"] *{color:var(--text)!important}
[data-testid="stExpander"] details{background:var(--panel)!important;border:1px solid var(--line)!important;border-radius:12px!important;box-shadow:var(--shadow)}
[data-testid="stExpander"] summary,[data-testid="stExpander"] details[open] > summary{background:transparent!important;color:var(--text)!important;font-weight:600;border-radius:12px}
[data-testid="stExpander"] summary:hover{background:var(--panel2)!important}
[data-testid="stExpanderDetails"]{border-color:var(--line)!important}
[role="tablist"]{gap:22px;border-bottom:1px solid var(--line)!important}
[data-testid="stTab"]{color:var(--muted)!important;font-weight:600;padding:8px 2px}
[data-testid="stTab"] p{color:inherit!important;font-weight:600;font-size:.95rem}
[data-testid="stTab"]:hover{color:var(--text)!important}
[data-testid="stTab"][aria-selected="true"]{color:var(--text)!important}
[data-testid="stTab"][aria-selected="true"] > div:not([data-testid]){background:var(--acc)!important}
button[data-testid="stBaseButton-primary"]{background:linear-gradient(135deg,#2563eb,#4f46e5)!important;color:#fff!important;border:0!important;border-radius:10px!important;font-weight:700!important;letter-spacing:.01em;padding:.6rem 1rem!important;box-shadow:0 4px 14px rgba(37,99,235,.30)!important;transition:filter .15s,box-shadow .15s}
button[data-testid="stBaseButton-primary"]:hover{filter:brightness(1.07);box-shadow:0 6px 18px rgba(37,99,235,.38)!important}
button[data-testid="stBaseButton-primary"] *{color:#fff!important}
button[data-testid="stBaseButton-secondary"],button[data-testid="stPopoverButton"]{background:var(--panel)!important;color:var(--text)!important;border:1px solid var(--line2)!important;border-radius:10px!important;font-weight:600!important;transition:border-color .15s,background .15s}
button[data-testid="stBaseButton-secondary"]:hover,button[data-testid="stPopoverButton"]:hover{border-color:var(--acc)!important;background:var(--panel2)!important}
button[data-testid="stBaseButton-secondary"] *,button[data-testid="stPopoverButton"] *{color:var(--text)!important}
[data-testid="stButtonGroup"] button{background:var(--panel)!important;border:1px solid var(--line2)!important;color:var(--text)!important}
[data-testid="stButtonGroup"] button *{color:var(--text)!important}
[data-testid="stButtonGroup"] button[aria-checked="true"],[data-testid="stButtonGroup"] button[aria-pressed="true"],[data-testid="stButtonGroup"] button[kind="segmented_controlActive"]{background:var(--acc)!important;border-color:var(--acc)!important}
[data-testid="stButtonGroup"] button[aria-checked="true"] *,[data-testid="stButtonGroup"] button[aria-pressed="true"] *,[data-testid="stButtonGroup"] button[kind="segmented_controlActive"] *{color:#fff!important}
[data-testid="stImage"] img{border-radius:10px;border:1px solid var(--line)}
[data-testid="stAlertContainer"]{border-radius:10px!important;border:1px solid var(--line2)!important}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]){background:color-mix(in srgb,var(--mid) 12%,var(--solid))!important;border-color:color-mix(in srgb,var(--mid) 45%,transparent)!important}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]){background:color-mix(in srgb,var(--down) 11%,var(--solid))!important;border-color:color-mix(in srgb,var(--down) 45%,transparent)!important}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]){background:color-mix(in srgb,var(--acc) 10%,var(--solid))!important;border-color:color-mix(in srgb,var(--acc) 40%,transparent)!important}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]){background:color-mix(in srgb,var(--up) 11%,var(--solid))!important;border-color:color-mix(in srgb,var(--up) 40%,transparent)!important}
[data-testid="stAlertContainer"] *{color:var(--text)!important}
.nx-sumgrid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px 16px}
@media (max-width:760px){.nx-sumgrid{grid-template-columns:minmax(0,1fr)}}
.nx-sum{display:flex;gap:12px;align-items:flex-start;padding:11px 13px;border-radius:12px;background:var(--panel2);border:1px solid var(--line);animation:nxRise .4s ease-out both}
.nx-sum:last-child{grid-column:1/-1;border:1px solid var(--acc)}
.nx-sum .ic{font-size:1.3rem;line-height:1.2}
.nx-sum .hd{font-weight:800;color:var(--text);font-size:.9rem;margin-bottom:2px}
.nx-sum .tx{color:var(--text);font-size:.92rem;line-height:1.5}
[data-testid="stChatMessage"]{background:var(--panel)!important;border:1px solid var(--line);border-radius:14px;padding:10px 14px;box-shadow:var(--shadow)}
[data-testid="stChatMessageAvatarUser"],[data-testid="stChatMessageAvatarAssistant"]{background:var(--acc)!important;color:#fff!important}
[data-testid="stChatMessageAvatarUser"] *,[data-testid="stChatMessageAvatarAssistant"] *{color:#fff!important}
[data-testid="stChatInput"],[data-testid="stChatInput"] > div{background:var(--field)!important;border-color:var(--line2)!important;border-radius:12px}
[data-testid="stChatInput"] textarea{color:var(--text)!important;-webkit-text-fill-color:var(--text)!important}
[data-testid="stChatInputSubmitButton"]{background:var(--acc)!important;color:#fff!important;border-radius:9px!important}
[data-testid="stChatInputSubmitButton"] *{color:#fff!important}
[data-testid="stChatInputSubmitButton"]:disabled{background:var(--panel2)!important;opacity:.7}
[data-testid="stChatInputSubmitButton"]:disabled *{color:var(--muted)!important}
.nx-qa{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0 12px}
.st-key-inputpanel{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:16px 20px 10px;box-shadow:var(--shadow)}
/* flowchart: how the result was made */
.fl{display:flex;flex-direction:column;align-items:stretch;gap:0;margin:6px 0}
.fl-stage{display:flex;gap:14px;align-items:flex-start;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:12px 16px;box-shadow:var(--shadow)}
.fl-stage.fl-final{border-color:var(--acc)}
.fl-n{flex:0 0 28px;height:28px;border-radius:50%;background:var(--acc);color:#fff;font-weight:800;font-size:.85rem;display:flex;align-items:center;justify-content:center}
.fl-b{flex:1;min-width:0}
.fl-t{font-weight:800;color:var(--text);margin-bottom:4px}
.fl-d{font-size:.88rem;line-height:1.5;color:var(--muted)}
.fl-d b{color:var(--text)}
.fl-res{color:var(--acc)}
.fl-arrow{text-align:center;color:var(--acc);font-size:1.05rem;line-height:1;padding:5px 0}
.fl-two{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
@media (max-width:900px){.fl-two{grid-template-columns:minmax(0,1fr)}}
.fl-col{display:flex;flex-direction:column;gap:6px}
.fl-col.fl-dim{opacity:.75}
.fl-ct{font-weight:800;font-size:.9rem;color:var(--text);margin-bottom:2px}
.fl-m{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:7px 11px;border-radius:10px;border:1px solid var(--line);background:var(--panel2);font-size:.85rem;cursor:help}
.fl-m.on{border-color:var(--acc);background:color-mix(in srgb,var(--acc) 12%,var(--solid))}
.fl-m.off{opacity:.45}
.fl-mt{font-weight:700;color:var(--text)}
.fl-mv{font-variant-numeric:tabular-nums;color:var(--muted);font-size:.8rem;text-align:right}
.fl-m.on .fl-mv{color:var(--text);font-weight:700}
.fl-rank{display:inline-block;margin-right:7px;padding:0 6px;border-radius:99px;background:var(--acc);color:#fff;font-size:.7rem;font-weight:800}
.fl-three{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
@media (max-width:900px){.fl-three{grid-template-columns:minmax(0,1fr)}}
.fl-o{font-size:.84rem;line-height:1.5;color:var(--muted);background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:9px 12px}
.fl-o b{color:var(--text)}
.fl-mx th.fl-vh{writing-mode:vertical-rl;transform:rotate(180deg);height:112px;text-align:left;vertical-align:bottom;padding:6px 4px;text-transform:none;letter-spacing:0;font-size:.74rem}
.fl-mx td.fl-c{text-align:center;padding:6px 4px}
.fl-mx .fl-sep{width:14px;padding:0;border-bottom:0}
.fl-dot,.fl-dot2{display:inline-block;width:12px;height:12px;border-radius:50%}
.fl-dot{background:var(--acc)}
.fl-dot2{background:var(--up)}
.fl-mx tr.fl-focus td{background:color-mix(in srgb,var(--acc) 12%,transparent)}
/* architecture: interactive pipeline with hover pop-ups */
.ar-flow{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:20px;margin:8px 0 6px;align-items:start}
.ar-flow.ar-live{grid-template-columns:repeat(4,minmax(0,1fr))}
@media (max-width:1000px){.ar-flow,.ar-flow.ar-live{grid-template-columns:minmax(0,1fr)}}
.ar-lane{position:relative;display:flex;flex-direction:column;gap:10px}
.ar-lane:not(:last-child)::after{content:"›";position:absolute;right:-15px;top:13px;color:var(--acc);font-size:1.7rem;font-weight:800;line-height:1}
@media (max-width:1000px){.ar-lane:not(:last-child)::after{display:none}}
.ar-lh{display:flex;gap:10px;align-items:center;padding:10px 12px;border-radius:12px;background:linear-gradient(135deg,#2563eb,#4f46e5);color:#fff}
.ar-ln{flex:0 0 26px;height:26px;border-radius:50%;background:rgba(255,255,255,.22);display:flex;align-items:center;justify-content:center;font-weight:800;font-size:.82rem}
.ar-lh .ar-lt{font-weight:800;font-size:.95rem;color:#fff}
.ar-lh .ar-ls{font-size:.74rem;color:rgba(255,255,255,.85)}
.ar-node{position:relative;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:10px 12px;cursor:help;outline:none;transition:border-color .15s,box-shadow .15s,transform .15s}
.ar-node:hover,.ar-node:focus{border-color:var(--acc);box-shadow:0 0 0 3px color-mix(in srgb,var(--acc) 18%,transparent);z-index:60;transform:translateY(-1px)}
.ar-nt{font-weight:700;color:var(--text);font-size:.92rem}
.ar-ns{font-size:.78rem;color:var(--muted);margin-top:1px}
.ar-tag{display:inline-block;margin-top:6px;padding:1px 8px;border-radius:99px;font-size:.7rem;font-weight:700;background:var(--panel2);border:1px solid var(--line2);color:var(--muted)}
.ar-tip{position:absolute;top:calc(100% + 8px);left:0;width:340px;max-width:88vw;opacity:0;visibility:hidden;transform:translateY(4px);transition:opacity .15s,transform .15s,visibility .15s;pointer-events:none;background:var(--solid);color:var(--text);border:1px solid var(--acc);border-radius:12px;padding:12px 14px;box-shadow:0 18px 40px rgba(0,0,0,.35);font-size:.84rem;line-height:1.5;z-index:80;text-align:left;cursor:default;font-weight:400}
.ar-node:hover .ar-tip,.ar-node:focus .ar-tip{opacity:1;visibility:visible;transform:none}
.ar-mid .ar-tip{left:50%;margin-left:-170px}
.ar-right .ar-tip{left:auto;right:0}
.ar-tt{font-weight:800;margin-bottom:5px;color:var(--acc);font-size:.9rem}
.ar-tip p{margin:0 0 6px;color:var(--text)}
.ar-tip ul{margin:4px 0 0;padding-left:18px;color:var(--text)}
.ar-tip li{margin:2px 0}
.ar-tip b,.ar-tip i{color:var(--text)}
.ar-pred{display:flex;flex-direction:column;gap:14px;margin:8px 0}
.ar-prow{display:grid;grid-template-columns:190px repeat(4,minmax(0,1fr));gap:14px;align-items:stretch}
@media (max-width:900px){.ar-prow{grid-template-columns:repeat(2,minmax(0,1fr))}.ar-ph{grid-column:1/-1}}
.ar-ph{display:flex;flex-direction:column;justify-content:center}
.ar-ph .ar-lt{font-weight:800;font-size:1.05rem;color:var(--text)}
.ar-ph .ar-ls{font-size:.76rem;color:var(--muted)}
.ar-prow .ar-node:nth-child(n+4) .ar-tip{left:auto;right:0}
.ar-ok{border-color:var(--up)}
.ar-ok .ar-tag{color:var(--up);border-color:var(--up);background:color-mix(in srgb,var(--up) 10%,transparent)}
.st-key-nx_light label,.st-key-nx_light p{white-space:nowrap}
"""


def build_css(T, motion):
    ua = candle_uri(11, T["up"], T["down"], T["grid"], 64, 1600)
    ub = candle_uri(29, T["acc"], T["acc2"], T["grid"], 26, 1000, thick=0.62)
    head = (":root{" + "".join(f"--{k}:{T[k]};" for k in ("bg", "solid", "panel", "panel2", "field", "line", "line2", "text", "muted", "acc", "acc2", "up", "down", "mid", "glow1", "glow2", "shadow", "cop")) + "}"
            f'.stApp::before{{background-image:url("{ua}");background-size:1600px 100%;opacity:var(--cop);animation:nxA 150s linear infinite}}'
            f'.stApp::after{{background-image:url("{ub}");background-size:1000px 100%;opacity:calc(var(--cop) * .55);animation:nxB 95s linear infinite}}')
    if not motion:
        head += ".stApp::before,.stApp::after{animation:none!important}"
    return "<style>" + head + "".join(line.strip() for line in STATIC_CSS.splitlines()) + "</style>"


def H(s):
    """Render HTML (not markdown) with line breaks and indentation removed."""
    st.html(" ".join(line.strip() for line in s.strip().splitlines()))


# ------------------------------------------------------------------ header and settings (these widgets decide the theme, so they come first)
head = st.columns([6, 1.8, 1.3], vertical_alignment="center")
with head[0]:
    H("""<div class="nx-brand">
    <div class="nx-logo"><i class="dn" style="height:13px"></i><i class="up" style="height:17px"></i><i class="up" style="height:14px"></i></div>
    <div><div class="nx-title">NEXUS <b>Alpha</b> v3<span>Research dashboard</span></div>
    <div class="nx-sub">Chart images and SEC filings, read by 13 AI models: direction, volatility and drawdown risk</div></div></div>""")
with head[1]:
    light = st.toggle("Light mode", key="nx_light", help="Switch between the dark and light look.", bind="query-params")
with head[2]:
    with st.popover("Settings"):
        motion = st.toggle("Moving candles", value=True, key="nx_motion", help="Slow drifting candlestick background. Turn off to save battery.")
        n_paths = st.select_slider("Simulated paths", [1000, 2000, 4000, 8000, 16000], value=4000, key="nx_paths")
        n_show = st.slider("Paths drawn on the chart", 0, 100, 30, step=10, key="nx_show")
        seed = int(st.number_input("Random seed", 0, 10000, 7, key="nx_seed"))
        lowmem = st.toggle("Low-memory mode", value=True, key="nx_lowmem",
                           help="Loads the language models one at a time and frees each after use. Same results, slower with filing text. Turn off only on a machine with 32 GB of RAM or a large GPU.")
        E._LOWMEM = bool(lowmem)
        if lowmem:                                   # switching it on also frees language models kept from earlier runs
            stale = [k for k in E._CACHE if k[0] == "nlp"]
            for k_ in stale:
                E._CACHE.pop(k_, None)
            if stale:
                gc.collect()
T = THEMES["light" if light else "dark"]
st.html(build_css(T, motion))

# ------------------------------------------------------------------ small helpers
CALL_META = {"UP": ("▲", "up", "UP"), "DOWN": ("▼", "down", "DOWN"), "SIDEWAYS": ("◆", "mid", "SIDEWAYS"), "NO CALL": ("○", "none", "NO CALL")}
HORIZON_TXT = {5: "about 1 week", 10: "about 2 weeks", 20: "about 1 month", 40: "about 2 months"}


# ------------------------------------------------------------------ company names
COMPANIES = {
    # the 50 stocks the models were trained on
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "NVDA": "NVIDIA Corporation", "TXN": "Texas Instruments", "INTC": "Intel Corporation",
    "CSCO": "Cisco Systems", "ORCL": "Oracle Corporation", "QCOM": "Qualcomm", "AMD": "Advanced Micro Devices", "ADBE": "Adobe Inc.",
    "JPM": "JPMorgan Chase & Co.", "BAC": "Bank of America", "WFC": "Wells Fargo & Company", "C": "Citigroup Inc.", "GS": "The Goldman Sachs Group",
    "MS": "Morgan Stanley", "V": "Visa Inc.", "MA": "Mastercard Incorporated", "BRK.B": "Berkshire Hathaway (Class B)", "SCHW": "The Charles Schwab Corporation",
    "UNH": "UnitedHealth Group", "JNJ": "Johnson & Johnson", "LLY": "Eli Lilly and Company", "PFE": "Pfizer Inc.", "MRK": "Merck & Co.",
    "ABT": "Abbott Laboratories", "TMO": "Thermo Fisher Scientific", "AMGN": "Amgen Inc.", "GILD": "Gilead Sciences", "CVS": "CVS Health",
    "AMZN": "Amazon.com", "HD": "The Home Depot", "MCD": "McDonald's Corporation", "NKE": "Nike Inc.", "SBUX": "Starbucks Corporation",
    "F": "Ford Motor Company", "LOW": "Lowe's Companies", "TJX": "The TJX Companies", "MAR": "Marriott International", "ROST": "Ross Stores",
    "GOOG": "Alphabet Inc. (Class C)", "NFLX": "Netflix Inc.", "DIS": "The Walt Disney Company", "T": "AT&T Inc.", "VZ": "Verizon Communications",
    "META": "Meta Platforms", "OMC": "Omnicom Group", "TTWO": "Take-Two Interactive Software", "NYT": "The New York Times Company", "LYV": "Live Nation Entertainment",
    # other well-known US stocks, so that a typed ticker also shows its company
    "GOOGL": "Alphabet Inc. (Class A)", "TSLA": "Tesla Inc.", "AVGO": "Broadcom Inc.", "CRM": "Salesforce", "UBER": "Uber Technologies", "PYPL": "PayPal Holdings",
    "COST": "Costco Wholesale", "WMT": "Walmart Inc.", "KO": "The Coca-Cola Company", "PEP": "PepsiCo", "PG": "Procter & Gamble", "XOM": "Exxon Mobil",
    "CVX": "Chevron Corporation", "BA": "The Boeing Company", "CAT": "Caterpillar Inc.", "GE": "GE Aerospace", "MMM": "3M Company", "IBM": "IBM",
    "NOW": "ServiceNow", "INTU": "Intuit Inc.", "AMAT": "Applied Materials", "MU": "Micron Technology", "LRCX": "Lam Research", "KLAC": "KLA Corporation",
    "PLTR": "Palantir Technologies", "SNOW": "Snowflake Inc.", "COIN": "Coinbase Global", "ABNB": "Airbnb Inc.", "BKNG": "Booking Holdings", "TGT": "Target Corporation",
    "CMG": "Chipotle Mexican Grill", "DE": "Deere & Company", "LMT": "Lockheed Martin", "RTX": "RTX Corporation", "HON": "Honeywell International", "UPS": "United Parcel Service",
    "FDX": "FedEx Corporation", "BLK": "BlackRock Inc.", "AXP": "American Express", "SPGI": "S&P Global", "MDT": "Medtronic", "ISRG": "Intuitive Surgical",
    "VRTX": "Vertex Pharmaceuticals", "REGN": "Regeneron Pharmaceuticals", "BMY": "Bristol-Myers Squibb", "ABBV": "AbbVie Inc.", "DHR": "Danaher Corporation",
    "CMCSA": "Comcast Corporation", "TMUS": "T-Mobile US", "SPY": "SPDR S&P 500 ETF", "QQQ": "Invesco QQQ Trust",
}


def _norm_tk(tk):
    return str(tk or "").strip().upper().replace("-", ".")


def company_name(tk):
    """Company name for a ticker: the built-in list first, then Yahoo Finance (cached), else an empty string."""
    t = _norm_tk(tk)
    if t in COMPANIES:
        return COMPANIES[t]
    return _yahoo_name(t) if t else ""


@st.cache_data(show_spinner=False, ttl=86400)
def _yahoo_name(t):
    try:
        import yfinance as yf
        info = yf.Ticker(t.replace(".", "-")).get_info()
        return str(info.get("longName") or info.get("shortName") or "")
    except Exception:
        return ""


def resolve_ticker(text):
    """Accept a ticker or a company name ('apple', 'nvidia', 'jp morgan'). Returns (ticker, company or '')."""
    raw = str(text or "").strip()
    if not raw:
        return "", ""
    t = _norm_tk(raw)
    if t in COMPANIES:
        return t, COMPANIES[t]
    key = "".join(ch for ch in raw.lower() if ch.isalnum() or ch == " ").strip()
    if len(key) >= 3:
        def clean(s):
            return "".join(ch for ch in s.lower() if ch.isalnum() or ch == " ")
        hits = [k for k, v in COMPANIES.items() if clean(v).startswith(key) or (" " + key) in (" " + clean(v))]
        if len(hits) == 1:
            return hits[0], COMPANIES[hits[0]]
        exact = [k for k in hits if clean(COMPANIES[k]).split(" ")[0] == key.split(" ")[0] and k in B["universe"]]
        if len(exact) == 1:
            return exact[0], COMPANIES[exact[0]]
    return raw.upper(), ""


def stock_label(t):
    """Drop-down label: ticker beside the company name."""
    return "Choose a stock…" if t == "—" else f"{t}  ·  {COMPANIES.get(t, '')}"


# ------------------------------------------------------------------ paste-a-screenshot component (Streamlit has no paste support in its file uploader)
_PASTE_HTML = """
<div class="wrap">
  <div class="zone" tabindex="0" role="button" aria-label="Paste a screenshot here">
    <div class="ic"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="8" y="3" width="8" height="4" rx="1"/><path d="M16 5h2a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h2"/><path d="M9 14l2 2 4-4"/></svg></div>
    <div class="t"></div>
    <div class="d">Click here, then press <kbd>Ctrl</kbd>+<kbd>V</kbd></div>
  </div>
  <button type="button" class="btn">Paste from clipboard</button>
  <div class="msg"></div>
</div>
"""

_PASTE_CSS = """
.wrap{display:flex;flex-direction:column;gap:8px;font-family:Inter,"Segoe UI",system-ui,sans-serif}
.zone{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px;min-height:112px;padding:12px;border:1.5px dashed var(--line2);border-radius:12px;background:var(--field);color:var(--muted);text-align:center;cursor:pointer;outline:none;transition:border-color .15s,background .15s,box-shadow .15s}
.zone:hover{border-color:var(--acc)}
.zone:focus,.zone.over{border-color:var(--acc);background:var(--accsoft);box-shadow:0 0 0 3px var(--ring)}
.zone.has{border-style:solid;border-color:var(--up)}
.zone .ic{color:var(--acc);line-height:0;margin-bottom:2px}
.zone.has .ic{color:var(--up)}
.zone .t{font-weight:700;font-size:.92rem;color:var(--text)}
.zone .d{font-size:.76rem;line-height:1.4}
kbd{font:600 .72rem Inter,system-ui,sans-serif;padding:1px 6px;border:1px solid var(--line2);border-bottom-width:2px;border-radius:5px;background:var(--panel2);color:var(--text)}
.btn{font:600 .82rem Inter,system-ui,sans-serif;padding:7px 10px;border-radius:9px;border:1px solid var(--line2);background:var(--panel);color:var(--text);cursor:pointer;transition:border-color .15s,background .15s}
.btn:hover{border-color:var(--acc);background:var(--panel2)}
.msg{min-height:16px;font-size:.78rem;color:var(--muted);text-align:center}
.msg.ok{color:var(--up)}.msg.bad{color:var(--down)}
"""

_PASTE_JS = """
export default function (component) {
  const { data, parentElement, setStateValue } = component
  const wrap = parentElement.querySelector(".wrap")
  const zone = parentElement.querySelector(".zone")
  const title = parentElement.querySelector(".t")
  const btn = parentElement.querySelector(".btn")
  const msg = parentElement.querySelector(".msg")
  if (!wrap || !zone) return
  const th = (data && data.theme) || {}
  for (const k of Object.keys(th)) wrap.style.setProperty("--" + k, th[k])
  const slot = (data && data.slot) || ""
  const say = (t, cls) => { msg.textContent = t || ""; msg.className = "msg" + (cls ? " " + cls : "") }
  title.textContent = slot + " screenshot"
  zone.classList.toggle("has", !!(data && data.has))
  if (data && data.has) { title.textContent = slot + " screenshot added"; say("Press Ctrl+V again to replace it", "ok") }
  else say("")

  const toUrl = (blob) => new Promise((res, rej) => { const fr = new FileReader(); fr.onload = () => res(fr.result); fr.onerror = rej; fr.readAsDataURL(blob) })
  const prepare = async (blob) => {
    if (blob.size < 3500000) return toUrl(blob)
    const bmp = await createImageBitmap(blob)
    const s = Math.min(1, 2600 / Math.max(bmp.width, bmp.height))
    const c = document.createElement("canvas")
    c.width = Math.round(bmp.width * s); c.height = Math.round(bmp.height * s)
    c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height)
    return c.toDataURL("image/png")
  }
  const take = async (blob) => {
    if (!blob || !String(blob.type).startsWith("image/")) { say("That was not an image. Copy a screenshot first.", "bad"); return }
    try {
      say("Reading the image...")
      const url = await prepare(blob)
      setStateValue("image", url)
      say("Pasted. Loading the preview...", "ok")
    } catch (e) { say("Could not read that image.", "bad") }
  }
  zone.onclick = () => zone.focus()
  zone.onpaste = (e) => {
    const items = (e.clipboardData && e.clipboardData.items) || []
    for (const it of items) {
      if (it.kind === "file" && String(it.type).startsWith("image/")) { e.preventDefault(); take(it.getAsFile()); return }
    }
    say("No image in the clipboard. Copy a screenshot first (for example Win + Shift + S).", "bad")
  }
  zone.ondragover = (e) => { e.preventDefault(); zone.classList.add("over") }
  zone.ondragleave = () => zone.classList.remove("over")
  zone.ondrop = (e) => {
    e.preventDefault(); zone.classList.remove("over")
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]
    if (f) take(f)
  }
  btn.onclick = async () => {
    try {
      const items = await navigator.clipboard.read()
      for (const it of items) {
        const t = it.types.find((x) => x.startsWith("image/"))
        if (t) { take(await it.getType(t)); return }
      }
      say("No image in the clipboard. Copy a screenshot first.", "bad")
    } catch (e) { say("The browser blocked clipboard access. Click the box and press Ctrl+V instead.", "bad") }
  }
}
"""

_PASTE = st.components.v2.component("nx_paste_zone", html=_PASTE_HTML, css=_PASTE_CSS, js=_PASTE_JS)


def paste_zone(slot, key, has_image):
    """A click-and-Ctrl+V box. Returns the pasted image as bytes, or None."""
    theme = {k: T[k] for k in ("field", "line2", "acc", "up", "down", "text", "muted", "panel", "panel2")}
    theme["accsoft"] = rgba(T["acc"], 0.10)
    theme["ring"] = rgba(T["acc"], 0.25)
    res = _PASTE(key=key, data={"slot": slot, "has": bool(has_image), "theme": theme}, on_image_change=lambda: None)
    url = getattr(res, "image", None)
    if isinstance(url, str) and "," in url:
        try:
            return base64.b64decode(url.split(",", 1)[1])
        except Exception:
            return None
    return None


def to_pil(src):
    """An uploaded file or pasted bytes -> an RGB PIL image."""
    return Image.open(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src).convert("RGB")


def pct(v, d=0):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:.{d}f}%"


def chip(call):
    sym, tone, label = CALL_META.get(call, CALL_META["NO CALL"])
    if tone == "none":
        return f'<span class="nx-chip" style="color:var(--muted);background:var(--panel2)">{sym} {label}</span>'
    col = T[tone]
    return f'<span class="nx-chip" style="color:{col};background:{rgba(col, 0.14)}">{sym} {label}</span>'


def pbar(pdn, psd, pup):
    return (f'<div class="nx-bar"><i class="b-down" style="width:{pdn * 100:.1f}%"></i><i class="b-mid" style="width:{psd * 100:.1f}%"></i><i class="b-up" style="width:{pup * 100:.1f}%"></i></div>'
            f'<div class="nx-legend"><span class="dn">▼ {pdn:.0%}</span><span class="md">◆ {psd:.0%}</span><span class="upc">▲ {pup:.0%}</span></div>')


def evidence_badge(task, ok_text, no_text):
    s = SK.get(task)
    if not s:
        return ""
    tip = esc(f"Locked test 2024-26: skill {s['skill']:+.3f}, 95% interval [{s['lo']:+.3f}, {s['hi']:+.3f}], Holm p = {s['p_holm']:.3f}")
    if s.get("supported"):
        return f'<span class="nx-ev ok" title="{tip}">✓ {ok_text}</span>'
    return f'<span class="nx-ev" title="{tip}">○ {no_text}</span>'


def kpi(label, value, sub="", badge=""):
    return f'<div class="nx-card nx-kpi"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div>{badge}</div>'


def row_at(r, h):
    t = r["table"]
    return t[t["horizon (sessions)"] == h].iloc[0]


def summary_points(r, pats, n_bars):
    """The analysis in simple words: (icon, heading, sentence). Plain text, so the same points feed the page and the AI chat."""
    row, sim = row_at(r, 20), r["sim"]
    ps = {"UP": float(row["P(UP)"]), "SIDEWAYS": float(row["P(SIDEWAYS)"]), "DOWN": float(row["P(DOWN)"])}
    order = sorted(ps, key=lambda k: -ps[k])
    gap = ps[order[0]] - ps[order[1]]
    strength = "close to a coin toss" if gap < 0.05 else ("a slight lean" if gap < 0.15 else "a clearer lean")
    head_ = ("the models see no clear direction, close to a coin toss" if gap < 0.05 else f"the models show {strength} towards {order[0]}")
    pts = [("🧭", "What the models think",
            f"Over the next month (20 trading sessions) {head_}: {ps['UP']:.0%} up, {ps['SIDEWAYS']:.0%} sideways, {ps['DOWN']:.0%} down.")]
    s20 = SK.get("dir_20")
    if s20 and s20.get("supported"):
        pts.append(("⚖️", "How much to trust it", "On the untouched 2024-2026 test data, calls like this were right only a little more often than chance. Treat it as a nudge, not a forecast."))
    else:
        pts.append(("⚖️", "How much to trust it", "On the untouched 2024-2026 test data, calls like this were not better than chance. Treat the lean as weak."))
    realised, exp_vol = r["sigma"] * math.sqrt(252), float(row["expected vol (ann.)"])
    ratio = exp_vol / max(realised, 1e-9)
    vs = ("calmer than" if ratio < 0.85 else "slightly calmer than" if ratio < 0.95 else "about as jumpy as" if ratio <= 1.05
          else "slightly more jumpy than" if ratio <= 1.15 else "more jumpy than")
    dd = float(row["P(drawdown > 10%)"])
    dd_word = "low" if dd < 0.10 else ("moderate" if dd < 0.25 else "high")
    pts.append(("🌊", "Risk", f"Expected price swings are about {exp_vol:.0%} a year, {vs} the last three months ({realised:.0%}). "
                               f"The chance of a fall of more than 10% within a month is {dd:.0%} ({dd_word})."))
    q = sim["quantiles"]
    pts.append(("🎯", "Likely price range", f"From ${r['p0']:,.2f} today, most simulated paths end between ${q[1][-1]:,.2f} and ${q[3][-1]:,.2f} in a month "
                                           f"(middle value ${q[2][-1]:,.2f}); 9 in 10 end between ${q[0][-1]:,.2f} and ${q[4][-1]:,.2f}."))
    if r.get("used_text"):
        cv_up, fin_up = float(row["CV only P(UP)"]), float(row["P(UP)"])
        w_dir = float(B.get("w_fresh", {}).get("dir_20", 1.0))
        pts.append(("📄", "What the filing text added", f"The charts alone gave {cv_up:.0%} for UP; with the filing text the final answer is {fin_up:.0%}. "
                                                       + ("At this horizon the blend gives the text no weight for direction, so the text mainly changes the risk numbers."
                                                          if w_dir >= 0.999 else f"At this horizon the text gets {1 - w_dir:.0%} of the weight for direction; the charts get the rest.")))
    else:
        pts.append(("📄", "Filing text", "No filing summary was given, so this outlook comes from the charts only."))
    recent = [p for p in pats if p["i"] >= n_bars - 10]
    if recent:
        last = recent[-1]
        pts.append(("🕯️", "Recent candlestick pattern", f"{last['name']} on {last['date'].date()} ({last['bias']}). "
                                                        "Shown for context only: the models do not use it and the study did not test it."))
    if gap < 0.05 or order[0] == "SIDEWAYS":
        lean = "no clear direction (close to a coin toss)"
    else:
        lean = ("a slight " if gap < 0.15 else "a clearer ") + ("positive lean" if order[0] == "UP" else "negative lean")
    pts.append(("✅", "Bottom line", f"{lean.capitalize()} with {dd_word} short-term drawdown risk. Use it as one input among many, not as a trading signal. Educational research, not financial advice."))
    return pts


def summary_html(pts):
    items = "".join(f'<div class="nx-sum"><div class="ic">{ic}</div><div><div class="hd">{esc(hd)}</div><div class="tx">{esc(tx)}</div></div></div>' for ic, hd, tx in pts)
    return f'<div class="nx-card"><div class="nx-kicker" style="margin-bottom:8px">Summary in simple words</div><div class="nx-sumgrid">{items}</div></div>'


@st.cache_data(show_spinner=False, ttl=900)
def history(ticker, asof):
    d, _ = E.fetch_market(ticker, asof)
    return d.tail(90).reset_index(drop=True)


def safe_history(r):
    if r.get("D_screen") is not None:
        return r["D_screen"]
    try:
        return history(r["ticker"], r["asof"])
    except Exception:
        return None


# ------------------------------------------------------------------ candlestick patterns (descriptive only: not a model input, not tested in the study)
PATTERN_INFO = {
    "Doji": ("neutral", "Open and close almost equal: buyers and sellers ended the day in balance."),
    "Hammer": ("bullish", "Long lower shadow after a fall: sellers pushed the price down but buyers took it back up."),
    "Hanging man": ("bearish", "Hammer shape after a rise: a sign that selling pressure appeared during an uptrend."),
    "Inverted hammer": ("bullish", "Long upper shadow after a fall: buyers tried to push the price up."),
    "Shooting star": ("bearish", "Long upper shadow after a rise: buyers pushed up but sellers drove the price back down."),
    "Bullish engulfing": ("bullish", "A rising candle whose body covers the previous falling candle's body."),
    "Bearish engulfing": ("bearish", "A falling candle whose body covers the previous rising candle's body."),
    "Morning star": ("bullish", "Three candles: a long fall, a small pause, then a strong rise."),
    "Evening star": ("bearish", "Three candles: a long rise, a small pause, then a strong fall."),
    "Three white soldiers": ("bullish", "Three long rising candles in a row, each closing higher."),
    "Three black crows": ("bearish", "Three long falling candles in a row, each closing lower."),
}


def detect_patterns(D):
    """Classic textbook candlestick patterns on daily bars. Rules are the standard definitions; trend = close versus 5 sessions earlier."""
    if D is None or len(D) < 8:
        return []
    o, h, l, c = (D[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    body, rng = np.abs(c - o), np.maximum(h - l, 1e-9)
    upper, lower = h - np.maximum(o, c), np.minimum(o, c) - l
    avg = pd.Series(body).rolling(10, min_periods=5).mean().bfill().to_numpy()
    out = []

    def add(i, name):
        bias = PATTERN_INFO[name][0]
        out.append({"i": i, "date": pd.Timestamp(D["date"].iloc[i]), "name": name, "bias": bias,
                    "y": float(h[i]) * 1.012 if bias != "bearish" else float(l[i]) * 0.988})
    for i in range(5, len(D)):
        down, up = c[i - 1] < c[i - 5], c[i - 1] > c[i - 5]
        small = body[i] <= 0.30 * avg[i]
        if body[i] <= 0.08 * rng[i]:
            add(i, "Doji")
        elif lower[i] >= 2 * body[i] and upper[i] <= 0.25 * rng[i] and body[i] > 0:
            if down:
                add(i, "Hammer")
            elif up:
                add(i, "Hanging man")
        elif upper[i] >= 2 * body[i] and lower[i] <= 0.25 * rng[i] and body[i] > 0:
            if down:
                add(i, "Inverted hammer")
            elif up:
                add(i, "Shooting star")
        if c[i - 1] < o[i - 1] and c[i] > o[i] and o[i] <= c[i - 1] and c[i] >= o[i - 1] and body[i] > body[i - 1]:
            add(i, "Bullish engulfing")
        if c[i - 1] > o[i - 1] and c[i] < o[i] and o[i] >= c[i - 1] and c[i] <= o[i - 1] and body[i] > body[i - 1]:
            add(i, "Bearish engulfing")
        if i >= 2 and body[i - 1] <= 0.30 * avg[i - 1] and not small:
            if c[i - 2] < o[i - 2] and body[i - 2] >= 0.6 * avg[i - 2] and c[i] > o[i] and c[i] > (o[i - 2] + c[i - 2]) / 2:
                add(i, "Morning star")
            if c[i - 2] > o[i - 2] and body[i - 2] >= 0.6 * avg[i - 2] and c[i] < o[i] and c[i] < (o[i - 2] + c[i - 2]) / 2:
                add(i, "Evening star")
        if i >= 2 and all(c[j] > o[j] and body[j] >= 0.6 * avg[j] for j in (i - 2, i - 1, i)) and c[i] > c[i - 1] > c[i - 2]:
            add(i, "Three white soldiers")
        if i >= 2 and all(c[j] < o[j] and body[j] >= 0.6 * avg[j] for j in (i - 2, i - 1, i)) and c[i] < c[i - 1] < c[i - 2]:
            add(i, "Three black crows")
    return out


def style_fig(fig, height):
    fig.update_layout(template=T["plotly"], paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=height,
                      font=dict(color=T["text"], family="Inter, Segoe UI, system-ui, sans-serif"), margin=dict(l=8, r=8, t=64, b=8),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1), hoverlabel=dict(bgcolor=T["solid"], font_color=T["text"]))
    fig.update_xaxes(gridcolor=T["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=T["grid"], zeroline=False)
    return fig


PLOT_CFG = {"displaylogo": False, "scrollZoom": True, "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]}


def outlook_chart(r, D, pats, show_patterns=True):
    """Recent daily candles joined to the 20-session Monte Carlo cone, with candlestick patterns marked.
    Drag to pan, scroll to zoom, or press Reset view."""
    sim, p0 = r["sim"], r["p0"]
    fig = go.Figure()
    if D is not None and len(D):
        fig.add_trace(go.Candlestick(x=D["date"], open=D["open"], high=D["high"], low=D["low"], close=D["close"], name="daily price",
                                     increasing_line_color=T["up"], decreasing_line_color=T["down"],
                                     increasing_fillcolor=T["up"], decreasing_fillcolor=T["down"]))
        start = pd.Timestamp(D["date"].iloc[-1])
        if show_patterns and pats:
            for bias, sym, col, nm in (("bullish", "triangle-up", T["up"], "bullish pattern"), ("bearish", "triangle-down", T["down"], "bearish pattern"),
                                       ("neutral", "circle", T["mid"], "neutral pattern (doji)")):
                ps_ = [p for p in pats if p["bias"] == bias]
                if ps_:
                    fig.add_trace(go.Scatter(x=[p["date"] for p in ps_], y=[p["y"] for p in ps_], mode="markers", name=nm,
                                             marker=dict(symbol=sym, size=11, color=col, line=dict(color=T["solid"], width=1)),
                                             text=[f"<b>{p['name']}</b><br>{PATTERN_INFO[p['name']][1]}<br><i>descriptive only, not used by the models</i>" for p in ps_],
                                             hovertemplate="%{x|%d %b %Y}<br>%{text}<extra></extra>"))
    else:
        start = pd.Timestamp(r["asof"])
    fut = list(pd.bdate_range(start + pd.Timedelta(days=1), periods=20))
    x = [start] + fut
    q = sim["quantiles"]
    for k in range(min(int(n_show), sim["paths"].shape[0])):
        fig.add_trace(go.Scatter(x=x, y=[p0] + list(sim["paths"][k]), mode="lines", line=dict(width=0.6, color=rgba(T["muted"], 0.30)),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=[p0] + list(q[4]), line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=[p0] + list(q[0]), fill="tonexty", fillcolor=rgba(T["acc"], 0.14), line=dict(width=0), name="5-95 % of outcomes"))
    fig.add_trace(go.Scatter(x=x, y=[p0] + list(q[3]), line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=[p0] + list(q[1]), fill="tonexty", fillcolor=rgba(T["acc"], 0.28), line=dict(width=0), name="middle 50 %"))
    fig.add_trace(go.Scatter(x=x, y=[p0] + list(q[2]), line=dict(color=T["acc"], width=2.4), name="median path"))
    fig.add_hline(y=p0 * 0.9, line_dash="dot", line_color=T["down"], annotation_text="-10 % drawdown line", annotation_font_color=T["down"])
    fig.add_shape(type="line", x0=start.isoformat(), x1=start.isoformat(), y0=0, y1=1, yref="paper", line=dict(color=T["muted"], width=1, dash="dash"))
    fig.update_layout(title=dict(text=f"{esc(r['ticker'])}{(' (' + esc(company_name(r['ticker'])) + ')') if company_name(r['ticker']) else ''}: recent candles and the next 20 sessions ({sim['paths'].shape[0]:,} simulated paths)", font=dict(size=14)),
                      dragmode="pan", xaxis_rangeslider_visible=False, yaxis_title="price ($)", hovermode="x unified")
    fig.update_xaxes(range=[(start - pd.Timedelta(days=100)).isoformat(), (fut[-1] + pd.Timedelta(days=3)).isoformat()],
                     rangeselector=dict(buttons=[dict(count=1, label="1M", step="month", stepmode="backward"), dict(count=3, label="3M", step="month", stepmode="backward"),
                                                 dict(step="all", label="All")], bgcolor=T["solid"], activecolor=T["acc"], font=dict(color=T["text"])))
    fig = style_fig(fig, 470)
    solid = "#000000" if not light else "#ffffff"
    fig.update_layout(paper_bgcolor=solid, plot_bgcolor=solid)       # solid panel: the drifting background candles must not show through
    return fig


# ------------------------------------------------------------------ result views
def horizon_cards(r):
    cards = []
    for h in (5, 10, 20, 40):
        row = row_at(r, h)
        sub = f"Charts only: ▲ {pct(row['CV only P(UP)'])}"
        if row["text only P(UP)"] is not None and not (isinstance(row["text only P(UP)"], float) and np.isnan(row["text only P(UP)"])):
            sub += f" &nbsp;·&nbsp; Filing text only: ▲ {pct(row['text only P(UP)'])}"
        cards.append(
            f'<div class="nx-card"><div class="nx-kicker">{h} sessions · {HORIZON_TXT[h]}</div>'
            f'<div style="margin:8px 0 10px">{chip(row["call"])} <span class="nx-muted" style="font-size:.8rem">sideways band {esc(str(row["sideways band"]))}</span></div>'
            f'{pbar(float(row["P(DOWN)"]), float(row["P(SIDEWAYS)"]), float(row["P(UP)"]))}'
            f'<div class="nx-muted" style="font-size:.78rem;margin-top:8px">{sub}</div>'
            f'{evidence_badge(f"dir_{h}", "edge on the locked test", "no proven edge on the locked test")}</div>')
    return '<div class="nx-grid4">' + "".join(cards) + "</div>"


def verdict_card(r):
    row = row_at(r, 20)
    sym, tone, label = CALL_META.get(row["call"], CALL_META["NO CALL"])
    sec = B.get("sector_of", {}).get(r["ticker"].replace("-", "."), "")
    meta = f"as of {esc(str(r['asof']))} · last close ${r['p0']:,.2f}" + (f" · {esc(sec)}" if sec else "")
    nm = company_name(r["ticker"])
    return (f'<div class="nx-card"><div class="nx-kicker">{esc(r["ticker"])} · outlook for the next 20 sessions</div>'
            + (f'<div class="nx-company">{esc(nm)}</div>' if nm else "") +
            f'<div class="nx-verdict" style="margin-top:10px"><div><div class="nx-big nx-{tone}">{sym} {label}</div><div class="nx-muted" style="margin-top:6px">{meta}</div></div>'
            f'<div>{pbar(float(row["P(DOWN)"]), float(row["P(SIDEWAYS)"]), float(row["P(UP)"]))}'
            f'<div class="nx-muted" style="font-size:.8rem;margin-top:6px">Probabilities of DOWN, SIDEWAYS and UP. SIDEWAYS means a move smaller than {B["k_sideways"]} x the stock\'s own volatility.</div></div></div></div>')


def risk_cards(r):
    row, sim = row_at(r, 20), r["sim"]
    realised = r["sigma"] * math.sqrt(252)
    q = sim["quantiles"]
    return ('<div class="nx-grid4">'
            + kpi("Expected volatility", pct(float(row["expected vol (ann.)"])), f"per year · last 60 sessions realised {realised:.0%}",
                  evidence_badge("vol_20", "level beat the benchmark on test", "level not proven better than simple numbers"))
            + kpi("Chance of a >10% fall", pct(float(row["P(drawdown > 10%)"])), f"model · simulation says {sim['p_dd']:.0%}",
                  evidence_badge("dd_20", "ranks risk on the locked test", "ranking not proven on test"))
            + kpi("Median price in 20 days", f"${q[2][-1]:,.2f}", f"{(q[2][-1] / r['p0'] - 1):+.1%} from the last close")
            + kpi("Likely range (90 %)", f"${q[0][-1]:,.0f} – ${q[4][-1]:,.0f}", "5th to 95th percentile of simulated prices")
            + "</div>")


def detail_table(r):
    t = r["table"]
    rows = ""
    for _, x in t.iterrows():
        rows += (f"<tr><td>{int(x['horizon (sessions)'])}</td><td>{chip(x['call'])}</td><td>{pct(x['P(DOWN)'])}</td><td>{pct(x['P(SIDEWAYS)'])}</td><td>{pct(x['P(UP)'])}</td>"
                 f"<td>{esc(str(x['sideways band']))}</td><td>{pct(x['expected vol (ann.)'])}</td><td>{pct(x['P(drawdown > 10%)'])}</td>"
                 f"<td>{pct(x['CV only P(UP)'])}</td><td>{pct(x['text only P(UP)'])}</td></tr>")
    head_ = "".join(f"<th>{c}</th>" for c in ["Sessions", "Call", "P(DOWN)", "P(SIDEWAYS)", "P(UP)", "Sideways band", "Exp. vol (ann.)", "P(fall > 10%)", "Charts only P(UP)", "Text only P(UP)"])
    return f'<div class="nx-card nx-tablewrap"><table class="nx-table"><thead><tr>{head_}</tr></thead><tbody>{rows}</tbody></table></div>'


def patterns_card(pats, n_bars):
    recent = [p for p in pats if p["i"] >= n_bars - 15][::-1]
    if not recent:
        body = '<div class="nx-muted">No classic candlestick pattern in the last 15 sessions.</div>'
    else:
        rows = "".join(f'<tr><td>{p["date"].date()}</td><td><b>{esc(p["name"])}</b></td>'
                       f'<td><span class="nx-chip" style="color:{T["up"] if p["bias"] == "bullish" else (T["down"] if p["bias"] == "bearish" else T["mid"])};'
                       f'background:{rgba(T["up"] if p["bias"] == "bullish" else (T["down"] if p["bias"] == "bearish" else T["mid"]), 0.14)}">{p["bias"]}</span></td>'
                       f'<td class="nx-muted">{esc(PATTERN_INFO[p["name"]][1])}</td></tr>' for p in recent[:8])
        body = f'<table class="nx-table"><thead><tr><th>Date</th><th>Pattern</th><th>Bias</th><th>What it means</th></tr></thead><tbody>{rows}</tbody></table>'
    return (f'<div class="nx-card nx-tablewrap"><div class="nx-kicker" style="margin-bottom:6px">Candlestick patterns, last 15 sessions (descriptive)</div>{body}'
            '<div class="nx-note" style="margin-top:10px">Detected with standard textbook rules on the daily candles. The AI models do not use these patterns, and this study did not '
            'test whether they predict anything, so read them as a description of recent trading, not as a signal.</div></div>')


def sources_card(r):
    rows = "".join(f"<tr><td><b>{esc(x['timeframe'])}</b></td><td>{esc(x['source'])}</td><td>{esc(x['candles'])}</td><td>{esc(x['scale'])}</td>"
                   f"<td>{esc(x['check'])}</td><td>{esc(x['volume'])}</td></tr>" for x in r["sources"])
    extra = "".join(f"<li>{esc(n_)}</li>" for n_ in r.get("notes", []))
    return (f'<div class="nx-card nx-tablewrap"><div class="nx-kicker" style="margin-bottom:6px">Data read from your screenshots · as of {esc(r["asof"])} '
            f'({esc(r.get("date_how", ""))})</div>'
            '<table class="nx-table"><thead><tr><th>Chart</th><th>Source</th><th>Candles</th><th>Price scale</th><th>Check against the legend</th><th>Volume</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>' + (f'<ul class="nx-muted" style="font-size:.82rem;margin:8px 0 0">{extra}</ul>' if extra else "")
            + '<div class="nx-note" style="margin-top:10px">Prices are measured from the pixels: the open, high, low and close of every candle are read against the '
            'price axis, and the last candle is checked against the O/H/L/C legend. Tested on 66 real TradingView screenshots against the true prices: median error 0.08% on the candles the models use, and model probabilities within about 1 to 5 percentage points of those from exact data. Very dense charts (several hundred candles) are read less precisely, so zoom in until roughly 100 to 300 candles are visible.</div></div>')


def flow_section(r):
    """Interactive flowchart: which neural network does what for the stock on screen."""
    if "_cv_each" not in r:
        r["_cv_each"] = FV.cv_each(B, r["img"])
    st.caption("Pick what you want to follow. The chart below shows the path from your input to that number, which networks were used, "
               "and what each of them said for this stock.")
    c1, c2 = st.columns([3, 2])
    with c1:
        kind_lbl = st.segmented_control("Follow this result", ["Direction", "Volatility", "Drawdown"], default="Direction", key=f"fl_k_{r['slot']}",
                                        selection_mode="single") or "Direction"
    with c2:
        h = st.segmented_control("Horizon (sessions)", [5, 10, 20, 40], default=20, key=f"fl_h_{r['slot']}", selection_mode="single") or 20
    kind = {"Direction": "dir", "Volatility": "vol", "Drawdown": "dd"}[kind_lbl]
    H(FV.flow_html(B, r, r["_cv_each"], kind, int(h)))
    st.markdown("**Which network is used for which target (all 12)**")
    H(FV.matrix_html(B, f"{kind}_{int(h)}"))


def render_stock(r):
    D = safe_history(r)
    pats = detect_patterns(D)
    n_bars = 0 if D is None else len(D)
    r["_summary"] = summary_points(r, pats, n_bars)
    r["_patterns"] = [{"date": str(p["date"].date()), "name": p["name"], "bias": p["bias"]} for p in pats if p["i"] >= n_bars - 15]
    H(verdict_card(r))
    H('<div style="height:12px"></div>' + summary_html(r["_summary"]))
    note = (f"Data: {esc(r.get('data_how', 'Yahoo Finance'))}. Ticker {esc(r['how'])}. "
            f"Text used: {'yes' if r['used_text'] else 'no (charts only)'}.")
    if not r["in_universe"]:
        note += " This stock is outside the 50 training stocks, so treat the output as an out-of-sample extrapolation."
    H(f'<div class="nx-note">{note}</div>')
    if r.get("sources"):
        H(sources_card(r))
    with st.expander("How was this result made? Which neural network did what", expanded=True, icon=":material/account_tree:"):
        flow_section(r)
    H('<div class="nx-h">All four horizons</div>')
    H(horizon_cards(r))
    H('<div class="nx-h">Risk and price range (20 sessions)</div>')
    H(risk_cards(r))
    c_t, c_b = st.columns([4, 1])
    with c_t:
        show_p = st.toggle("Mark candlestick patterns on the chart", value=True, key=f"pat_{r['slot']}")
    with c_b:
        if st.button("↺ Reset view", key=f"rv_{r['slot']}", width="stretch", help="Return the chart to its original zoom and position"):
            st.session_state["nx_rv"] = st.session_state.get("nx_rv", 0) + 1
    st.plotly_chart(outlook_chart(r, D, pats, show_p), width="stretch", config=PLOT_CFG, theme=None,
                    key=f"chart_{r['slot']}_{st.session_state.get('nx_rv', 0)}")
    st.caption("Drag to pan, scroll to zoom, or press Reset view. Hover a ▲ ▼ ● marker to see the pattern. The cone is a Monte Carlo simulation matched to the models' "
               "direction and volatility; it is a range of what could happen, not a prediction of one price.")
    H(patterns_card(pats, n_bars))
    with st.expander("What the models saw"):
        st.caption("With screenshots, the candles, prices and volume are read from your pictures, then redrawn in the exact training format below "
                   "(the models never see the broker's colours, fonts or indicators). Without screenshots the bars come from Yahoo Finance.")
        cA, cB = st.columns(2)
        with cA:
            st.markdown("**Your screenshots**")
            cc = st.columns(3)
            for c_, k in zip(cc, ["1W", "1D", "1H"]):
                if r["shots"].get(k) is not None:
                    c_.image(r["shots"][k], caption=k, width="stretch")
        with cB:
            st.markdown("**Redrawn for the models**")
            cc = st.columns(3)
            for c_, j, k in zip(cc, range(3), ["weekly · 52 candles", "daily · 63 candles", "hourly · 70 candles"]):
                c_.image(PALETTE_DARK[r["img"][j]], caption=k, width="stretch")
    with st.expander("All numbers"):
        H(detail_table(r))


def loader_html(label, k, n, with_text, lowmem_on, from_shots=False):
    """Pure-CSS loading scene: candles build themselves, an AI scan beam sweeps across them, and a pipeline readout cycles through
    the real steps of the analysis. It runs in the browser, so it keeps moving while Python works."""
    rnd = random.Random(sum(map(ord, str(label))) + k)
    lvl, candles = 40.0, []
    for i in range(18):
        o = lvl
        lvl = min(max(lvl + rnd.gauss(0.6, 9.0), 8.0), 78.0)
        lo_, hi_ = min(o, lvl), max(o, lvl)
        candles.append(f'<i class="{"up" if lvl >= o else "dn"}" style="height:{max(hi_ - lo_, 5):.0f}px;margin-bottom:{lo_:.0f}px;animation-delay:{i * 0.13:.2f}s"></i>')
    stages = (["Reading the price scale of each screenshot", "Finding every candle and its open, high, low and close", "Reading the volume bars and dating the candles"]
              if from_shots else ["Fetching daily and hourly price bars"])
    stages += ["Redrawing the 1W, 1D and 1H charts in the training format", "7 vision networks reading the charts"]
    if with_text:
        stages.append("6 language models reading the filing" + (" (one at a time, low-memory mode)" if lowmem_on else ""))
    stages += ["Blending and calibrating the forecasts", f"Simulating {int(n_paths):,} price paths for the next 20 sessions"]
    m = len(stages)
    p = 100.0 / m
    acc_glow, acc_beam = rgba(T["acc"], 0.55), rgba(T["acc"], 0.35)
    css = ("<style>"
           ".nx-load{position:relative;overflow:hidden;padding:20px 22px}"
           ".nx-load-top{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}"
           ".nx-ai{display:flex;align-items:center;gap:12px;font-weight:800;font-size:1.05rem;color:var(--text)}"
           f".nx-orb{{width:14px;height:14px;border-radius:50%;background:var(--acc);animation:nxPulse 1.4s ease-out infinite}}"
           f"@keyframes nxPulse{{0%{{box-shadow:0 0 0 0 {acc_glow}}}100%{{box-shadow:0 0 0 16px rgba(0,0,0,0)}}}}"
           ".nx-lc{position:relative;height:130px;margin:16px 0 12px;display:flex;align-items:flex-end;gap:9px;padding:0 8px;border-bottom:1px dashed var(--line);overflow:hidden}"
           ".nx-lc i{position:relative;flex:1;max-width:18px;border-radius:3px;transform-origin:bottom;opacity:0;animation:nxBuild 3.6s ease-out infinite}"
           ".nx-lc i::before{content:'';position:absolute;left:50%;width:2px;margin-left:-1px;top:-9px;bottom:-9px;background:inherit;border-radius:2px;opacity:.85}"
           ".nx-lc i.up{background:var(--up)}.nx-lc i.dn{background:var(--down)}"
           "@keyframes nxBuild{0%{opacity:0;transform:scaleY(.1)}12%{opacity:1;transform:scaleY(1)}82%{opacity:1;transform:scaleY(1)}100%{opacity:0;transform:scaleY(1)}}"
           f".nx-beam{{position:absolute;top:0;bottom:0;width:80px;left:-90px;background:linear-gradient(90deg,transparent,{acc_beam},transparent);animation:nxBeam 2.4s ease-in-out infinite;pointer-events:none}}"
           "@keyframes nxBeam{0%{left:-90px}100%{left:100%}}"
           ".nx-stages{position:relative;height:26px;font-weight:700;color:var(--text)}"
           f".nx-stages span{{position:absolute;left:0;top:0;opacity:0;animation:nxStage{m} {2 * m}s infinite}}"
           f"@keyframes nxStage{m}{{0%{{opacity:0;transform:translateY(6px)}}{p * 0.12:.2f}%{{opacity:1;transform:none}}{p * 0.85:.2f}%{{opacity:1;transform:none}}{p:.2f}%{{opacity:0}}100%{{opacity:0}}}}"
           ".nx-indet{height:4px;border-radius:99px;background:var(--panel2);overflow:hidden;position:relative;margin-top:12px}"
           ".nx-indet::after{content:'';position:absolute;top:0;bottom:0;width:35%;left:-35%;background:linear-gradient(90deg,var(--acc),var(--acc2));border-radius:99px;animation:nxIndet 1.6s ease-in-out infinite}"
           "@keyframes nxIndet{0%{left:-35%}100%{left:100%}}"
           "@property --nxt{syntax:'\\3c integer\\3e';initial-value:0;inherits:false}"   # escaped '<integer>': a raw '<' makes DOMPurify drop the style
           ".nx-time{animation:nxTime 3600s linear forwards;counter-reset:nxt var(--nxt);font-variant-numeric:tabular-nums;color:var(--muted);font-weight:700}"
           ".nx-time::after{content:counter(nxt) ' s elapsed'}"
           "@keyframes nxTime{from{--nxt:0}to{--nxt:3600}}"
           "</style>")
    spans = "".join(f'<span style="animation-delay:{2 * i}s">{esc(s_)}</span>' for i, s_ in enumerate(stages))
    hint = "The first analysis loads the models (about 1 to 2 minutes on a laptop CPU)."
    if with_text and lowmem_on:
        hint += " With filing text, low-memory mode loads each language model in turn, so it takes longer."
    return (css + f'<div class="nx-card nx-load"><div class="nx-load-top"><div class="nx-ai"><div class="nx-orb"></div>'
            f'AI analysing {esc(str(label).upper())} <span class="nx-muted" style="font-weight:600">· stock {k} of {n}</span></div><div class="nx-time"></div></div>'
            f'<div class="nx-lc">{"".join(candles)}<div class="nx-beam"></div></div>'
            f'<div class="nx-stages">{spans}</div><div class="nx-indet"></div>'
            f'<div class="nx-muted" style="font-size:.8rem;margin-top:10px">{hint}</div></div>')


def compare_view(ok):
    H('<div class="nx-h">Side by side</div>')
    cards = []
    for r in ok:
        row = row_at(r, 20)
        sym, tone, _ = CALL_META.get(row["call"], CALL_META["NO CALL"])
        q = r["sim"]["quantiles"]
        nm_ = company_name(r["ticker"])
        cards.append(f'<div class="nx-card"><div class="nx-kicker">{esc(r["ticker"])} · {esc(str(r["asof"]))}</div>'
                     + (f'<div class="nx-company" style="font-size:.9rem">{esc(nm_)}</div>' if nm_ else "") +
                     f'<div class="nx-big nx-{tone}" style="font-size:2rem;margin:6px 0">{sym} {esc(str(row["call"]))}</div>'
                     f'{pbar(float(row["P(DOWN)"]), float(row["P(SIDEWAYS)"]), float(row["P(UP)"]))}'
                     f'<div class="nx-muted" style="font-size:.82rem;margin-top:8px">vol {pct(float(row["expected vol (ann.)"]))} · P(fall>10%) {pct(float(row["P(drawdown > 10%)"]))}<br>'
                     f'90 % range ${q[0][-1]:,.0f} – ${q[4][-1]:,.0f}</div></div>')
    H('<div class="nx-grid4">' + "".join(cards) + "</div>")
    xs = [float(row_at(r, 20)["expected vol (ann.)"]) for r in ok]
    ys = [float(row_at(r, 20)["P(drawdown > 10%)"]) for r in ok]
    lean = [float(row_at(r, 20)["P(UP)"]) - float(row_at(r, 20)["P(DOWN)"]) for r in ok]
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="markers+text", text=[r["ticker"] for r in ok], textposition="top center",
                               marker=dict(size=34, color=lean, colorscale=[[0, T["down"]], [0.5, T["mid"]], [1, T["up"]]], cmin=-0.5, cmax=0.5, line=dict(color=T["muted"], width=1),
                               colorbar=dict(title="P(UP) − P(DOWN)", thickness=10)), hovertemplate="%{text}<br>expected vol %{x:.0%}<br>P(fall>10%) %{y:.0%}<extra></extra>"))
    fig.update_layout(title=dict(text="Risk map: further right = more volatile, higher = more drawdown risk; colour = direction lean (20 sessions)", font=dict(size=13)),
                      xaxis_title="expected volatility (annualised)", yaxis_title="P(fall > 10%)", xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    st.plotly_chart(style_fig(fig, 380), width="stretch", config=PLOT_CFG, theme=None, key="riskmap")


# ------------------------------------------------------------------ pages
# ------------------------------------------------------------------ Ask AI: Google Gemini (free key) answers questions about the study and the current analysis
CHAT_RULES = """You are the NEXUS Alpha assistant inside a university research dashboard (Master of Applied AI final project).
You help the user understand (1) the stocks they analysed, (2) how the system works and (3) what the study found. Give SOLID, useful answers.
How to answer:
- Start with the direct answer to the question. Explain the concept or the result first, in plain language, with the numbers from the facts below.
- You MAY use standard textbook knowledge to explain concepts (what a candlestick pattern is and how traders read it, what R-squared, AUC, macro-F1, volatility or drawdown mean, what a Monte Carlo simulation is, what a transformer is). Mark such parts as general background, for example "In general, ..." or "Textbook reading: ...", and keep them separate from what THIS study measured.
- For study results and the user's analysed stock, use ONLY the numbers in the facts below. Never invent a statistic, a result or a test the study did not run.
- Caveats go at the END in one short sentence at most (for example: "The dashboard shows these patterns for context; the models do not use them."). NEVER open with, or build the answer around, "this was not tested" or "I do not know". If one specific detail is not in the facts, answer everything else fully and name that one detail briefly.
- Be honest about weak evidence when it matters: the direction edge is small; simple numbers beat charts and text on volatility and drawdown. Say it once, factually, not repeatedly.
- Use short paragraphs or bullets. Give numbers with units and say what they mean for this stock.
- Write all maths as plain text (R-squared, AUC, macro-F1, x, sqrt). Never use LaTeX or special maths symbols.
- Do not tell the user to buy, sell or hold anything and do not give personalised investment advice; explaining what the numbers mean is fine. This is educational research, not financial advice.
"""

STUDY_FACTS = """STUDY FACTS (final A3b reports; locked test 2024-01 to 2026-08, opened once after a hashed freeze):
- Data: 50 large US stocks in 5 sectors (IT, Financials, Health Care, Consumer Discretionary, Communication Services); decisions 2016-10 to 2026-08; train 2016-2020, early stopping 2021, model selection 2022-2023, test 2024-2026.
- Targets at 5, 10, 20 and 40 trading sessions: direction (DOWN / SIDEWAYS / UP, sideways band = 0.5 x the stock's own daily volatility x sqrt(horizon)), realised volatility, and a fall of more than 10% (drawdown).
- Inputs: weekly, daily and hourly candlestick chart images (7 vision networks: custom CNN, ResNet-50, EfficientNet-B0, ConvNeXt-Tiny, ViT-B/16, Swin-T, YOLOv8s-cls) and SEC filing text (TF-IDF plus FinBERT, SEC-BERT, RoBERTa-large, Longformer, DeBERTa-v3-large, FinGPT). The dashboard averages the top-3 vision models per task and blends the top-3 deployable text models when a filing is fresh.
- Vision results: a small, statistically supported direction signal at 5 and 10 sessions (about +0.03 macro-F1 over a shuffled null). Judged fairly against simple numbers-only baselines, charts were better at 1 of 4 horizons (5 sessions). Plain accuracy is about the same as always predicting SIDEWAYS. For volatility and drawdown, simple numbers (trailing volatility, momentum, technical indicators) beat the chart models in all 8 tasks; one likely reason is that each chart is scaled to its own high and low, which hides the absolute volatility level.
- Text results: language models showed no direction or volatility skill; their drawdown signal vanished against no-text baselines on the same filings. Most of what the text models pick up is which company filed and when (shortcut learning). For volatility, filing-date (calendar) columns add about +0.063 R-squared to a technical model; the words add a further +0.013.
- Event study: filing days move prices about 2.1 times as much as matched non-filing days; by reaction size 10-Q > 10-K > 8-K; 8-K items 2.02 (results of operations) and 2.05 (exit costs) have the largest reactions. The words of a filing explain part of how BIG the reaction is, even beyond form, item, sector, timing and company (+0.043 R-squared), but not its direction.
- Combining networks: combinations of vision networks beat the best single one in 5 of 12 tasks (mostly volatility); combinations of language models in 0 of 12; adding language models to vision combinations helped in 2 of 12 (drawdown).
- Drawdown probabilities rank risk (ROC-AUC about 0.54 to 0.63) but their level is no better than a constant guess at the training base rate (Brier skill <= 0): read them as relative risk.
- Monte Carlo price ranges: with a volatility scale of 1.25 chosen on validation, the nominal 80% / 90% ranges contained the realised 20-day return 83% / 91% of the time on the test.
- Candlestick patterns (doji, hammer, engulfing, stars, soldiers/crows) are shown in the dashboard only as a description of recent trading; the study did not test them.
- Limitations: 50 large surviving US stocks; one decade with a regime change between validation (2022 bear market) and test (strong 2024-2026); one training seed per model; no trading costs; everything after the test was opened is labelled post-hoc.
"""


PROJECT_FACTS = """HOW THE SYSTEM WORKS (architecture)
- Data: daily, weekly and hourly price bars (2016-10 to 2026-10) for 50 stocks, 10 per sector; 554 10-K, 1,700 10-Q and 8,015 8-K filings from EDGAR (8-Ks are about 79% of the training events).
- Rows: train 2016-10 to 2020 = 51,250; early stopping 2021 = 2,150; model selection 2022-2023 = 4,600; locked test 2024-01 to 2026-08 = 6,550 (64,550 decision rows, 193,650 chart images, 224 x 224 pixels). A 40-session gap separates the blocks so labels cannot overlap.
- Charts: each decision day has a weekly (52 candles), daily (63) and hourly (70) chart with a volume strip, scaled to its own high and low. Design consequence: the absolute volatility level is hidden from the charts.
- Models: 7 vision networks (custom JKX-style CNN trained from scratch, ResNet-50, EfficientNet-B0, ConvNeXt-Tiny, ViT-B/16, Swin-T, YOLOv8s-cls) and 6 language models in the dashboard (TF-IDF + logistic regression, FinBERT, SEC-BERT, RoBERTa-large, Longformer, DeBERTa-v3-large; FinGPT frozen 4-bit was analysed in the research only). Every network outputs 12 numbers: direction, volatility and drawdown at 5, 10, 20 and 40 sessions.
- Deployment recipe: for each of the 12 targets the 3 best chart networks and the 3 best text models on 2022-2023 data are averaged; if a filing text is given the two averages are blended with a weight per target; a final calibration makes stated probabilities honest. The same recipe is used for every stock (it is not chosen by sector).
- Price cone: 4,000 simulated 20-session paths with normally distributed daily shocks; drift matches the 20-session direction probabilities, spread uses the 20-session volatility times a width factor of 1.25 chosen on validation. On the locked test the nominal 80% and 90% ranges contained 83% and 91% of realised 20-day returns.
- Screenshot reading: with screenshots, the candles, prices and volume come from the pictures (OCR of the price axis plus a pixel reader, checked against the O/H/L/C legend; tested on 66 real TradingView screenshots with a median price error of 0.08%); without screenshots the bars come from Yahoo Finance. The models always see the three charts redrawn in the training format.
- Dashboard pages: Analyse (result, the flowchart showing which network did what, price chart with patterns), Ask AI (this chat), Architecture (pipeline and what it predicts), Models (which networks serve which target), Guide.

BENCHMARKS AND RESULTS (locked test 2024-2026)
- Benchmarks: uninformed (shuffled-label null, R-squared 0, AUC 0.5); numbers-only models without images or text: baseline_simple (direction: logistic on 20/60/120-day momentum; volatility: HAR regression; drawdown: logistic on trailing volatility, distance from the 52-week high and 20-day return) and tech_gbm (LightGBM on 20 technical indicators); on filing events also company_event (per-company base rates), simple_event and tech_event (same numbers-only models plus the form); ML-style baselines: the scratch CNN, TF-IDF + logistic regression, frozen vs fine-tuned networks, random-weight and shuffled-label controls.
- Computer vision: beat the uninformed benchmark in 6 of 12 tests (direction at 5 and 10 sessions, volatility at 5, drawdown at 5, 10 and 40). Against the strong numbers-only baselines the chart networks win on direction at all four horizons (ConvNeXt-T at 5, 10, 20; EfficientNet-B0 at 40) but lose on all volatility and drawdown tests. With each model's own shuffled null only the 5-session direction edge is clearly better.
- NLP: beat the uninformed benchmark in 4 of 12 tests (all drawdown, Longformer). Against no-text baselines on the same filings it wins 1 of 12 (SEC-BERT, direction at 20 sessions, about +0.09 macro-F1 over company base rates), loses the 4 volatility tests, and is not clearly different on drawdown. TF-IDF matches the large transformers almost everywhere, and most of the text signal is company and calendar shortcuts.
- CV + NLP combined: the late-fusion ensemble beats numbers-only baselines on direction at all four horizons, loses on volatility and drawdown in 7 of 8 tests (one tie). A LightGBM that adds CV and NLP outputs to the technical indicators beats the technical indicators alone in 6 of 12 tests, mostly because of filing-calendar columns, not the words.
- Pre-trained chart networks (ConvNeXt-T, ViT-B/16, Swin-T) reach direction macro-F1 about 0.35-0.38; the scratch CNN reaches 0.22-0.28, about the same as the shuffled-label controls.
- Dashboard recipe versus the no-skill benchmark: supported for all 4 direction tasks (about +0.023 to +0.036 macro-F1), all 4 drawdown tasks (AUC about +0.04 to +0.13 above 0.5) and only 1 of 4 volatility tasks (5 sessions). Volatility R-squared at 40 sessions is negative (-0.10).
- Negative R-squared means a constant guess of the average would have been better. Causes: each chart is scaled to its own high and low (hides the volatility level), vertical-scale augmentation hurt volatility, the test period (strong 2024-2026) differs from validation, and IT-sector volatility is far above the pooled average.
- By sector (descriptive, no confidence intervals): CV direction macro-F1 is similar across sectors (0.34-0.37; Financials best, IT lowest) and ConvNeXt-T is the best direction network in every sector; volatility R-squared is positive only for Financials (+0.07) and very negative for IT (-0.44); drawdown AUC is best for IT (0.62) and about chance for Communication Services (0.49); NLP is strongest in Financials (Financials files about twice as many 8-Ks). Which network is best for volatility and drawdown depends on the sector (hypothesis H6, supported in 2 of 3).
- Which networks serve which target (examples): direction at 5, 10 and 20 sessions uses ConvNeXt-T, ViT-B/16 and Swin-T; direction at 40 uses EfficientNet-B0, ResNet-50 and ConvNeXt-T; short volatility uses Swin-T, ViT-B/16 and ResNet-50; drawdown uses combinations of Swin-T, ResNet-50, EfficientNet-B0 and ViT-B/16.
- Text share of the final blend when a fresh filing text is given: direction 0% at 5, 10 and 40 sessions and 25% at 20; volatility 25% at 5, 10 and 20 and 75% at 40; drawdown 100%. Without a filing text the answer comes from the charts only.

CANDLESTICK PATTERNS (shown on the price chart and in the patterns card)
- The dashboard detects 11 classic patterns on the daily candles with textbook rules: Doji, Hammer, Hanging man, Inverted hammer, Shooting star, Bullish engulfing, Bearish engulfing, Morning star, Evening star, Three white soldiers, Three black crows. Markers: green up-triangle = bullish, red down-triangle = bearish, amber circle = neutral (doji).
- Meanings: Doji = open and close almost equal, buyers and sellers ended the day in balance (indecision). Hammer = long lower shadow after a fall, sellers pushed down but buyers took it back (possible reversal up). Hanging man = hammer shape after a rise (possible weakening of an uptrend). Inverted hammer = long upper shadow after a fall (buyers tried to push up). Shooting star = long upper shadow after a rise (sellers drove the price back down). Bullish engulfing = a rising candle whose body covers the previous falling body. Bearish engulfing = the reverse. Morning star = long fall, small pause, strong rise (three candles). Evening star = long rise, small pause, strong fall. Three white soldiers = three long rising candles each closing higher. Three black crows = three long falling candles each closing lower.
- Role in the system: patterns are a human-readable description of recent trading, computed from the same candles; the 13 networks do not use them as inputs, and the study did not run a hypothesis test on them.
"""


def study_context():
    lines = [STUDY_FACTS, PROJECT_FACTS, "DASHBOARD RECIPE SKILL ON THE LOCKED TEST (skill, 95% interval, supported after Holm correction):"]
    for t, s in SK.items():
        lines.append(f"- {t}: {s['skill']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}] supported={bool(s.get('supported'))}")
    lab_c = {m: v["label"] for m, v in B["cv_models"].items()}
    lab_n = {m: v["label"] for m, v in B["nlp_models"].items()}
    lines.append("EXACT NETWORKS USED PER TARGET IN THIS DASHBOARD (top 3 on 2022-2023 validation, best first) AND TEXT SHARE OF THE BLEND WHEN A FILING TEXT IS GIVEN:")
    for t in B["tasks"]:
        k_, h_ = t.split("_")
        lines.append(f"- {({'dir': 'direction', 'vol': 'volatility', 'dd': 'drawdown'})[k_]} {h_} sessions: chart networks {', '.join(lab_c.get(m, m) for m in B['cv_members'][t])}; "
                     f"text models {', '.join(lab_n.get(m, m) for m in B['nlp_members'].get(t, []))}; text share {1 - float(B['w_fresh'].get(t, 1.0)):.0%}")
    lines.append("ANSWERS TO THE TEN PROJECT QUESTIONS (as written by the notebook):")
    lines += [f"- {k}: {v}" for k, v in B.get("answers", {}).items()]
    return "\n".join(lines)


def analysis_context(results):
    ok_ = [r for r in results if "error" not in r]
    if not ok_:
        return "CURRENT ANALYSIS: no stock has been analysed yet in this session."
    out = ["CURRENT ANALYSIS (what the user sees on the Analyse page):"]
    for r in ok_:
        out.append(f"\n## {r['ticker']} as of {r['asof']}, last close ${r['p0']:,.2f}; sector {B.get('sector_of', {}).get(r['ticker'].replace('-', '.'), 'outside the 50 training stocks')}; "
                   f"data: {r.get('data_how', 'Yahoo Finance')}; ticker {r['how']}; filing text used: {'yes (' + r.get('form', '') + ')' if r['used_text'] else 'no'}")
        for _, x in r["table"].iterrows():
            tx = x["text only P(UP)"]
            tx_s = "n/a" if tx is None or (isinstance(tx, float) and np.isnan(tx)) else f"{tx:.0%}"
            out.append(f"- {int(x['horizon (sessions)'])} sessions: call {x['call']}; P(UP) {x['P(UP)']:.0%}, P(SIDEWAYS) {x['P(SIDEWAYS)']:.0%}, P(DOWN) {x['P(DOWN)']:.0%}; "
                       f"sideways band {x['sideways band']}; expected volatility {x['expected vol (ann.)']:.0%} a year; P(fall > 10%) {x['P(drawdown > 10%)']:.0%}; "
                       f"charts-only P(UP) {x['CV only P(UP)']:.0%}; text-only P(UP) {tx_s}")
        q = r["sim"]["quantiles"]
        out.append(f"- Monte Carlo, 20 sessions: 5th/25th/50th/75th/95th percentile prices ${q[0][-1]:,.2f} / ${q[1][-1]:,.2f} / ${q[2][-1]:,.2f} / ${q[3][-1]:,.2f} / ${q[4][-1]:,.2f}; "
                   f"simulated P(fall > 10%) {r['sim']['p_dd']:.0%}; realised volatility of the last 60 sessions {r['sigma'] * math.sqrt(252):.0%} a year")
        if r.get("_summary"):
            out.append("- Summary shown to the user: " + " | ".join(f"{hd}: {tx}" for _, hd, tx in r["_summary"]))
        pats_ = r.get("_patterns") or []
        out.append("- Candlestick patterns detected in the last 15 sessions: " + (", ".join(f"{p['name']} ({p['bias']}) on {p['date']}" for p in pats_) or "none"))
        for nm_ in sorted({p["name"] for p in pats_}):
            out.append(f"  * {nm_}: {PATTERN_INFO[nm_][1]}")
        if r.get("text"):
            out.append(f"- Filing text the user pasted (first 1,500 characters): {r['text'][:1500]}")
    return "\n".join(out)


GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_PREFERRED = ["gemini-3.5-flash", "gemini-3-flash-preview", "gemini-3.1-flash-lite", "gemini-flash-lite-latest", "gemini-flash-latest"]
GEMINI_FALLBACK = GEMINI_PREFERRED[:4]


@st.cache_data(show_spinner=False, ttl=3600)
def gemini_models(api_key):
    """Text models this key may use, flash variants first (the free tier covers them). Falls back to a fixed list."""
    import requests
    try:
        resp = requests.get(f"{GEMINI_BASE}/models", headers={"x-goog-api-key": api_key}, params={"pageSize": 200}, timeout=15)
        resp.raise_for_status()
        names = []
        for m in resp.json().get("models", []):
            nm = m.get("name", "").split("/")[-1]
            if ("generateContent" in m.get("supportedGenerationMethods", []) and nm.startswith("gemini")
                    and not any(w in nm for w in ("tts", "image", "embedding", "live", "audio", "robotics", "computer", "exp"))):
                names.append(nm)
        def rank(n):                              # preferred current models first, then other flash models, retired 1.x / 2.x family last
            if n in GEMINI_PREFERRED:
                return (0, GEMINI_PREFERRED.index(n), n)
            retired = n.startswith(("gemini-1", "gemini-2"))
            return (2 if retired else 1, ("flash" not in n), ("lite" in n), n)
        names = sorted(set(names), key=rank)
        return names or GEMINI_FALLBACK
    except Exception:
        return GEMINI_FALLBACK


def _gemini_stream(api_key, model, system_text, msgs):
    import requests
    import json as _json
    body = {"systemInstruction": {"parts": [{"text": system_text}]},
            "contents": [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]} for m in msgs],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 8192}}
    with requests.post(f"{GEMINI_BASE}/models/{model}:streamGenerateContent", params={"alt": "sse"}, headers={"x-goog-api-key": api_key},
                       json=body, stream=True, timeout=(10, 120)) as resp:
        if resp.status_code != 200:
            try:
                detail = resp.json().get("error", {}).get("message", "")
            except Exception:
                detail = ""
            raise RuntimeError(f"{resp.status_code}|{detail}")
        resp.encoding = "utf-8"
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            try:
                chunk = _json.loads(line[5:].strip())
            except ValueError:
                continue
            fb = chunk.get("promptFeedback", {})
            if fb.get("blockReason"):
                raise RuntimeError(f"blocked|{fb['blockReason']}")
            for cand in chunk.get("candidates", []):
                for part in cand.get("content", {}).get("parts", []):
                    if part.get("text"):
                        yield part["text"]


def _stream_with_fallback(api_key, models, system_text, msgs):
    """Tries the chosen model, then the others, as long as nothing has been shown yet (retired model, busy model, rate limit)."""
    last = None
    for i, m in enumerate(models):
        gen = _gemini_stream(api_key, m, system_text, msgs)
        try:
            first = next(gen)
        except StopIteration:
            return
        except RuntimeError as e:
            code = str(e).partition("|")[0]
            last = e
            if code in ("404", "429", "500", "503") and i < len(models) - 1:
                continue
            raise
        st.session_state["nx_chat_used"] = m
        yield first
        yield from gen
        return
    if last:
        raise last


def ask_gemini(api_key, model, history_msgs, results):
    """Streams Gemini's answer into the page; returns (text, ok). Failed calls are not stored as answers."""
    import requests
    system_text = CHAT_RULES + "\n" + study_context() + "\n\n" + analysis_context(results)
    try:
        order = [model] + [m for m in gemini_models(api_key)[:6] if m != model]
        text = st.write_stream(_stream_with_fallback(api_key, order, system_text, history_msgs))
        text = text if isinstance(text, str) else "".join(map(str, text))
        if not text.strip():
            msg = "The model returned no text (it may have declined). Try rephrasing the question."
            st.warning(msg)
            return msg, False
        return text, True
    except RuntimeError as e:
        code, _, detail = str(e).partition("|")
        if code == "blocked":
            msg = f"Google's safety filter blocked this question ({detail}). Try rephrasing it."
        elif code in ("401", "403") or (code == "400" and ("API key" in detail or "API_KEY" in detail)):
            msg = "Google rejected the API key. Check that you copied the whole key from Google AI Studio."
        elif code == "429":
            msg = "Free-tier limit reached on every model tried (requests per minute or per day). Wait a minute and ask again."
        elif code == "404":
            msg = "None of the Gemini models tried is available for this key. Google retires models for new keys; pick another model in the list or create a new key in Google AI Studio."
        elif code == "503":
            msg = "Google's Gemini service is busy right now. Ask again in a moment."
        else:
            msg = f"Gemini error {code}. {detail[:200]}"
    except requests.exceptions.RequestException:
        msg = "Could not reach Google's Gemini API. Check the internet connection."
    st.error(msg)
    return msg, False


def resolve_api_key():
    """The Gemini key: .streamlit/secrets.toml first, then the GEMINI_API_KEY / GOOGLE_API_KEY environment variable."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        try:
            k = str(st.secrets.get(name, "")).strip()
        except Exception:
            k = ""
        if not k:
            k = os.environ.get(name, "").strip()
        if k:
            return k
    return ""


SUGGESTED = ["Explain my result in simple words", "How reliable is the direction call?", "What does the risk part mean for me?",
             "What did the filing text add?", "What do the candlestick patterns show?", "What did the study prove, and what not?"]


tab_a, tab_ai, tab_arch, tab_m, tab_g = st.tabs(["Analyse", "Ask AI", "Architecture", "Models", "Guide"])

with tab_a:
    H("""<div class="nx-steps">
    <div class="nx-step"><div class="n">1</div><div><div class="t">Pick a stock</div><div class="d">Type a ticker, or drop in chart screenshots.</div></div></div>
    <div class="nx-step"><div class="n">2</div><div><div class="t">Add a filing summary (optional)</div><div class="d">Paste a 10-K, 10-Q or 8-K summary for the text models.</div></div></div>
    <div class="nx-step"><div class="n">3</div><div><div class="t">Read the outlook</div><div class="d">Direction, volatility, drawdown risk and a price range.</div></div></div></div>""")
    inputs = []
    with st.container(key="inputpanel"):
        n = st.segmented_control("Stocks to analyse", [1, 2, 3, 4], default=1, key="nx_n", selection_mode="single") or 1
        for i, tab in enumerate(st.tabs([f"Stock {i + 1}" for i in range(n)])):
            with tab:
                c1, c2 = st.columns([1, 2])
                with c1:
                    tk = st.text_input("Ticker or company name", key=f"tk_{i}", placeholder="for example NVDA or Nvidia")
                    rt_, rn_ = resolve_ticker(tk)
                    if rt_ and tk.strip():
                        H(f'<div class="nx-hint">Will analyse <b>{esc(rt_)}</b>' + (f" · {esc(rn_)}" if rn_ else " · company name is looked up after you press Analyse") + "</div>")
                    pick = st.selectbox("…or pick one of the 50 trained stocks", ["—"] + sorted(B["universe"]), format_func=stock_label, key=f"pick_{i}",
                                        help="Type a ticker or a company name to search the list.")
                    form = st.selectbox("Type of the filing text", ["8-K", "10-Q", "10-K"], key=f"form_{i}")
                with c2:
                    txt = st.text_area("Recent filing summary (optional)", height=170, key=f"txt_{i}",
                                       placeholder="Paste a short summary of the latest 10-K, 10-Q or 8-K. Leave empty to use the charts only.")
                with st.expander("Optional: add chart screenshots (1W, 1D, 1H). Paste with Ctrl+V or upload"):
                    st.caption("Take a screenshot (Windows: Win + Shift + S), click a box below and press Ctrl+V. You can also drag an image onto a box or upload a file. "
                               "The candles, prices and volume are read straight from the pictures (no market download for those timeframes). "
                               "Add all three for a fully screenshot-based analysis; a missing one is filled from Yahoo Finance.")
                    ups = {}
                    for col, slot in zip(st.columns(3), ["1W", "1D", "1H"]):
                        with col:
                            gen = st.session_state.get(f"pgen_{i}_{slot}", 0)
                            pkey = f"paste_{i}_{slot}_{gen}"
                            prev = st.session_state.get(pkey)
                            try:
                                had = bool(prev and prev.get("image"))
                            except Exception:
                                had = False
                            pasted = paste_zone(slot, pkey, had)
                            up = st.session_state.get(f"up_{i}_{slot}")
                            chosen = up if up is not None else pasted
                            if chosen is not None:
                                st.image(chosen, width="stretch")
                                if up is not None and pasted is not None:
                                    st.caption("A pasted image and an uploaded file are both set. The uploaded file is used.")
                                if pasted is not None and st.button("Remove pasted image", key=f"rmp_{i}_{slot}", width="stretch"):
                                    st.session_state[f"pgen_{i}_{slot}"] = gen + 1
                                    st.rerun()
                            st.file_uploader(f"{slot} screenshot file", type=["png", "jpg", "jpeg"], key=f"up_{i}_{slot}", label_visibility="collapsed",
                                             help="Or choose a PNG / JPG file instead of pasting.")
                            ups[slot] = chosen
                inputs.append((ups, resolve_ticker(tk)[0] if tk.strip() else (pick if pick != "—" else ""), form, txt))
        go_btn = st.button("Analyse", type="primary", width="stretch", icon=":material/analytics:")
    if go_btn:
        res, valid = [], [(j, x) for j, x in enumerate(inputs) if x[1] or any(u is not None for u in x[0].values())]
        if not valid:
            st.warning("Type a ticker or upload a screenshot first.")
        else:
            loader = st.empty()
            for k, (j, (ups, tk, form, txt)) in enumerate(valid):
                has_shots = any(u is not None for u in ups.values())
                loader.html(loader_html(tk or "from screenshot", k + 1, len(valid), bool((txt or "").strip()), lowmem, from_shots=has_shots))
                shots = {s: (to_pil(u) if u is not None else None) for s, u in ups.items()}
                try:
                    if has_shots:          # prices, candles and volume come from the pictures
                        r = SE.analyse_screens(B, tk, shots, form, txt, n_paths=n_paths, seed=seed)
                    else:                  # ticker only: market data from Yahoo Finance
                        r = E.analyse(B, tk, shots, form, txt, n_paths=n_paths, seed=seed)
                        r["data_how"] = "prices from Yahoo Finance (no screenshots given)"
                    r["slot"] = j + 1
                    r["shots"] = shots
                    r["text"], r["form"] = (txt or "").strip(), form
                    res.append(r)
                except Exception as e:
                    res.append({"slot": j + 1, "error": f"{type(e).__name__}: {e}"})
            loader.empty()
            st.session_state["res"] = res
    results = st.session_state.get("res", [])
    for r in results:
        if "error" in r:
            st.error(f"Stock {r['slot']}: {r['error']}")
    ok = [r for r in results if "error" not in r]
    if ok:
        if len(ok) > 1:
            compare_view(ok)
            for tab, r in zip(st.tabs([(x["ticker"] + " · " + company_name(x["ticker"])[:24]) if company_name(x["ticker"]) else x["ticker"] for x in ok]), ok):
                with tab:
                    render_stock(r)
        else:
            render_stock(ok[0])
    else:
        H('<div class="nx-note">No analysis yet. Type a ticker above and press Analyse. The models were trained on 50 large US stocks (2016-2021), chosen on 2022-2023 and scored once on 2024-2026.</div>')

with tab_ai:
    H('<div class="nx-card"><div class="nx-kicker">Ask AI · powered by Google Gemini (free)</div><div class="nx-read" style="margin-top:6px">Ask anything about your analysis or the study. '
      'The assistant answers only from the study\'s results and the numbers on your Analyse page, and says so when it does not know.</div></div>')
    env_key = resolve_api_key()
    c_k, c_m = st.columns([2, 1])
    with c_k:
        if env_key:
            st.caption("Using the Gemini API key stored on this computer (.streamlit/secrets.toml or the GEMINI_API_KEY setting).")
            api_key = env_key
        else:
            api_key = st.text_input("Your free Google Gemini API key", type="password", key="nx_api_key", placeholder="AIza...",
                                    help="Get a free key at aistudio.google.com/apikey (no card needed). Kept only in this browser session's memory; never saved to disk. You can also set GEMINI_API_KEY before starting the app.").strip()
    with c_m:
        chat_model = st.selectbox("Model", gemini_models(api_key) if api_key else GEMINI_FALLBACK, key="nx_chat_model",
                                  help="Flash models are listed first; they are covered by Google's free tier.")
    results_now = st.session_state.get("res", [])
    n_ok = len([r for r in results_now if "error" not in r])
    st.caption((f"Context: the study's findings plus {n_ok} analysed stock(s) from the Analyse page." if n_ok else
                "Context: the study's findings only. Analyse a stock first to ask about it.")
               + " Your questions and these numbers are sent to Google's Gemini API. On the free tier Google may use them to improve its products, so do not paste anything private.")
    if "nx_chat" not in st.session_state:
        st.session_state["nx_chat"] = []

    def _queue(q_):
        st.session_state["nx_pending"] = q_
    cols_q = st.columns(3)
    for i_, q_ in enumerate(SUGGESTED):
        cols_q[i_ % 3].button(q_, key=f"sq_{i_}", on_click=_queue, args=(q_,), width="stretch")
    for m_ in st.session_state["nx_chat"]:
        with st.chat_message(m_["role"], avatar=":material/person:" if m_["role"] == "user" else ":material/candlestick_chart:"):
            st.markdown(m_["content"])
    typed = st.chat_input("Ask about your analysis or the study…", key="nx_chat_in")
    question = typed or st.session_state.pop("nx_pending", None)
    if question:
        if not api_key:
            st.warning("Paste your free Gemini API key above first (get one at aistudio.google.com/apikey).")
        else:
            st.session_state["nx_chat"].append({"role": "user", "content": question})
            with st.chat_message("user", avatar=":material/person:"):
                st.markdown(question)
            hist = st.session_state["nx_chat"][-12:]
            while hist and hist[0]["role"] != "user":          # the API needs the history to start with a user turn
                hist = hist[1:]
            with st.chat_message("assistant", avatar=":material/candlestick_chart:"):
                answer, ok_ = ask_gemini(api_key, chat_model, hist, results_now)
                if ok_ and st.session_state.get("nx_chat_used"):
                    st.caption(f"Answered by {st.session_state['nx_chat_used']}")
            if ok_:
                st.session_state["nx_chat"].append({"role": "assistant", "content": answer})
    if st.session_state["nx_chat"]:
        st.button("Clear the conversation", key="nx_clear", on_click=lambda: st.session_state.update(nx_chat=[]))

with tab_arch:
    H('<div class="nx-card"><div class="nx-kicker">Architecture · how the whole project works</div><div class="nx-read" style="margin-top:6px">'
      'From the raw data to the prediction on your screen. <b>Hover over (or tap) any box</b> for a plain explanation of what it does, why it is there and the numbers behind it.</div></div>')
    views = ["Research pipeline", "What happens when you click Analyse", "What it predicts"]
    view = st.segmented_control("View", views, default=views[0], key="nx_arch_view", selection_mode="single") or views[0]
    if view == views[0]:
        st.caption("How the models were built offline, step by step. This is the part of the project that was run once on a cloud GPU and then frozen.")
        H(AV.research_html(B))
    elif view == views[1]:
        st.caption("The live path on this page: your screenshots are read, the frozen networks run, and the result page is drawn. Nothing is retrained.")
        H(AV.live_html(B))
    else:
        st.caption("The 12 numbers the system produces for every stock: three kinds of prediction at four horizons.")
        H(AV.predicts_html(B))

with tab_m:
    H('<div class="nx-h">Which models make each call</div>')
    H('<div class="nx-note">For every task the three best models on validation data are averaged. When a fresh filing text is supplied, the text models are blended in with the weight shown; a weight of 1 means charts only.</div>')
    lab_cv = {m: v["label"] for m, v in B["cv_models"].items()}
    lab_nl = {m: v["label"] for m, v in B["nlp_models"].items()}
    names = {"dir": "Direction", "vol": "Volatility", "dd": "Drawdown"}
    body = ""
    for t in B["tasks"]:
        kd, h = t.split("_")
        cvs = "".join(f'<span class="nx-pill">{esc(lab_cv.get(m, m))}</span>' for m in B["cv_members"].get(t, []))
        nls = "".join(f'<span class="nx-pill">{esc(lab_nl.get(m, m))}</span>' for m in B["nlp_members"].get(t, []))
        body += f"<tr><td>{names[kd]} · {h} sessions</td><td>{cvs}</td><td>{nls}</td><td>{float(B['w_fresh'].get(t, 1.0)):.2f}</td></tr>"
    H(f'<div class="nx-card nx-tablewrap"><table class="nx-table"><thead><tr><th>Task</th><th>Chart models (top 3)</th><th>Text models (top 3)</th><th>Weight on charts</th></tr></thead><tbody>{body}</tbody></table></div>')
    H('<div class="nx-h">The full line-up</div>')
    H('<div class="nx-grid2"><div class="nx-card"><div class="nx-kicker">Chart-vision models (7)</div>' + "".join(f'<span class="nx-pill">{esc(v)}</span>' for v in lab_cv.values())
      + '</div><div class="nx-card"><div class="nx-kicker">Language models in the dashboard (6)</div>' + "".join(f'<span class="nx-pill">{esc(v)}</span>' for v in lab_nl.values()) + "</div></div>")

with tab_g:
    H("""<div class="nx-grid2">
    <div class="nx-card"><div class="nx-kicker">How to use it</div><div class="nx-read">Type a ticker and press Analyse. Screenshots are optional: when you add them (1W, 1D and 1H), every candle, price and volume bar is read from the pictures instead of being downloaded, and the price chart and price range use those prices. Paste a filing summary to let the text models join in. Add up to four stocks to compare them side by side.</div></div>
    <div class="nx-card"><div class="nx-kicker">Why the charts are redrawn</div><div class="nx-read">The models were trained on charts drawn in one exact format. Broker screenshots differ in colours, fonts and zoom, so the dashboard reads the candles, prices and volume out of your screenshots (or downloads them when you only type a ticker) and redraws the three charts (weekly, daily, hourly) in the training format. It shows you both.</div></div>
    <div class="nx-card"><div class="nx-kicker">Reading the outlook</div><div class="nx-read"><b>UP / SIDEWAYS / DOWN</b> compares the price after the horizon with a band of 0.5 x the stock's own volatility x √time. SIDEWAYS therefore means a small move for that stock, not a fixed percentage. <b>Drawdown</b> means the price dips more than 10% below today's close at some point in the window.</div></div>
    <div class="nx-card"><div class="nx-kicker">The price cone</div><div class="nx-read">A Monte Carlo simulation of thousands of possible paths, tuned so that its direction odds and volatility match the models. It shows a range of outcomes, calibrated on the test so that the nominal 80 % and 90 % ranges contain about that share of realised returns.</div></div></div>
    <div class="nx-note">Educational research. Not financial advice. A predictive association is not a cause, the models were trained on 50 large surviving US stocks, and nothing here accounts for costs, taxes or your own situation.</div>""")

H('<div class="nx-foot">NEXUS Alpha v3 · University of Wollongong in Dubai · frozen recipe ' + esc(str(B.get("stamp", ""))) + " · models frozen " + esc(str(B.get("frozen_utc", ""))) + "</div>")
