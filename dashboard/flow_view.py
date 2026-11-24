"""'How this result was made': an interactive flowchart of the analysis for the stock on screen.

Everything shown comes from the frozen bundle (which networks are used for which target, the text weights, the calibration and
the Monte Carlo settings) and from this stock's own run (what each vision network said). Nothing is invented for the picture.
"""
import html
import math

import numpy as np
import torch

import nav3_engine as E
from nav3_core import PALETTE_DARK, apply_calibration, chart_raw_outputs

esc = html.escape

TARGET = {"dir": "Direction", "vol": "Volatility", "dd": "Drawdown"}
CV_ROLE = {
    "convnext_t_ft": "ConvNeXt-Tiny: a modern convolutional network",
    "effnet_b0_ft": "EfficientNet-B0: a small, efficient convolutional network",
    "jkx_cnn_ft": "Custom CNN: a small convolutional network built for this project",
    "resnet50_ft": "ResNet-50: the classic 50-layer convolutional network",
    "swin_t_ft": "Swin-T: a vision transformer that looks at shifted windows",
    "vit_b16_ft": "ViT-B/16: a vision transformer that reads the chart as 16x16 patches",
    "yolov8_cls_ft": "YOLOv8s-cls: the image-classification version of YOLO",
}
NLP_ROLE = {
    "tfidf_lr": "TF-IDF + logistic regression: a classic word-count model",
    "finbert_ft": "FinBERT: BERT trained on financial text",
    "secbert_ft": "SEC-BERT: BERT trained on SEC filings",
    "roberta_l_ft": "RoBERTa-large: a large general language model",
    "longformer_ft": "Longformer: a language model for long documents",
    "deberta_l_ft": "DeBERTa-v3-large: a large general language model",
}


def cv_each(B, img3):
    """What every vision network says about this chart stack (calibrated): {model: {task: value}}.
    dir_h -> [P(DOWN), P(SIDEWAYS), P(UP)], vol_h -> log volatility, dd_h -> P(fall > 10 %)."""
    pal = torch.tensor(PALETTE_DARK, dtype=torch.float32, device=E.DEV)
    out = {}
    for mid in B["cv_union"]:
        net, mu, sd = E._cv_model(B, mid)
        cal = apply_calibration(chart_raw_outputs(net, np.asarray(img3)[None], pal, mu, sd, E.DEV), B["cal"][mid])
        out[mid] = {t: (cal[t][0].tolist() if t.startswith("dir") else float(cal[t][0])) for t in B["tasks"]}
    return out


def _fmt(kind, v):
    if kind == "dir":
        return f"&#9650; {v[2]:.0%} &middot; &#9660; {v[0]:.0%}"
    if kind == "vol":
        return f"{math.exp(v):.0%} / yr"
    return f"{v:.0%}"


def _ens(kind, vals):
    if kind == "vol":
        return float(np.mean(vals))
    return np.mean(np.asarray(vals, float), axis=0)


def _final(r, kind, h):
    row = r["table"][r["table"]["horizon (sessions)"] == h].iloc[0]
    if kind == "dir":
        return f"&#9650; {row['P(UP)']:.0%} &middot; &#9660; {row['P(DOWN)']:.0%} &middot; call {esc(str(row['call']))}"
    if kind == "vol":
        return f"{row['expected vol (ann.)']:.0%} / yr"
    return f"{row['P(drawdown > 10%)']:.0%}"


def _stage(n, title, body, cls=""):
    return f'<div class="fl-stage {cls}"><div class="fl-n">{n}</div><div class="fl-b"><div class="fl-t">{title}</div>{body}</div></div>'


ARROW = '<div class="fl-arrow">&#9660;</div>'


def flow_html(B, r, each, kind, h):
    task = f"{kind}_{h}"
    lab_cv = {m: v["label"] for m, v in B["cv_models"].items()}
    lab_nl = {m: v["label"] for m, v in B["nlp_models"].items()}
    members = B["cv_members"][task]
    nl_members = B["nlp_members"].get(task, [])
    w = float(B["w_fresh"].get(task, 1.0))
    used_text = bool(r.get("used_text"))

    # 1. input and reading
    shots = r.get("sources")
    src = ("your screenshots" if shots and any(s["source"] == "your screenshot" for s in shots) else "market data (no screenshots given)")
    s1 = _stage(1, "Your input", f'<div class="fl-d">Charts from <b>{src}</b>' + ("; filing text: <b>yes</b>" if used_text else "; filing text: <b>none</b> (charts only)") +
                '. The price axis, candles and volume are read straight from the picture; a small OCR network (RapidOCR) reads only the numbers and the ticker.</div>')
    s2 = _stage(2, "Redrawn in the training format",
                '<div class="fl-d">Three charts at 224&times;224: <b>weekly</b> (52 candles), <b>daily</b> (63) and <b>hourly</b> (70). '
                'The networks never see your broker&#39;s colours, fonts or indicators.</div>')

    # 3a. vision networks
    pills = ""
    order = {m: i + 1 for i, m in enumerate(members)}
    for m in B["cv_union"]:
        on = m in order
        val = _fmt(kind, each[m][task]) if on else "not used for this target"
        pills += (f'<div class="fl-m {"on" if on else "off"}" title="{esc(CV_ROLE.get(m, ""))}">'
                  f'<div class="fl-mt">{(f"<span class=fl-rank>#{order[m]}</span>" if on else "")}{esc(lab_cv.get(m, m))}</div>'
                  f'<div class="fl-mv">{val}</div></div>')
    vis = (f'<div class="fl-col"><div class="fl-ct">7 vision networks read the three charts</div>{pills}'
           f'<div class="fl-d">For <b>{TARGET[kind].lower()} at {h} sessions</b> only the <b>3 best on validation data (2022-23)</b> are averaged. '
           'Hover a network for what it is.</div></div>')

    # 3b. language networks
    if used_text:
        npills = ""
        norder = {m: i + 1 for i, m in enumerate(nl_members)}
        for m in B["nlp_union"]:
            on = m in norder
            npills += (f'<div class="fl-m {"on" if on else "off"}" title="{esc(NLP_ROLE.get(m, ""))}"><div class="fl-mt">'
                       f'{(f"<span class=fl-rank>#{norder[m]}</span>" if on else "")}{esc(lab_nl.get(m, m))}</div>'
                       f'<div class="fl-mv">{"in the top 3" if on else "not used for this target"}</div></div>')
        txt = (f'<div class="fl-col"><div class="fl-ct">6 language networks read the filing text</div>{npills}'
               f'<div class="fl-d">Top 3 for this target, averaged. Their weight in the final blend: <b>{1 - w:.0%}</b> '
               + ("(none: text was not useful for this target on validation data)" if w >= 1 else "") + '</div></div>')
    else:
        txt = ('<div class="fl-col fl-dim"><div class="fl-ct">6 language networks (not used now)</div>'
               '<div class="fl-d">No filing text was given, so this path is skipped. Paste a 10-K / 10-Q / 8-K summary to switch it on.</div></div>')
    s3 = f'<div class="fl-stage fl-wide"><div class="fl-n">3</div><div class="fl-b"><div class="fl-two">{vis}{txt}</div></div></div>'

    # 4. average and blend
    avg = _fmt(kind, _ens(kind, [each[m][task] for m in members]))
    if used_text and w < 1:
        blend_txt = (f'Chart average <b>{avg}</b> is blended with the text average: <b>{w:.0%}</b> charts + <b>{1 - w:.0%}</b> text.')
    elif used_text:
        blend_txt = f'Chart average <b>{avg}</b>. The text models have weight 0 for this target, so the answer comes from the charts only.'
    else:
        blend_txt = f'Average of the 3 chart networks: <b>{avg}</b>.'
    s4 = _stage(4, "Average the top 3 and blend", f'<div class="fl-d">{blend_txt}</div>')
    s5 = _stage(5, "Calibrate", '<div class="fl-d">One last correction (fitted on validation data) so that a stated 30 % really means about 30 %. '
                f'Result for this target: <b class="fl-res">{_final(r, kind, h)}</b></div>', "fl-final")

    # 6. what the numbers feed
    row20 = r["table"][r["table"]["horizon (sessions)"] == 20].iloc[0]
    s6 = _stage(6, "What the numbers become",
                '<div class="fl-three">'
                f'<div class="fl-o"><b>Calls and probabilities</b><br>DOWN / SIDEWAYS / UP at 5, 10, 20, 40 sessions. A call is shown for a horizon only if the locked test supported it (all four direction horizons did).</div>'
                f'<div class="fl-o"><b>Risk numbers</b><br>Expected volatility and chance of a &gt;10 % fall, each from its own top-3 networks.</div>'
                f'<div class="fl-o"><b>Price cone (Monte Carlo)</b><br>{int(len(r["sim"]["paths"])):,} paths. Drift from the 20-session direction '
                f'(P(UP) {row20["P(UP)"]:.0%}), spread from the 20-session volatility ({row20["expected vol (ann.)"]:.0%} &times; {float(B.get("mc_scale", 1.0)):.2f} '
                f'width factor), normally distributed daily shocks.</div></div>')
    return '<div class="fl">' + s1 + ARROW + s2 + ARROW + s3 + ARROW + s4 + ARROW + s5 + ARROW + s6 + "</div>"


def matrix_html(B, focus=None):
    """Which network is used for which of the 12 targets (the whole routing table)."""
    lab_cv = {m: v["label"] for m, v in B["cv_models"].items()}
    lab_nl = {m: v["label"] for m, v in B["nlp_models"].items()}
    head = "<th>Target</th>" + "".join(f'<th class="fl-vh" title="{esc(CV_ROLE.get(m, ""))}">{esc(lab_cv[m])}</th>' for m in B["cv_union"])
    head += '<th class="fl-sep"></th>' + "".join(f'<th class="fl-vh" title="{esc(NLP_ROLE.get(m, ""))}">{esc(lab_nl[m])}</th>' for m in B["nlp_union"]) + "<th>Text weight</th>"
    rows = ""
    for t in B["tasks"]:
        k, h = t.split("_")
        cells = "".join(f'<td class="fl-c">{"<span class=fl-dot></span>" if m in B["cv_members"][t] else ""}</td>' for m in B["cv_union"])
        cells += '<td class="fl-sep"></td>' + "".join(f'<td class="fl-c">{"<span class=fl-dot2></span>" if m in B["nlp_members"].get(t, []) else ""}</td>' for m in B["nlp_union"])
        w = 1 - float(B["w_fresh"].get(t, 1.0))
        cls = ' class="fl-focus"' if focus == t else ""
        rows += f"<tr{cls}><td><b>{TARGET[k]}</b> &middot; {h} sessions</td>{cells}<td>{w:.0%}</td></tr>"
    return (f'<div class="nx-card nx-tablewrap"><table class="nx-table fl-mx"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>'
            '<div class="nx-note" style="margin-top:10px">A dot means the network is one of the top 3 for that target and is averaged into the answer. '
            'The same recipe is used for every stock: the networks are chosen by target and horizon, not by sector. '
            'Text weight is the share of the final blend that comes from the language networks when a filing text is given.</div></div>')
