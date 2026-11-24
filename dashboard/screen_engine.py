"""Screenshot-driven analysis for the dashboard.

Same frozen models and the same calibration / blending / Monte Carlo chain as nav3_engine.analyse (the notebook code is not
changed); the only difference is where the bars come from: the candles, prices and volume are read out of the user's
TradingView screenshots by screen_digitizer, not downloaded. A timeframe without a screenshot falls back to Yahoo Finance and
is labelled as such.
"""
import math

import numpy as np
import pandas as pd

import nav3_engine as E
import screen_digitizer as SD
from nav3_core import CHART_BARS, CLASSES, HORIZONS, mc_simulate, prepare_arrays, render_candles, text_input, trailing_sigma, views_at

SLOT_VIEW = {"1W": "W", "1D": "D", "1H": "H"}
VIEW_NAME = {"W": "weekly", "D": "daily", "H": "hourly"}


def _arrays(df, n):
    x = df.tail(n)
    return tuple(x[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))


def analyse_screens(B, ticker, screenshots, form, text, n_paths=4000, seed=7, progress=None):
    """screenshots: {'1W'|'1D'|'1H': PIL image or None}. Returns the same fields as nav3_engine.analyse plus
    'D_screen' (daily bars read from the picture, for the price chart) and 'sources' (one row per timeframe)."""
    reads, problems, notes = {}, [], []
    for slot in ("1D", "1W", "1H"):
        im = screenshots.get(slot)
        if im is None:
            continue
        if progress:
            progress(f"Reading the {slot} screenshot")
        try:
            r = SD.read_chart(im, slot=slot)
        except SD.ChartReadError as e:
            problems.append(f"{slot} screenshot: {e}")
            continue
        tf = r.get("timeframe") or slot
        if tf in SLOT_VIEW and tf != slot and tf not in reads and screenshots.get(tf) is None:
            notes.append(f"the picture in the {slot} box is a {tf} chart, so it was used as {tf}")
            slot = tf
        reads[slot] = r
    if problems and not reads:
        raise ValueError("Could not read the screenshots: " + "; ".join(problems))

    # ticker: typed by the user, else read from the screenshot header
    tk = (ticker or "").strip().upper()
    how = "typed by the user"
    if not tk:
        for slot in ("1D", "1W", "1H"):
            c = (reads.get(slot) or {}).get("ticker_candidates") or []
            if c:
                tk, how = c[0].upper(), "read from the screenshot"
                break
    if not tk:
        tk, how = "CHART", "not found in the screenshot (type it in the ticker box to label the result)"

    # daily bars and their dates
    D = None
    sources = {}
    if "1D" in reads:
        r = reads["1D"]
        days, date_how = SD.assign_daily_dates(r)
        D = r["bars"].copy()
        D.insert(0, "date", pd.to_datetime(days))
        sources["D"] = dict(source="screenshot", read=r, date_how=date_how)
    Y = None                                         # Yahoo fallback, only for timeframes without a screenshot
    missing = [v for v in ("W", "D", "H") if {"W": "1W", "D": "1D", "H": "1H"}[v] not in reads]
    if missing:
        if tk == "CHART":
            raise ValueError("Add all three screenshots (1W, 1D, 1H), or type the ticker so the missing ones can be filled from market data.")
        if progress:
            progress("Filling the missing timeframes from Yahoo Finance")
        asof = D["date"].iloc[-1] if D is not None else None
        Y = E.fetch_market(tk, asof)
        if D is None:
            D = Y[0].tail(400).reset_index(drop=True)
            sources["D"] = dict(source="Yahoo Finance (no 1D screenshot)", read=None, date_how="market data")

    views, no_vol = {}, []
    for v in ("W", "D", "H"):
        slot = {"W": "1W", "D": "1D", "H": "1H"}[v]
        need = CHART_BARS[v]
        if slot in reads:
            df = reads[slot]["bars"]
            if not reads[slot]["has_volume"]:
                # the models were trained with a volume strip under every chart; flat bars stay closer to their training
                # inputs than an empty strip (measured on true data), and the page asks the user to switch Volume on
                df = df.assign(volume=1.0)
                no_vol.append(slot)
            views[v] = _arrays(df, need)
            sources.setdefault(v, dict(source="screenshot", read=reads[slot]))
            if len(df) < need:
                notes.append(f"the {slot} screenshot shows {len(df)} candles; the models were trained on {need}. Zoom out in TradingView for a closer match")
        else:
            S = prepare_arrays(*Y)
            vw = views_at(S, len(Y[0]) - 1)
            views[v] = vw[v]
            sources[v] = dict(source=f"Yahoo Finance (no {slot} screenshot)", read=None)
    img = np.stack([render_candles(*views[v]) for v in ("W", "D", "H")])[None]
    if no_vol:
        notes.append(f"no volume bars in the {', '.join(no_vol)} screenshot(s), so flat volume bars were drawn for the models. Turn on the Volume "
                     "indicator in TradingView for the closest match to how the models were trained (without it the probabilities can move by "
                     "about 3 to 6 percentage points)")

    if progress:
        progress("Running the vision models")
    cv = E.cv_chain(B, img)
    nlp = None
    if (text or "").strip() and B["nlp_union"]:
        if progress:
            progress("Running the language models")
        nlp = E.nlp_chain(B, [text_input(form, "summary", text)])
    fin = E.final_chain(B, cv, nlp)

    close = D["close"].to_numpy(float)
    p0 = float(close[-1])
    win = min(60, len(close) - 1)
    if win < 60:
        notes.append(f"volatility measured over the last {win} daily candles (60 in the study)")
    sig = trailing_sigma(close, win)
    if progress:
        progress("Simulating price paths")
    sim = mc_simulate(p0, sig, math.exp(float(fin["vol_20"][0])) * float(B.get("mc_scale", 1.0)), fin["dir_20"][0], nu=B["mc_nu"], n_days=20,
                      n_paths=n_paths, seed=seed)
    rows = []
    for h in HORIZONS:
        p = fin[f"dir_{h}"][0]
        k = int(np.argmax(p))
        thr = B["k_sideways"] * sig * math.sqrt(h)
        call = CLASSES[k] if not B["no_call"].get(f"dir_{h}", True) else "NO CALL"
        rows.append({"horizon (sessions)": h, "P(UP)": float(p[2]), "P(SIDEWAYS)": float(p[1]), "P(DOWN)": float(p[0]), "call": call,
                     "sideways band": f"+/-{100 * (math.exp(thr) - 1):.1f}%", "expected vol (ann.)": float(math.exp(fin[f"vol_{h}"][0])),
                     "P(drawdown > 10%)": float(fin[f"dd_{h}"][0]),
                     "CV only P(UP)": float(cv[f"dir_{h}"][0][2]), "text only P(UP)": (float(nlp[f"dir_{h}"][0][2]) if nlp is not None else None)})

    n_shot = sum(1 for v in sources.values() if v["source"] == "screenshot")
    data_how = ((("all prices and candles read from your screenshots" + (" (they show no volume)" if no_vol else ", volume too"))) if n_shot == 3 else
                f"{n_shot} of 3 timeframes read from your screenshots; the rest from Yahoo Finance")
    table = []
    for v in ("W", "D", "H"):
        s_ = sources[v]
        r_ = s_.get("read")
        if r_ is None:
            table.append(dict(timeframe=VIEW_NAME[v], source=s_["source"], candles="-", scale="-", check="-", volume="-"))
        else:
            table.append(dict(timeframe=VIEW_NAME[v], source="your screenshot", candles=f"{len(r_['bars'])} read, last {min(len(r_['bars']), CHART_BARS[v])} used",
                              scale=f"{r_['scale']} ({r_['axis_labels']} axis labels, fit within {r_['axis_err_px']:.1f} px)",
                              check=r_["legend_status"], volume="read" if r_["has_volume"] else "not in the screenshot (flat bars used)"))
    Dout = D[["date", "open", "high", "low", "close", "volume"]].tail(90).reset_index(drop=True)
    return {"ticker": tk, "asof": str(pd.Timestamp(D["date"].iloc[-1]).date()), "how": how, "p0": p0, "sigma": sig, "img": img[0],
            "table": pd.DataFrame(rows), "sim": sim, "used_text": nlp is not None, "in_universe": tk.replace("-", ".") in B["universe"],
            "ocr": {}, "D_screen": Dout, "data_how": data_how, "sources": table, "date_how": sources["D"].get("date_how", ""),
            "notes": notes + problems}
