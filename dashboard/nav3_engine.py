
"""NEXUS Alpha v3 dashboard engine: loads the frozen bundle and reproduces the notebook's deploy_recipe chain exactly."""
import os
import json
import math
import numpy as np
import pandas as pd
import torch
from nav3_core import (PALETTE_DARK, HORIZONS, CLASSES, ChartNet, TextNet, prepare_arrays, render_views, trailing_sigma,
                       chart_raw_outputs, text_raw_outputs, apply_calibration, raw_from_probs, blend, text_input, mc_simulate)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NY = "America/New_York"
_CACHE = {}


def load_bundle(path):
    with open(os.path.join(path, "bundle.json")) as f:
        B = json.load(f)
    B["dir"] = path
    return B


def _cv_model(B, mid):
    if ("cv", mid) not in _CACHE:
        ck = torch.load(B["cv_models"][mid]["path"], map_location="cpu", weights_only=False)
        net = ChartNet(ck["spec"], pretrained=False)
        net.load_state_dict(ck["state_dict"])
        _CACHE[("cv", mid)] = (net.to(DEV).eval(), np.asarray(ck["vol_mu"]), np.asarray(ck["vol_sd"]))
    return _CACHE[("cv", mid)]


def _nlp_model(B, mid):
    if ("nlp", mid) not in _CACHE:
        spec = B["nlp_models"][mid]
        if spec["kind"] == "tfidf":
            import joblib
            _CACHE[("nlp", mid)] = joblib.load(spec["path"])
        else:
            from transformers import AutoTokenizer
            ck = torch.load(spec["path"], map_location="cpu", weights_only=False)
            net = TextNet(ck["hf"], from_config=True)
            net.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in ck["state_dict"].items()})
            _CACHE[("nlp", mid)] = (net.to(DEV).eval(), AutoTokenizer.from_pretrained(ck["hf"]), int(ck["max_len"]),
                                    np.asarray(ck["vol_mu"]), np.asarray(ck["vol_sd"]), bool(ck.get("amp", True)))
    return _CACHE[("nlp", mid)]


def cv_chain(B, img_idx):
    """img_idx: (n, 3, 224, 224) uint8 palette indices -> calibrated cv_top3 predictions."""
    pal = torch.tensor(PALETTE_DARK, dtype=torch.float32, device=DEV)
    cal = {}
    for mid in B["cv_union"]:
        net, mu, sd = _cv_model(B, mid)
        cal[mid] = apply_calibration(chart_raw_outputs(net, img_idx, pal, mu, sd, DEV), B["cal"][mid])
    avg = {t: np.mean([cal[m][t] for m in B["cv_members"][t]], axis=0) for t in B["tasks"]}
    return apply_calibration(raw_from_probs(avg), B["cal"]["cv_top3"])


def nlp_raw(B, mid, texts):
    spec = B["nlp_models"][mid]
    obj = _nlp_model(B, mid)
    if spec["kind"] == "tfidf":
        X = obj["vectorizer"].transform(texts)
        raw = {}
        for t in B["tasks"]:
            m = obj["models"][t]
            kd = t.split("_")[0]
            if kd == "vol":
                raw[t] = m.predict(X)
            elif kd == "dir":
                raw[t] = np.log(np.clip(m.predict_proba(X), 1e-9, 1))
            else:
                q = np.clip(m.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
                raw[t] = np.log(q / (1 - q))
        return raw
    net, tok, L, mu, sd, amp = obj
    return text_raw_outputs(net, tok, [x[: L * 8] for x in texts], L, mu, sd, DEV, batch=4, use_amp=amp)


def nlp_chain(B, texts):
    tr = {}
    for mid in B["nlp_union"]:
        ev = apply_calibration(nlp_raw(B, mid, texts), B["cal_event"][mid])
        tr[mid] = apply_calibration(raw_from_probs(ev), B["cal"][B["nlp_models"][mid]["transfer_id"]])
    avg = {t: np.mean([tr[m][t] for m in B["nlp_members"][t]], axis=0) for t in B["tasks"]}
    return apply_calibration(raw_from_probs(avg), B["cal"]["nlp_deploy_top3"])


def final_chain(B, cv, nlp=None):
    mix = cv if nlp is None else blend(cv, nlp, B["w_fresh"])
    return apply_calibration(raw_from_probs(mix), B["cal"]["deploy_recipe"])


def fetch_market(ticker, asof=None):
    """Yahoo daily (3 years) and hourly (60 days) bars, split-adjusted and not dividend-adjusted (as in training);
    today's unfinished session is dropped; asof (date) truncates both."""
    import yfinance as yf
    tk = yf.Ticker(ticker.upper().replace(".", "-"))
    d = tk.history(period="3y", interval="1d", auto_adjust=False, actions=False)
    h = tk.history(period="60d", interval="60m", auto_adjust=False, actions=False, prepost=False)
    if d is None or d.empty or h is None or h.empty:
        raise ValueError(f"Yahoo returned no data for {ticker}")
    di = pd.DatetimeIndex(d.index)
    di = di.tz_convert(NY).tz_localize(None) if di.tz is not None else di
    D = pd.DataFrame({"date": di.normalize(), "open": d["Open"].values, "high": d["High"].values, "low": d["Low"].values,
                      "close": d["Close"].values, "volume": d["Volume"].values.astype(float)}).dropna().reset_index(drop=True)
    hi = pd.DatetimeIndex(h.index)
    hi = hi.tz_localize(NY) if hi.tz is None else hi.tz_convert(NY)
    Hh = pd.DataFrame({"timestamp": hi.tz_convert("UTC"), "session": hi.tz_localize(None).normalize(), "minute": hi.minute,
                       "open": h["Open"].values, "high": h["High"].values, "low": h["Low"].values, "close": h["Close"].values,
                       "volume": h["Volume"].values.astype(float)}).dropna()
    Hh = Hh[Hh["minute"] == 30].drop(columns="minute").reset_index(drop=True)
    now = pd.Timestamp.now(tz=NY)
    if len(D) and D["date"].iloc[-1] == now.tz_localize(None).normalize() and now.hour * 60 + now.minute < 16 * 60 + 15:
        D = D.iloc[:-1]
    if asof is not None:
        D = D[D["date"] <= pd.Timestamp(asof)]
    Hh = Hh[Hh["session"].isin(set(D["date"]))].reset_index(drop=True)
    return D.reset_index(drop=True), Hh


def match_legend(D, ohlc, tol=0.006):
    """Session whose daily O/H/L/C matches a screenshot legend (pins the as-of date), else None."""
    if not ohlc:
        return None
    x = D.tail(400)
    err = np.max(np.abs(x[["open", "high", "low", "close"]].values / np.array([ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"]]) - 1), axis=1)
    k = int(np.argmin(err))
    return x["date"].iloc[k] if err[k] < tol else None


def analyse(B, ticker, screenshots, form, text, n_paths=4000, seed=7):
    """screenshots: dict '1W'/'1D'/'1H' -> PIL image or None. Returns everything the page shows."""
    info, how = {}, "typed by the user"
    if any(v is not None for v in screenshots.values()):
        try:
            import screen_reader
            for slot, im in screenshots.items():
                if im is not None:
                    info[slot] = screen_reader.read_header(im)
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"
    cands = [c for s in ("1D", "1W", "1H") for c in (info.get(s) or {}).get("candidates", [])]
    legend = (info.get("1D") or {}).get("ohlc")
    tk = (ticker or "").strip().upper()
    D = Hh = None
    if not tk:
        for c in cands[:6]:
            try:
                D_, H_ = fetch_market(c)
            except Exception:
                continue
            if legend is None or match_legend(D_, legend) is not None:
                tk, D, Hh, how = c, D_, H_, ("read from the screenshot and confirmed by its price legend" if legend else "read from the screenshot (unconfirmed)")
                break
        if not tk:
            raise ValueError("Could not identify the ticker from the screenshots: type it in the ticker box.")
    if D is None:
        D, Hh = fetch_market(tk)
    asof = match_legend(D, legend)
    if asof is not None:
        D, Hh = fetch_market(tk, asof)
        how += f"; as-of date pinned by the daily screenshot's legend ({pd.Timestamp(asof).date()})"
    if len(D) < 300 or len(Hh) < 70:
        raise ValueError(f"Not enough history for {tk} (daily {len(D)}, hourly {len(Hh)} bars).")
    S = prepare_arrays(D, Hh)
    img = render_views(S, len(D) - 1)[None]
    cv = cv_chain(B, img)
    nlp = nlp_chain(B, [text_input(form, "summary", text)]) if (text or "").strip() and B["nlp_union"] else None
    fin = final_chain(B, cv, nlp)
    p0 = float(D["close"].iloc[-1]); sig = trailing_sigma(D["close"].values, 60)
    sim = mc_simulate(p0, sig, math.exp(float(fin["vol_20"][0])) * float(B.get("mc_scale", 1.0)), fin["dir_20"][0], nu=B["mc_nu"], n_days=20, n_paths=n_paths, seed=seed)
    rows = []
    for h in HORIZONS:
        p = fin[f"dir_{h}"][0]; k = int(np.argmax(p)); thr = B["k_sideways"] * sig * math.sqrt(h)
        call = CLASSES[k] if not B["no_call"].get(f"dir_{h}", True) else "NO CALL"
        rows.append({"horizon (sessions)": h, "P(UP)": float(p[2]), "P(SIDEWAYS)": float(p[1]), "P(DOWN)": float(p[0]), "call": call,
                     "sideways band": f"+/-{100 * (math.exp(thr) - 1):.1f}%", "expected vol (ann.)": float(math.exp(fin[f"vol_{h}"][0])),
                     "P(drawdown > 10%)": float(fin[f"dd_{h}"][0]),
                     "CV only P(UP)": float(cv[f"dir_{h}"][0][2]), "text only P(UP)": (float(nlp[f"dir_{h}"][0][2]) if nlp is not None else None)})
    return {"ticker": tk, "asof": str(pd.Timestamp(D["date"].iloc[-1]).date()), "how": how, "p0": p0, "sigma": sig, "img": img[0],
            "table": pd.DataFrame(rows), "sim": sim, "used_text": nlp is not None, "in_universe": tk.replace("-", ".") in B["universe"],
            "ocr": {k: {kk: v for kk, v in (vv or {}).items() if kk != "header_text"} if isinstance(vv, dict) else vv for k, vv in info.items()}}
