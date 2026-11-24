"""Reads candles, volume, the price scale and the as-of date straight out of a TradingView-style chart screenshot.

Nothing here downloads data: every number comes from the pixels and the text in the picture.

  1. OCR the price-axis labels and fit pixel row -> price (linear or logarithmic scale, whichever fits).
  2. Learn the up / down candle colours and the background from the picture; a pixel belongs to a candle when its colour lies
     on the line from the background to a candle colour. Volume bars are the same colours blended toward the background
     (TradingView draws them semi-transparent), so the blend strength separates candles (strong) from volume (weak).
  3. Remove text (OCR boxes) and thin horizontal lines (the dotted last-price line) from the candle masks.
  4. Find the regular spacing of the candles, then read each candle's high / low (wick ends) and open / close (body ends).
  5. Check the last candle against the O/H/L/C legend; when they agree, the legend's high and low refine the price scale.
  6. Daily charts: the month labels on the time axis sit on the first session of each month, which pins the date of the
     last candle (NYSE calendar).
"""
import math
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd
from PIL import Image

MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
NUM = re.compile(r"^-?\d[\d,]*(?:\.\d+)?$")


class ChartReadError(ValueError):
    pass


# ------------------------------------------------------------------ OCR
def _ocr_items(arr, scale=1.0, dx=0, dy=0):
    import screen_reader
    res, _ = screen_reader._ocr()(arr)
    out = []
    for box, text, score in res or []:
        xs = [p[0] / scale + dx for p in box]
        ys = [p[1] / scale + dy for p in box]
        out.append(dict(x0=min(xs), x1=max(xs), y0=min(ys), y1=max(ys), xc=sum(xs) / 4, yc=sum(ys) / 4, t=str(text).strip(), s=float(score)))
    return out


def _ocr_region(img, box, up=2):
    """OCR a crop, upscaled for small fonts; coordinates come back in full-image pixels."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    crop = img.crop((x0, y0, x1, y1))
    if up != 1:
        crop = crop.resize((crop.size[0] * up, crop.size[1] * up), Image.LANCZOS)
    return _ocr_items(np.asarray(crop), scale=up, dx=x0, dy=y0)


def _num(t):
    t = t.replace(" ", "").replace("−", "-")
    if not NUM.match(t):
        return None
    try:
        return float(t.replace(",", ""))
    except ValueError:
        return None


# ------------------------------------------------------------------ price axis
def _fit_axis(pts):
    """pts: [(y_pixel, price)]. Robust (pair-RANSAC) fit of price against pixel row on a linear and a log scale."""
    best = None
    ys_all = np.array([p[0] for p in pts], float)
    for kind in ("linear", "log"):
        vals = np.array([p[1] for p in pts], float)
        if kind == "log":
            if np.any(vals <= 0):
                continue
            vals = np.log(vals)
        n = len(ys_all)
        best_in, best_res = None, np.inf
        for i in range(n):
            for j in range(i + 1, n):
                if abs(ys_all[i] - ys_all[j]) < 8:
                    continue
                b = (vals[j] - vals[i]) / (ys_all[j] - ys_all[i])
                if b >= 0:
                    continue
                a = vals[i] - b * ys_all[i]
                rpx = np.abs(a + b * ys_all - vals) / abs(b)
                inl = rpx < 3.0
                score = rpx[inl].sum()
                if best_in is None or inl.sum() > best_in.sum() or (inl.sum() == best_in.sum() and score < best_res):
                    best_in, best_res = inl, score
        if best_in is None or best_in.sum() < 3:
            continue
        b, a = np.polyfit(ys_all[best_in], vals[best_in], 1)
        rpx = np.abs(a + b * ys_all[best_in] - vals[best_in]) / abs(b)
        cand = dict(kind=kind, a=float(a), b=float(b), n=int(best_in.sum()), err_px=float(rpx.mean()), inliers=best_in)
        if best is None or cand["n"] > best["n"] or (cand["n"] == best["n"] and cand["err_px"] < best["err_px"]):
            best = cand
    return best


def _price(fit, y):
    v = fit["a"] + fit["b"] * np.asarray(y, float)
    return np.exp(v) if fit["kind"] == "log" else v


# ------------------------------------------------------------------ NYSE calendar (for dating daily candles)
def _nyse_holidays(y):
    def obs(d):
        return d - timedelta(1) if d.weekday() == 5 else (d + timedelta(1) if d.weekday() == 6 else d)

    def nth(m, wd, n):
        d = date(y, m, 1)
        d += timedelta((wd - d.weekday()) % 7)
        return d + timedelta(7 * (n - 1))

    def last(m, wd):
        d = (date(y, m + 1, 1) if m < 12 else date(y + 1, 1, 1)) - timedelta(1)
        return d - timedelta((d.weekday() - wd) % 7)

    a, b, c = y % 19, y // 100, y % 100
    d_, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d_ - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    easter = date(y, (h + l_ - 7 * m + 114) // 31, ((h + l_ - 7 * m + 114) % 31) + 1)
    hs = {nth(1, 0, 3), nth(2, 0, 3), easter - timedelta(2), last(5, 0), obs(date(y, 7, 4)), nth(9, 0, 1), nth(11, 3, 4), obs(date(y, 12, 25))}
    if date(y, 1, 1).weekday() != 5:
        hs.add(obs(date(y, 1, 1)))
    if y >= 2022:
        hs.add(obs(date(y, 6, 19)))
    return hs


def trading_days(start, end):
    hol = set()
    for y in range(start.year - 1, end.year + 2):
        hol |= _nyse_holidays(y)
    return [d.date() for d in pd.bdate_range(start, end) if d.date() not in hol]


def last_completed_session(now=None):
    now = pd.Timestamp.now(tz="America/New_York") if now is None else now
    days = trading_days(now.date() - timedelta(15), now.date())
    if days and days[-1] == now.date() and now.hour * 60 + now.minute < 16 * 60 + 15:
        days = days[:-1]
    return days[-1]


# ------------------------------------------------------------------ helpers
def _mode_color(px):
    q = (px // 6).astype(np.int32)
    key = q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2]
    vals, cnt = np.unique(key, return_counts=True)
    k = vals[np.argmax(cnt)]
    sel = key == k
    return px[sel].mean(0)


def _hue_chroma(px):
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    mx, mn = px.max(1), px.min(1)
    ch = mx - mn
    hue = np.zeros(len(px))
    nz = ch > 0
    rr = nz & (mx == r)
    gg = nz & (mx == g) & ~rr
    bb = nz & ~rr & ~gg
    hue[rr] = (60 * ((g[rr] - b[rr]) / ch[rr])) % 360
    hue[gg] = 60 * ((b[gg] - r[gg]) / ch[gg]) + 120
    hue[bb] = 60 * ((r[bb] - g[bb]) / ch[bb]) + 240
    return hue, ch


def _longest_run(rows, gap=2):
    """rows: sorted row indices. Longest run allowing small gaps (keeps one candle, drops stray pixels)."""
    if len(rows) == 0:
        return rows
    br = np.where(np.diff(rows) > gap + 1)[0]
    starts = np.r_[0, br + 1]
    ends = np.r_[br, len(rows) - 1]
    k = int(np.argmax(rows[ends] - rows[starts] + (ends - starts) * 1e-3))
    return rows[starts[k]: ends[k] + 1]


# ------------------------------------------------------------------ main reader
def read_chart(img, slot=None):
    """Returns a dict with the bars (oldest first), the fitted price scale, the legend check and diagnostics."""
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    img = img.convert("RGB")
    W, H = img.size
    arr = np.asarray(img).astype(np.int16)
    notes = []

    import screen_reader
    head = screen_reader.read_header(img)

    # --- text everywhere (to mask it) and the two axes (upscaled for the small fonts)
    items = _ocr_items(np.asarray(img))
    ax_items = _ocr_region(img, (int(W * 0.80), 0, W, H), up=2)
    labels = []
    for it in ax_items:
        v = _num(it["t"])
        if v is not None and it["xc"] > W * 0.82 and it["y1"] - it["y0"] < 40:
            labels.append(it | {"v": v})
    if len(labels) < 3:
        raise ChartReadError("could not read the price scale (fewer than 3 price labels on the right axis)")
    rights = np.array([l["x1"] for l in labels])                 # the main column of right-aligned labels
    med = np.median(rights)
    labels = [l for l in labels if abs(l["x1"] - med) < 30]

    # --- time axis (bottom) and the top of the chart
    title_y = [it["y0"] for it in items if ("·" in it["t"] or "•" in it["t"]) and it["y0"] < H * 0.25]
    y_top = int(max(0, (min(title_y) - 6) if title_y else H * 0.06))
    t_items = _ocr_region(img, (0, int(H * 0.75), int(min(l["x0"] for l in labels)) - 4, H), up=2)
    time_tok = [it for it in t_items if it["t"].lower()[:3] in MONTHS or re.fullmatch(r"(19|20)\d\d", it["t"]) or re.fullmatch(r"\d{1,2}", it["t"])]
    if time_tok:
        ty = np.median([it["y0"] for it in time_tok])
        time_tok = [it for it in time_tok if abs(it["y0"] - ty) < 12]
        y_bot = int(ty) - 4
    else:
        y_bot = int(H * 0.94)
        notes.append("time axis not read")

    # --- indicator panes (RSI, MACD...) sit under the price pane, split by full-width separator lines; keep the price pane
    lab_x0 = int(min(l["x0"] for l in labels))
    bg0 = _mode_color(arr[y_top:y_bot, 80:lab_x0 - 20].reshape(-1, 3).astype(float))
    seps = []
    for r in range(y_top + 10, y_bot - 10):
        row = arr[r, 80: W - 70].astype(float)
        mc = np.median(row, 0)
        if np.abs(mc - bg0).max() > 18 and (np.abs(row - mc).max(1) <= 5).mean() > 0.985:
            seps.append(r)
    bounds = [y_top] + [r for i, r in enumerate(seps) if i == 0 or r - seps[i - 1] > 3] + [y_bot]
    panes = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] - bounds[i] > 40]
    if len(panes) > 1:
        cnt = [sum(1 for l in labels if a_ <= l["yc"] <= b_) for a_, b_ in panes]
        y_top, y_bot = panes[int(np.argmax(cnt))]
        y_bot -= 2
        labels = [l for l in labels if y_top <= l["yc"] <= y_bot]
        notes.append(f"{len(panes) - 1} indicator pane(s) below the price chart ignored")
    fit = _fit_axis([(l["yc"], l["v"]) for l in labels])
    if fit is None:
        raise ChartReadError("the price labels on the right axis do not form a consistent scale")
    in_labels = [l for l, ok in zip(labels, fit["inliers"]) if ok]
    x_right = int(min(l["x0"] for l in in_labels)) - 12      # tag boxes start a few pixels left of the label text
    x_left = 0

    # --- colours: background and the two candle colours
    pane = arr[y_top:y_bot, x_left:x_right].reshape(-1, 3).astype(float)
    bg = _mode_color(pane)
    hue, ch = _hue_chroma(pane)
    dist = np.linalg.norm(pane - bg, axis=1)
    fam = {"up": (ch > 45) & (hue > 95) & (hue < 200), "down": (ch > 45) & ((hue < 25) | (hue > 330))}
    default = {"up": np.array([8, 153, 129.0]), "down": np.array([242, 54, 69.0])}
    col = {}
    for k_, m_ in fam.items():
        if m_.sum() < 40:
            col[k_] = default[k_]
            notes.append(f"{k_} colour not found, default used")
            continue
        fp = pane[m_]
        q = (fp // 6).astype(np.int32)
        key = q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2]
        vals, cnt = np.unique(key, return_counts=True)
        common = vals[cnt >= max(15, 0.03 * cnt.sum())]            # colours used a lot: candle bodies and volume bars
        if len(common) == 0:
            common = vals[np.argsort(cnt)[-3:]]
        ctr = np.array([fp[key == v].mean(0) for v in common])
        col[k_] = ctr[int(np.argmax(np.linalg.norm(ctr - bg, axis=1)))]   # the strongest of them is the candle colour

    # --- per-pixel membership: on the line background -> candle colour; alpha = how far along it
    P = arr.astype(float) - bg
    masks = {}
    for k_, c in col.items():
        u = c - bg
        uu = float(u @ u)
        alpha = (P @ u) / uu
        perp = np.linalg.norm(P - alpha[..., None] * u, axis=2)
        on = perp < np.maximum(14.0, 0.16 * math.sqrt(uu))
        masks[k_] = (on & (alpha >= 0.80), on & (alpha >= 0.28) & (alpha < 0.74))
    cand = {k_: v[0].copy() for k_, v in masks.items()}
    vol = {k_: v[1].copy() for k_, v in masks.items()}
    x_ax = min(W, x_right + 40)
    strips = {id(m_): m_[:, x_right:x_ax].copy() for m_ in list(cand.values()) + list(vol.values())}   # price tags run into the axis
    for m_ in list(cand.values()) + list(vol.values()):
        m_[:y_top] = False
        m_[y_bot:] = False
        m_[:, x_right:] = False
        m_[:, :x_left] = False
    # earnings / dividend markers sit in the bottom margin of the pane, where TradingView never draws candles
    band = int(0.05 * (y_bot - y_top))
    for m_ in cand.values():
        m_[y_bot - band: y_bot] = False
    # thin horizontal lines (dotted last-price line etc.)
    for m_ in cand.values():
        thin = m_ & ~np.roll(m_, 2, 0) & ~np.roll(m_, -2, 0)          # lines up to 3 px thick
        width = x_right - x_left
        seg = np.array_split(np.arange(x_left, x_right), 10)
        for r in np.where(thin.sum(1) > 0.22 * width)[0]:
            if sum(thin[r, s].any() for s in seg) >= 8:
                m_[r, :] = False

    # price tags: once the horizontal price line is gone, a coloured shape that runs into the price-axis column is a tag
    try:
        from scipy import ndimage as ndi
    except ImportError:
        ndi = None
    if ndi is not None:
        for m_ in list(cand.values()) + list(vol.values()):
            strip = strips[id(m_)]
            ext = np.concatenate([m_[y_top:y_bot, :x_right], strip[y_top:y_bot]], axis=1)
            lab, nlab = ndi.label(ext, structure=np.ones((3, 3), int))
            if nlab:
                tags = np.unique(lab[:, x_right + 2:])
                tags = tags[tags > 0]
                if len(tags):
                    kill = np.isin(lab[:, :x_right], tags)
                    m_[y_top:y_bot, :x_right][kill] = False
    # text (legend, buttons, markers) off the candle masks; tags next to the price axis are coloured boxes, so mask to the axis
    def _is_text(t):            # OCR sometimes "reads" rows of tiny bars as symbols or CJK characters: those are not text
        t = t.strip()
        return bool(t) and re.search(r"[A-Za-z0-9]", t) is not None and sum(c.isascii() for c in t) >= 0.8 * len(t)
    for it in items + ax_items:
        if it["yc"] < y_bot + 10 and it["x0"] < x_right and _is_text(it["t"]) and it["s"] >= 0.5:
            ya, yb = int(it["y0"]) - 4, int(it["y1"]) + 5
            xa, xb = int(it["x0"]) - 6, int(it["x1"]) + 6
            if it["x1"] > x_right - 0.06 * W:          # price tag at the axis: mask from just left of its text, so a touching candle survives
                xa, xb = int(it["x0"]) - 2, x_right
            for m_ in list(cand.values()) + list(vol.values()):
                m_[max(0, ya):yb, max(0, xa):xb] = False
    # indicator lines drawn in the candle colours (Bollinger Bands, coloured moving averages): pixels whose vertical run is
    # short, chained sideways over more than a candle spacing, are a line, not a candle
    try:
        from scipy import ndimage as ndi
    except ImportError:
        ndi = None
    if ndi is not None:
        for m_ in cand.values():
            up_run = np.zeros(m_.shape, np.int16)
            dn_run = np.zeros(m_.shape, np.int16)
            for r in range(1, m_.shape[0]):
                up_run[r] = np.where(m_[r], up_run[r - 1] + 1, 0)
            for r in range(m_.shape[0] - 2, -1, -1):
                dn_run[r] = np.where(m_[r], dn_run[r + 1] + 1, 0)
            vrun = np.where(m_, up_run + dn_run - 1, 0)
            thin = m_ & (vrun <= 3)
            lab, nlab = ndi.label(thin, structure=np.ones((3, 3), int))
            if nlab:
                sl = ndi.find_objects(lab)
                occ_ = m_.any(0)
                xo_ = np.where(occ_)[0]
                p_guess = max(4.0, (xo_[-1] - xo_[0]) / max(1, len(xo_)) * 1.6) if len(xo_) else 8.0
                for i_, s_ in enumerate(sl, start=1):
                    if s_ is not None and (s_[1].stop - s_[1].start) > max(14, 2.2 * p_guess):
                        m_[s_][lab[s_] == i_] = False
    # solid boxes wider than a candle body (price tags, buttons) are not candles
    cm0 = cand["up"] | cand["down"]
    occ0 = cm0.any(0)
    xo = np.where(occ0)[0]
    if len(xo) > 20:
        s0 = occ0[xo[0]: xo[-1] + 1].astype(float) - occ0[xo[0]: xo[-1] + 1].mean()
        ac0 = np.array([np.dot(s0[:-L], s0[L:]) for L in range(2, min(80, len(s0) // 3))])
        p_rough = 2 + int(np.argmax(ac0))
        lim = max(12, int(1.8 * p_rough))
        for m_ in cand.values():
            for r in np.where(m_.sum(1) > lim)[0]:
                row = m_[r]
                d_ = np.diff(np.r_[0, row.astype(np.int8), 0])
                st_, en_ = np.where(d_ == 1)[0], np.where(d_ == -1)[0]
                for a_, b_ in zip(st_, en_):
                    if b_ - a_ > lim:
                        row[a_:b_] = False
    # every real candle has a volume bar: nothing right of the last volume bar is a candle (price tags, labels)
    vm_low = (vol["up"] | vol["down"])[y_bot - int(0.35 * (y_bot - y_top)): y_bot]
    vcols = np.where(vm_low.sum(0) >= 1)[0]
    if len(vcols) > 0.25 * (x_right - x_left):
        cut = int(vcols.max()) + 3
        for m_ in cand.values():
            m_[:, cut:] = False
    cm = cand["up"] | cand["down"]
    colcnt = cm.sum(0).astype(float)
    occ = colcnt > 0
    xs_occ = np.where(occ)[0]
    if len(xs_occ) < 10:
        raise ChartReadError("no candles found (colours not recognised)")

    # --- candle spacing: autocorrelation, then a least-squares grid through well-separated candles
    s = occ[xs_occ[0]: xs_occ[-1] + 1].astype(float)
    s -= s.mean()
    ac = np.array([np.dot(s[:-L], s[L:]) for L in range(2, min(80, len(s) // 3))])
    lags = np.arange(2, 2 + len(ac))
    peaks = [i for i in range(1, len(ac) - 1) if ac[i] >= ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > 0.3 * ac.max()]
    p0 = float(lags[peaks[0]]) if peaks else float(lags[int(np.argmax(ac))])
    runs, start = [], None
    for x in range(xs_occ[0], xs_occ[-1] + 2):
        if x < len(occ) and occ[x]:
            start = x if start is None else start
        elif start is not None:
            runs.append((start, x - 1)); start = None
    centers = []
    for a_, b_ in runs:
        if b_ - a_ + 1 > 1.25 * p0:
            continue
        ext = []
        for x in range(a_, b_ + 1):
            rr = np.where(cm[:, x])[0]
            ext.append(rr.max() - rr.min() if len(rr) else 0)
        ext = np.array(ext)
        centers.append((np.mean(np.arange(a_, b_ + 1)[ext >= 0.9 * ext.max()]), ext.max()))
    merged = []                                  # specks next to a candle (text remnants, markers) are not candles of their own
    for c_, e_ in centers:
        if merged and c_ - merged[-1][0] < 0.55 * p0:
            if e_ > merged[-1][1]:
                merged[-1] = (c_, e_)
            continue
        merged.append((c_, e_))
    centers = np.array([c_ for c_, e_ in merged if e_ >= 2])
    tf = head.get("timeframe") or slot
    p_lab = _label_spacing(time_tok, tf)
    spacing_from = "candles"
    if len(centers) >= 8:
        # number the candles step by step (neighbour gaps are short, so rounding them is safe), then fit centre = c0 + k * p
        d = np.diff(centers)
        near = d[(d > 0.6 * p0) & (d < 1.4 * p0)]
        p = float(near.mean()) if len(near) >= 5 else p0
        for _ in range(3):
            k = np.r_[0, np.cumsum(np.maximum(1, np.round(d / p)))]
            p, c0 = np.polyfit(k, centers, 1)
            res_ = centers - (c0 + k * p)
            keep = np.abs(res_ - np.median(res_)) < 0.3 * p
            if keep.sum() >= 6:
                p, c0 = np.polyfit(k[keep], centers[keep], 1)
    if p_lab is not None and (len(centers) < 8 or abs(p / p_lab - 1) > 0.004):
        p = p_lab
        if len(centers) >= 8:
            right = centers[centers > np.quantile(centers, 0.6)]
            w_ = np.ones(len(right))
        else:                                   # candles touch: take the phase from the wicks (tallest column of each candle)
            xs_ = np.arange(cm.shape[1])
            ext_ = np.array([(np.ptp(np.where(cm[:, x])[0]) if cm[:, x].any() else 0) for x in xs_])
            right, w_ = xs_[ext_ > 0], ext_[ext_ > 0].astype(float) ** 2
        ang = np.angle(np.sum(w_ * np.exp(2j * np.pi * right / p)))
        c0 = float((ang / (2 * np.pi)) * p)
        spacing_from = "time-axis labels"
    elif len(centers) < 8:
        raise ChartReadError("too few separate candles to measure their spacing (zoom in a little in TradingView)")
    if p < 7 or spacing_from != "candles":
        xs_ = np.arange(cm.shape[1])
        ext_ = np.zeros(cm.shape[1])
        cols_ = np.where(cm.any(0))[0]
        for x in cols_:
            rr = np.where(cm[:, x])[0]
            ext_[x] = rr.max() - rr.min() + 1
        half_ = xs_occ[0] + 0.5 * (xs_occ[-1] - xs_occ[0])
        sel = (ext_ > 0) & (xs_ > half_)
        if sel.sum() > 20:
            xw, ww = xs_[sel].astype(float), ext_[sel] ** 2
            best_ = None
            for pt in p * (1 + np.linspace(-0.008, 0.008, 33)):
                z = np.sum(ww * np.exp(2j * np.pi * xw / pt)) / ww.sum()
                if best_ is None or abs(z) > best_[0]:
                    best_ = (abs(z), pt, np.angle(z))
            p = float(best_[1])
            c0 = float(best_[2] / (2 * np.pi) * p)
    kmin = int(math.floor((xs_occ[0] - c0) / p + 0.5))
    kmax = int(math.floor((xs_occ[-1] - c0) / p + 0.5))

    # --- body width (to tell bodies from 1-pixel wicks)
    half = max(1, int(0.45 * p))
    rows_w = []
    for k in range(kmin, kmax + 1, max(1, (kmax - kmin) // 60)):
        c = int(round(c0 + k * p))
        sub = cm[:, max(0, c - half): c + half + 1]
        if sub.any():
            rows_w.append(sub.sum(1).max())
    body_w = float(np.median(rows_w)) if rows_w else 1.0
    min_body = 2 if body_w >= 2 else 1

    # --- read every candle
    bars, missing = [], 0
    vtop_lim = y_bot - int(0.55 * (y_bot - y_top))
    vm_all = vol["up"] | vol["down"]
    for k in range(kmin, kmax + 1):
        c = c0 + k * p
        xa, xb = int(round(c - 0.42 * p)), int(round(c + 0.42 * p)) + 1
        xa, xb = max(x_left, xa), min(x_right, max(xb, xa + 1))
        nu, nd = int(cand["up"][:, xa:xb].sum()), int(cand["down"][:, xa:xb].sum())
        vsub = vm_all[vtop_lim:y_bot, xa:xb]
        vrows = np.where(vsub.any(1))[0]
        vtop = vtop_lim + int(vrows.min()) if len(vrows) else None
        vbot = vtop_lim + int(vrows.max()) if len(vrows) else None
        if nu + nd == 0:
            bars.append({"k": k} if vtop is None else {"k": k, "vtop": vtop, "vbot": vbot})
            missing += 1
            continue
        up = nu >= nd
        m_ = (cand["up"] if up else cand["down"])[:, xa:xb]
        rows = _longest_run(np.where(m_.any(1))[0])
        wid = m_[rows].sum(1)
        body = rows[wid >= max(min_body, 0.5 * wid.max())] if wid.max() >= min_body else rows
        bars.append({"up": up, "hy": int(rows.min()), "ly": int(rows.max()), "bt": int(body.min()), "bb": int(body.max()), "x": float(c), "k": k,
                     "vtop": vtop, "vbot": vbot})
    # trim leading / trailing empty slots
    while bars and "hy" not in bars[0]:
        bars.pop(0)
    while bars and "hy" not in bars[-1]:
        bars.pop()
    filled_ = [("hy" not in b_) for b_ in bars]
    for j in range(len(bars) - 1, max(0, len(bars) - 8), -1):
        if j >= 3 and all(filled_[j - 3: j]) and sum(1 for f_ in filled_[j:] if not f_) <= 3:
            notes.append(f"{len(bars) - j} mark(s) after a gap at the right edge ignored (price tag, not a candle)")
            del bars[j - 3:]
            while bars and "hy" not in bars[-1]:
                bars.pop()
            break

    def to_df(fit_):
        out, prev = [], None
        for b_ in bars:
            if "hy" not in b_:
                out.append(dict(open=prev, high=prev, low=prev, close=prev, vtop=b_.get("vtop"), vbot=b_.get("vbot"), filled=True))
                continue
            hi, lo = float(_price(fit_, b_["hy"])), float(_price(fit_, b_["ly"] + 1))
            top, bot = float(_price(fit_, b_["bt"])), float(_price(fit_, b_["bb"] + 1))
            o, c = (bot, top) if b_["up"] else (top, bot)
            out.append(dict(open=o, high=hi, low=lo, close=c, vtop=b_["vtop"], vbot=b_["vbot"], filled=False))
            prev = c
        return pd.DataFrame(out)

    df = to_df(fit)
    if len(df) < 20:
        raise ChartReadError(f"only {len(df)} candles could be read")

    # --- volume, measured up from the common baseline
    vb = df["vbot"].dropna()
    if len(vb) >= 0.5 * len(df):
        base = float(np.median(vb)) + 1
        df["volume"] = [max(0.0, base - t) if t is not None and not (isinstance(t, float) and np.isnan(t)) else 0.0 for t in df["vtop"]]
        has_vol = bool((df["volume"] > 0).mean() > 0.5)
    else:
        df["volume"] = 0.0
        has_vol = False
    if not has_vol:
        notes.append("no volume bars found")
    else:                                          # every real candle has a volume bar: trailing slots without one are line remnants
        while len(df) > 20 and df["volume"].iloc[-1] == 0:
            df = df.iloc[:-1]
            bars.pop()

    # --- legend check: the legend normally shows the last candle; when it agrees, its high/low refine the scale
    legend = head.get("ohlc")
    legend_status = "no legend read"
    if legend:
        last = df.iloc[-1]
        err = max(abs(last["high"] / legend["high"] - 1), abs(last["low"] / legend["low"] - 1), abs(last["close"] / legend["close"] - 1))
        if err < 0.03:
            lb = bars[-1]
            pts = [(l["yc"], l["v"]) for l in in_labels] + [(lb["hy"], legend["high"])] * 2 + [(lb["ly"] + 1, legend["low"])] * 2
            fit2 = _fit_axis(pts)
            if fit2 is not None and fit2["kind"] == fit["kind"]:
                fit = fit2
                vol_keep = df["volume"].to_numpy()
                df = to_df(fit)
                df["volume"] = vol_keep
            for kk in ("open", "high", "low", "close"):
                df.loc[df.index[-1], kk] = legend[kk]
            legend_status = f"last candle matches the legend ({100 * err:.2f}% apart before refining)"
        else:
            legend_status = f"legend differs from the last candle by {100 * err:.1f}% (probably a hovered bar); not used"

    df = df.drop(columns=["vtop", "vbot"])

    # --- dates (daily charts): month / year labels sit on the first session of the month
    dates, date_note = None, None
    if tf == "1D" and time_tok:
        dates, date_note = _date_daily(time_tok)
    return dict(_xs=np.array([c0 + b_["k"] * p for b_ in bars]), bars=df, timeframe=tf, scale=fit["kind"], axis_labels=len(in_labels), axis_err_px=fit["err_px"], spacing_px=float(p), spacing_from=spacing_from,
                legend=legend, legend_status=legend_status, has_volume=has_vol, filled=int(df["filled"].sum()), dates=dates, date_note=date_note,
                ticker_candidates=head.get("candidates", []), name=head.get("name"), notes=notes)


def _date_daily(time_tok):
    """(year, month) for every month / year label on a daily chart's time axis, plus the labels themselves."""
    toks = []
    for it in sorted(time_tok, key=lambda t: t["xc"]):
        t = it["t"].lower()
        if re.fullmatch(r"(19|20)\d\d", t):
            toks.append(["year", int(t), it["xc"]])
        elif t[:3] in MONTHS:
            toks.append(["month", MONTHS[t[:3]], it["xc"]])
    if not toks:
        return None, "no month labels"
    years = [None] * len(toks)
    yi = [i for i, t in enumerate(toks) if t[0] == "year"]
    if yi:
        cur, prev = None, 0
        for i, t in enumerate(toks):
            if t[0] == "year":
                cur, prev = t[1], 1
                years[i] = (cur, 1)
            elif cur is not None:
                if t[1] < prev:
                    cur += 1
                years[i] = (cur, t[1]); prev = t[1]
        cur, prev = toks[yi[0]][1] - 1, 13
        for i in range(yi[0] - 1, -1, -1):
            m = toks[i][1]
            if m > prev:
                cur -= 1
            years[i] = (cur, m); prev = m
    else:
        today = date.today()
        m = toks[-1][1]
        cur = today.year if m <= today.month else today.year - 1
        prev = 13
        for i in range(len(toks) - 1, -1, -1):
            m = toks[i][1]
            if m > prev:
                cur -= 1
            years[i] = (cur, m); prev = m
    return years, toks


def _label_spacing(time_tok, tf):
    """Pixels per candle from the time-axis labels: the number of candles between two labels is known from the calendar
    (trading days for daily, weeks for weekly, 7 regular-hours bars per session for hourly). None when not enough labels."""
    if not time_tok or tf not in ("1D", "1W", "1H"):
        return None
    toks = sorted(time_tok, key=lambda t: t["xc"])
    pts = []
    if tf in ("1D", "1W"):
        ym, raw = _date_daily([t for t in toks if not re.fullmatch(r"\d{1,2}", t["t"])])
        if not raw:
            return None
        for (yy, mm), t in zip(ym, raw):
            first = trading_days(date(yy, mm, 1), date(yy, mm, 1) + timedelta(10))[0]
            pts.append((t[2], first))
        base = pts[0][1]
        if tf == "1D":
            idx = [len(trading_days(base, d)) - 1 if d >= base else -(len(trading_days(d, base)) - 1) for _, d in pts]
        else:
            idx = [((d - timedelta(d.weekday())) - (base - timedelta(base.weekday()))).days // 7 for _, d in pts]
    else:
        today = date.today()
        mon = [i for i, t in enumerate(toks) if t["t"].lower()[:3] in MONTHS]
        if not mon:
            return None
        m0 = MONTHS[toks[mon[0]]["t"].lower()[:3]]
        best = None
        for back in (1, 2, 0):                     # month of the day numbers before the first month label (a label can be missed)
            m = (m0 - back - 1) % 12 + 1
            y = (today.year if m0 <= today.month else today.year - 1) - (1 if m > m0 else 0)
            prev_day, ds = 0, []
            for t in toks:
                tt = t["t"].lower()
                if tt[:3] in MONTHS:
                    mm = MONTHS[tt[:3]]
                    if mm < m:
                        y += 1
                    m, prev_day = mm, 1
                    ds.append((t["xc"], date(y, m, 1)))
                elif re.fullmatch(r"\d{1,2}", tt):
                    dd = int(tt)
                    if dd < prev_day:
                        m = m + 1 if m < 12 else 1
                        y = y + 1 if m == 1 else y
                    prev_day = dd
                    try:
                        ds.append((t["xc"], date(y, m, dd)))
                    except ValueError:
                        continue
            if len(ds) < 3:
                continue
            base = min(d for _, d in ds)
            kk = np.array([7 * (len(trading_days(base, d)) - 1) for _, d in ds], float)
            xx = np.array([x_ for x_, _ in ds], float)
            if kk.max() - kk.min() < 20:
                continue
            pp, cc = np.polyfit(kk, xx, 1)
            rms = float(np.sqrt(np.mean((xx - (cc + pp * kk)) ** 2))) / max(pp, 1e-6)
            if best is None or rms < best[0]:
                best = (rms, ds, list(kk))
        if best is None:
            return None
        pts, idx = best[1], best[2]
    x = np.array([p_[0] for p_ in pts], float)
    k = np.array(idx, float)
    if len(x) < 3 or k.max() - k.min() < 20:
        return None
    for _ in range(3):
        pp, cc = np.polyfit(k, x, 1)
        r_ = np.abs(x - (cc + pp * k))
        keep = r_ < max(3.0, 1.2 * pp)
        if keep.sum() < 3 or keep.all():
            break
        x, k = x[keep], k[keep]
    return float(pp) if pp > 0.8 else None


def assign_daily_dates(res):
    """Turns the label positions into one date per bar. Returns (dates, how)."""
    df = res["bars"]
    n = len(df)
    if res.get("dates") is None:
        d = last_completed_session()
        days = trading_days(d - timedelta(int(n * 1.6) + 20), d)[-n:]
        return days, f"time axis not read; assumed the last candle is the latest completed session ({d})"
    years, toks = res["dates"], res["date_note"]
    xs_bar = res["_xs"]
    implied = []
    for (yy, mm), t in zip(years, toks):
        x = t[2]
        j = int(np.argmin(np.abs(xs_bar - x)))
        if abs(xs_bar[j] - x) > 0.75 * res["spacing_px"]:
            continue
        first = trading_days(date(yy, mm, 1), date(yy, mm, 1) + timedelta(10))[0]
        fwd = trading_days(first, first + timedelta(int((n - 1 - j) * 1.6) + 30))
        if n - 1 - j < len(fwd):
            implied.append(fwd[n - 1 - j])
    if not implied:
        d = last_completed_session()
        days = trading_days(d - timedelta(int(n * 1.6) + 20), d)[-n:]
        return days, f"month labels could not be matched to candles; assumed the last candle is the latest completed session ({d})"
    vals, cnt = np.unique(np.array(implied, dtype="datetime64[D]"), return_counts=True)
    asof = pd.Timestamp(vals[np.argmax(cnt)]).date()
    how = f"dated from the time-axis month labels ({cnt.max()} of {len(implied)} labels agree)"
    latest = last_completed_session()
    if asof > latest:
        how += f"; that is after the latest completed session, so the last candle was set to {latest}"
        asof = latest
    days = trading_days(asof - timedelta(int(n * 1.6) + 30), asof)[-n:]
    return days, how
