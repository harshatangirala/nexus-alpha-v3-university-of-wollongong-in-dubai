
"""NEXUS Alpha v3 shared core: chart rendering, model architectures, calibration and Monte Carlo.
Executed by the notebook and imported by the Streamlit dashboard (identical code in both)."""
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HORIZONS = [5, 10, 20, 40]
CLASSES = ["DOWN", "SIDEWAYS", "UP"]
VIEWS = ["W", "D", "H"]
CHART_BARS = {"W": 52, "D": 63, "H": 70}
# palette index: 0 background, 1 up candle, 2 down candle, 3 up volume, 4 down volume
PALETTE_DARK = np.array([[19, 23, 34], [38, 166, 154], [239, 83, 80], [26, 110, 103], [160, 56, 54]], np.uint8)
PALETTE_LIGHT = np.array([[255, 255, 255], [8, 153, 129], [242, 54, 69], [150, 210, 200], [245, 170, 170]], np.uint8)


# ------------------------------------------------------------------ bars -> arrays
def prepare_arrays(d, hb):
    """d: daily bars (date, open, high, low, close, volume), oldest first; hb: hourly bars with a 'session' column
    (New York date of the bar). Returns plain arrays used by views_at (fast, no pandas in the loop)."""
    dates = pd.DatetimeIndex(d["date"]).values.astype("datetime64[D]")
    days = dates.astype(np.int64)
    wkey = days - (days + 3) % 7                      # Monday of each week (1970-01-01 was a Thursday)
    A = {k: d[k].values.astype(float) for k in ("open", "high", "low", "close", "volume")}
    ukeys, wstart = np.unique(wkey, return_index=True)
    wend = np.r_[wstart[1:], len(days)]
    W = {"open": A["open"][wstart], "close": A["close"][wend - 1],
         "high": np.maximum.reduceat(A["high"], wstart), "low": np.minimum.reduceat(A["low"], wstart),
         "volume": np.add.reduceat(A["volume"], wstart)}
    hs = pd.DatetimeIndex(hb["session"]).values.astype("datetime64[D]")
    sidx = np.searchsorted(dates, hs)
    ok = (sidx < len(dates)) & (dates[np.minimum(sidx, len(dates) - 1)] == hs)
    Hh = {k: hb[k].values.astype(float)[ok] for k in ("open", "high", "low", "close", "volume")}
    return {"dates": dates, "D": A, "wkey": wkey, "ukeys": ukeys, "wstart": wstart, "W": W, "H": Hh, "hsidx": sidx[ok]}


def views_at(S, i, bars=None):
    """(o, h, l, c, v) arrays for the weekly, daily and hourly chart of as-of daily row i. Uses nothing after row i."""
    bars = bars or CHART_BARS
    A, W = S["D"], S["W"]
    sl = slice(max(0, i - bars["D"] + 1), i + 1)
    out = {"D": tuple(A[k][sl] for k in ("open", "high", "low", "close", "volume"))}
    wp = int(np.searchsorted(S["ukeys"], S["wkey"][i]))
    a = max(0, wp - bars["W"] + 1)
    s0 = int(S["wstart"][wp])
    part = (A["open"][s0], A["high"][s0: i + 1].max(), A["low"][s0: i + 1].min(), A["close"][i], A["volume"][s0: i + 1].sum())
    out["W"] = tuple(np.r_[W[k][a:wp], part[j]] for j, k in enumerate(("open", "high", "low", "close", "volume")))
    e = int(np.searchsorted(S["hsidx"], i, side="right"))
    hs = slice(max(0, e - bars["H"]), e)
    out["H"] = tuple(S["H"][k][hs] for k in ("open", "high", "low", "close", "volume"))
    return out


def render_candles(o, h, l, c, v, size=224):
    """Palette-index candlestick chart (uint8, size x size) from bar arrays in time order (oldest first)."""
    o, h, l, c, v = (np.asarray(x, dtype=float) for x in (o, h, l, c, v))
    n = len(c)
    img = np.zeros((size, size), np.uint8)
    if n == 0:
        return img
    top, bot = 4, int(round(size * 0.79))
    vtop, vbot = int(round(size * 0.82)), size - 4
    lo, hi = float(np.nanmin(l)), float(np.nanmax(h))
    span = hi - lo if hi > lo else max(abs(hi), 1.0) * 1e-6
    def ypix(p):
        return top + (bot - 1 - top) * (hi - np.asarray(p, dtype=float)) / span
    pxpb = size / n
    xs = ((np.arange(n) + 0.5) * pxpb).astype(int)
    half = max(1, int(pxpb * 0.35))
    yh, yl, yo, yc = (np.clip(np.round(ypix(a)), top, bot - 1).astype(int) for a in (h, l, o, c))
    up = c >= o
    for k in range(n):
        col = 1 if up[k] else 2
        img[yh[k]: yl[k] + 1, xs[k]] = col
        b0, b1 = min(yo[k], yc[k]), max(yo[k], yc[k])
        img[b0: b1 + 1, max(0, xs[k] - half): min(size, xs[k] + half + 1)] = col
    vv = np.nan_to_num(v)
    vmax = float(vv.max()) if len(vv) else 0.0
    if vmax > 0:
        hgt = np.round(vv / vmax * (vbot - vtop)).astype(int)
        for k in range(n):
            if hgt[k] > 0:
                img[vbot - hgt[k]: vbot, max(0, xs[k] - half): min(size, xs[k] + half + 1)] = 3 if up[k] else 4
    return img


def render_views(S, i, size=224):
    """(3, size, size) palette-index stack in the order W, D, H."""
    vw = views_at(S, i)
    return np.stack([render_candles(*vw[k], size=size) for k in VIEWS])


def trailing_sigma(close, window=60):
    lr = np.diff(np.log(np.asarray(close, dtype=float)))
    return float(np.std(lr[-window:], ddof=1)) if len(lr) >= window else float("nan")


def to_float_images(idx, palette):
    """idx: uint8 tensor (B, V, H, W) of palette indices; palette: float tensor (P, 3). Returns (B, V, 3, H, W) in [0, 1]."""
    x = palette[idx.long()]
    return x.permute(0, 1, 4, 2, 3).contiguous() / 255.0


# ------------------------------------------------------------------ vision models
class JKXNet(nn.Module):
    """CNN in the style of Jiang, Kelly and Xiu (2023): (5x3) convolutions, BatchNorm, LeakyReLU and max-pooling,
    the first layer with vertical stride 3 and dilation 2; widened to four blocks for 224-px charts; trained from scratch."""
    def __init__(self, chans=(64, 128, 256, 512)):
        super().__init__()
        layers, cin = [], 3
        for j, co in enumerate(chans):
            if j == 0:
                conv = nn.Conv2d(cin, co, (5, 3), stride=(3, 1), dilation=(2, 1), padding=(12, 1))
            else:
                conv = nn.Conv2d(cin, co, (5, 3), padding=(2, 1))
            layers += [conv, nn.BatchNorm2d(co), nn.LeakyReLU(0.01), nn.MaxPool2d((2, 2))]
            cin = co
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = chans[-1]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.pool(self.body(x)).flatten(1)


class YoloFeat(nn.Module):
    """Backbone of an Ultralytics YOLOv8 classification model; returns the pooled features that feed its linear layer."""
    def __init__(self, weights="yolov8s-cls.pt"):
        super().__init__()
        import os
        os.environ.setdefault("YOLO_VERBOSE", "False")
        from ultralytics import YOLO
        seq = YOLO(weights).model.model
        head = seq[-1]
        self.layers = nn.ModuleList(list(seq)[:-1])
        self.conv, self.pool = head.conv, head.pool
        self.feat_dim = int(head.linear.in_features)
        for p in self.parameters():
            p.requires_grad_(True)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.pool(self.conv(x)).flatten(1)


def build_backbone(spec, pretrained=True):
    """(module, feature_dim, mean, std) for a CV model spec from the configuration."""
    if spec["kind"] == "timm":
        import timm
        from timm.data import resolve_data_config
        m = timm.create_model(spec["timm"], pretrained=pretrained, num_classes=0)
        dc = resolve_data_config({}, model=m)
        return m, int(m.num_features), tuple(dc["mean"]), tuple(dc["std"])
    if spec["kind"] == "jkx":
        m = JKXNet()
        return m, m.feat_dim, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    if spec["kind"] == "yolo":
        m = YoloFeat(spec["weights"])
        return m, m.feat_dim, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    raise ValueError(spec["kind"])


class MultiTaskHead(nn.Module):
    """Shared MLP, then direction logits (n_h x 3), standardised log-volatility (n_h) and drawdown logits (n_h)."""
    def __init__(self, in_dim, n_h=4, hidden=512, p=0.2):
        super().__init__()
        self.n_h = n_h
        self.body = nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(p), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(p))
        self.dir = nn.Linear(hidden, n_h * 3)
        self.vol = nn.Linear(hidden, n_h)
        self.dd = nn.Linear(hidden, n_h)
    def forward(self, z):
        b = self.body(z)
        return {"dir": self.dir(b).view(-1, self.n_h, 3), "vol": self.vol(b), "dd": self.dd(b)}


class ChartNet(nn.Module):
    """One backbone shared by the weekly, daily and hourly charts; the three feature vectors are concatenated."""
    def __init__(self, spec, pretrained=True, n_views=3, n_h=4):
        super().__init__()
        self.backbone, self.feat_dim, mean, std = build_backbone(spec, pretrained)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.n_views = n_views
        self.head = MultiTaskHead(self.feat_dim * n_views, n_h)
    def embed(self, x):
        b, v = x.shape[:2]
        z = self.backbone((x.flatten(0, 1) - self.mean) / self.std)
        return z.view(b, v, -1)
    def forward(self, x):
        return self.head(self.embed(x).flatten(1))


# ------------------------------------------------------------------ text models
class TextNet(nn.Module):
    """Hugging Face encoder, attention-masked mean pooling, multi-task head."""
    def __init__(self, hf_id, n_h=4, from_config=False, grad_ckpt=False):
        super().__init__()
        from transformers import AutoModel, AutoConfig
        if from_config:
            self.enc = AutoModel.from_config(AutoConfig.from_pretrained(hf_id))
        else:
            self.enc = AutoModel.from_pretrained(hf_id, torch_dtype=torch.float32)   # some checkpoints are stored in fp16
        self.enc = self.enc.float()
        if grad_ckpt and hasattr(self.enc, "gradient_checkpointing_enable"):
            self.enc.gradient_checkpointing_enable()
        self.is_longformer = getattr(self.enc.config, "model_type", "") == "longformer"
        self.head = MultiTaskHead(int(self.enc.config.hidden_size), n_h)
    def pooled(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        kw = {}
        if self.is_longformer:
            g = torch.zeros_like(attention_mask); g[:, 0] = 1
            kw["global_attention_mask"] = g
        if inputs_embeds is not None:
            out = self.enc(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kw).last_hidden_state
        else:
            out = self.enc(input_ids=input_ids, attention_mask=attention_mask, **kw).last_hidden_state
        out = out.float()
        m = attention_mask.unsqueeze(-1).float()
        return (out * m).sum(1) / m.sum(1).clamp(min=1.0)
    def forward(self, input_ids, attention_mask):
        return self.head(self.pooled(input_ids, attention_mask))


def text_input(form, section, body, max_chars=60000):
    """The exact string a text model reads: form and section tags, then the text."""
    return f"[{form}] [{section}] " + " ".join(str(body or "").split())[:max_chars]


# ------------------------------------------------------------------ raw outputs -> calibrated predictions
def softmax_np(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def apply_calibration(raw, cal):
    """raw: {'dir_h': (n,3) logits, 'vol_h': (n,) log-vol, 'dd_h': (n,) logit}; cal from calibrate_fit (notebook).
    Returns {'dir_h': (n,3) probabilities, 'vol_h': (n,), 'dd_h': (n,) probabilities}."""
    out = {}
    for h in HORIZONS:
        T = cal.get(f"T_{h}", 1.0)
        out[f"dir_{h}"] = softmax_np(np.asarray(raw[f"dir_{h}"], dtype=float) / T)
        a, b = cal.get(f"vol_ab_{h}", (0.0, 1.0))
        out[f"vol_{h}"] = a + b * np.asarray(raw[f"vol_{h}"], dtype=float)
        a, b = cal.get(f"dd_ab_{h}", (0.0, 1.0))
        out[f"dd_{h}"] = 1.0 / (1.0 + np.exp(-(a + b * np.asarray(raw[f"dd_{h}"], dtype=float))))
    return out


def _default_amp():
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


@torch.no_grad()
def chart_raw_outputs(model, idx_u8, palette_t, vol_mean, vol_std, device, batch=128, amp_dtype=None):
    """Raw outputs of a ChartNet for palette-index images (n, 3, H, W) as numpy uint8."""
    model.eval()
    res = {"dir": [], "vol": [], "dd": []}
    for s in range(0, len(idx_u8), batch):
        x = to_float_images(torch.from_numpy(np.ascontiguousarray(idx_u8[s: s + batch])).to(device), palette_t)
        with torch.autocast(device_type=device.type, dtype=amp_dtype or _default_amp(), enabled=device.type == "cuda"):
            o = model(x)
        for k in res:
            res[k].append(o[k].float().cpu().numpy())
    d, v, q = (np.concatenate(res[k]) for k in ("dir", "vol", "dd"))
    out = {}
    for j, h in enumerate(HORIZONS):
        out[f"dir_{h}"] = d[:, j]
        out[f"vol_{h}"] = v[:, j] * vol_std[j] + vol_mean[j]
        out[f"dd_{h}"] = q[:, j]
    return out


@torch.no_grad()
def text_raw_outputs(model, tok, texts, max_len, vol_mean, vol_std, device, batch=8, amp_dtype=None, use_amp=True):
    model.eval()
    res = {"dir": [], "vol": [], "dd": []}
    for s in range(0, len(texts), batch):
        enc = tok(list(texts[s: s + batch]), truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype or _default_amp(), enabled=use_amp and device.type == "cuda"):
            o = model(enc["input_ids"], enc["attention_mask"])
        for k in res:
            res[k].append(o[k].float().cpu().numpy())
    d, v, q = (np.concatenate(res[k]) for k in ("dir", "vol", "dd"))
    out = {}
    for j, h in enumerate(HORIZONS):
        out[f"dir_{h}"] = d[:, j]
        out[f"vol_{h}"] = v[:, j] * vol_std[j] + vol_mean[j]
        out[f"dd_{h}"] = q[:, j]
    return out


def raw_from_probs(pred):
    """Calibrated predictions back to 'raw' form (log-probabilities, logits, log-volatility) so an ensemble or blend
    can be re-calibrated like any single model."""
    raw = {}
    for t, v in pred.items():
        v = np.asarray(v, dtype=float)
        if t.startswith("dir"):
            raw[t] = np.log(np.clip(v, 1e-9, 1))
        elif t.startswith("dd"):
            q = np.clip(v, 1e-6, 1 - 1e-6)
            raw[t] = np.log(q / (1 - q))
        else:
            raw[t] = v
    return raw


def average_predictions(preds):
    """Equal-weight ensemble of calibrated predictions (probabilities and log-volatility are averaged)."""
    keys = preds[0].keys()
    return {k: np.mean([np.asarray(p[k], dtype=float) for p in preds], axis=0) for k in keys}


def blend(p_a, p_b, w):
    """w * p_a + (1 - w) * p_b for every task (w can differ by task: dict task -> weight)."""
    return {k: (w[k] if isinstance(w, dict) else w) * np.asarray(p_a[k]) + (1 - (w[k] if isinstance(w, dict) else w)) * np.asarray(p_b[k])
            for k in p_a}


# ------------------------------------------------------------------ Monte Carlo price paths
def mc_simulate(p0, sigma_t, vol_ann, p_dir, nu=6, n_days=20, n_paths=4000, k=0.5, dd=-0.10, seed=0):
    """Daily log returns = mu + s * e, e ~ standardised Student-t(nu) (nu = 0 -> normal), s = vol_ann / sqrt(252).
    mu is chosen on a grid so that the simulated DOWN / SIDEWAYS / UP shares at n_days (same thresholds as the labels:
    +-k * sigma_t * sqrt(n_days)) are closest (KL divergence) to the model's calibrated probabilities p_dir.
    Intraday lows between consecutive closes come from the exact Brownian-bridge minimum, giving P(drawdown > 10 %)."""
    rng = np.random.default_rng(seed)
    s = float(vol_ann) / math.sqrt(252.0)
    if nu and nu > 2:
        e = rng.standard_t(nu, size=(n_paths, n_days)) * math.sqrt((nu - 2) / nu)
    else:
        e = rng.standard_normal((n_paths, n_days))
    thr = k * float(sigma_t) * math.sqrt(n_days)
    S = e.sum(1)
    grid = np.linspace(-0.5, 0.5, 201) * s
    R = grid[:, None] * n_days + s * S[None, :]
    pu, pdn = (R > thr).mean(1), (R < -thr).mean(1)
    q = np.clip(np.vstack([pdn, 1 - pu - pdn, pu]).T, 1e-4, 1.0)
    p = np.clip(np.asarray(p_dir, dtype=float), 1e-4, 1.0)
    mu = float(grid[int(np.argmin((p * np.log(p / q)).sum(1)))])
    lr = mu + s * e
    logp = math.log(p0) + np.cumsum(lr, axis=1)
    a = np.concatenate([np.full((n_paths, 1), math.log(p0)), logp[:, :-1]], axis=1)
    u = rng.random((n_paths, n_days))
    lows = (a + logp - np.sqrt((logp - a) ** 2 - 2 * s ** 2 * np.log(u))) / 2
    p_dd = float((lows.min(1) <= math.log(p0) + math.log(1 + dd)).mean())
    R_end = logp[:, -1] - math.log(p0)
    sim_dir = [float((R_end < -thr).mean()), float(((R_end >= -thr) & (R_end <= thr)).mean()), float((R_end > thr).mean())]
    qs = np.percentile(np.exp(logp), [5, 25, 50, 75, 95], axis=0)
    return {"paths": np.exp(logp), "mu_daily": mu, "sd_daily": s, "p_dd": p_dd, "sim_dir": sim_dir, "quantiles": qs,
            "thr": thr, "ret_q": np.percentile(R_end, [5, 10, 25, 50, 75, 90, 95])}
