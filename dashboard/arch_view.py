"""Architecture tab: how the whole project works, from data collection to the prediction on screen.

Three interactive views (all pop-ups open on hover, tap or keyboard focus; pure CSS, so nothing is sent anywhere):
  1. Research pipeline  - how the models were built offline (data -> pre-processing -> models -> combine -> locked test)
  2. Live pipeline      - what happens between pressing Analyse and the result page
  3. What it predicts   - the 12 targets (3 kinds x 4 horizons) with the locked-test verdict for each

Numbers come from the frozen bundle or from the study's result tables; nothing here is invented for the picture.
"""
import html

esc = html.escape

TARGET_INFO = {
    "dir": ("Direction", "Will the price end the window clearly UP, clearly DOWN, or SIDEWAYS? SIDEWAYS means a move smaller than 0.5 x the stock's own "
                         "60-day volatility x the square root of the horizon, so it is a small move for that stock, not a fixed percentage.",
            "macro-F1 against a shuffled-label null"),
    "vol": ("Volatility", "How jumpy will the price be over the window? The target is the log of realised volatility measured from hourly bars.",
            "R-squared against predicting the average"),
    "dd": ("Drawdown", "Will the price fall more than 10 % below today's close at some point inside the window? A yes / no event, reported as a probability.",
           "AUC (0.5 = guessing)"),
}
HZ = {5: "about 1 week", 10: "about 2 weeks", 20: "about 1 month", 40: "about 2 months"}


def node(title, sub, tip_title, tip, tag="", cls=""):
    """One box with a pop-up. `tip` is HTML (short paragraphs and bullet lists)."""
    return (f'<div class="ar-node {cls}" tabindex="0"><div class="ar-nt">{title}</div><div class="ar-ns">{sub}</div>'
            + (f'<div class="ar-tag">{tag}</div>' if tag else "")
            + f'<div class="ar-tip"><div class="ar-tt">{tip_title}</div>{tip}</div></div>')


def lane(n, title, sub, nodes, side="mid"):
    return (f'<div class="ar-lane ar-{side}"><div class="ar-lh"><span class="ar-ln">{n}</span><div><div class="ar-lt">{title}</div>'
            f'<div class="ar-ls">{sub}</div></div></div>{"".join(nodes)}</div>')


def _ul(*items):
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


# ------------------------------------------------------------------ 1. research pipeline
def research_html(B):
    cv = {m: v["label"] for m, v in B["cv_models"].items()}
    n_cv, n_nlp = len(B["cv_union"]), len(B["nlp_union"])
    l1 = lane(1, "Data collection", "what we gathered", [
        node("Price bars", "daily, weekly, hourly", "Price data",
             "<p>Open, high, low, close and volume for every stock, Oct 2016 to Oct 2026. Weekly bars are built from the daily ones; hourly bars use "
             "regular trading hours only (7 per session).</p>" + _ul("Daily and weekly bars give the long view", "Hourly bars give the short-term texture and the volatility target"),
             "2016 to 2026"),
        node("SEC filings", "10-K, 10-Q, 8-K", "Company filings from EDGAR",
             "<p>The text the language models read. 10-K = annual report, 10-Q = quarterly report, 8-K = event announcement (results, officer changes, deals).</p>"
             + _ul("554 10-Ks, 1,700 10-Qs, 8,015 8-Ks downloaded", "8-Ks are 79 % of the training events, so most text samples are short announcements"),
             "10,269 filings"),
        node("Universe", "50 stocks, 5 sectors", "Which stocks",
             "<p>50 large US stocks, 10 in each of five sectors: Information Technology, Financials, Health Care, Consumer Discretionary and Communication Services. "
             "Stocks without a full decade of hourly bars were swapped for reserves (for example AVGO for TXN).</p>"
             + _ul("Large, liquid, well covered by filings", "Limitation: survivors only, no small caps"), "50 stocks"),
    ], "left")
    l2 = lane(2, "Pre-processing", "turning raw data into model inputs", [
        node("Chart images", "3 pictures per row", "Charts as images",
             "<p>For every decision day the weekly (52 candles), daily (63) and hourly (70) charts are drawn at 224 x 224 pixels in one fixed style, "
             "with a volume strip underneath. Each chart is scaled to its own high and low.</p>"
             + _ul("193,650 images, 9.7 GB", "Fixed style = the networks never see broker colours or fonts",
                   "Design consequence: the absolute volatility level is hidden, which is why charts struggle on the volatility target"), "224 x 224"),
        node("Filing text", "cleaned and tagged", "Text preparation",
             "<p>Each filing becomes the string the models read: a form tag, a section tag, then the cleaned text (very long filings are cut to a maximum length).</p>"
             + _ul("Built so that only information available at the decision date is used (leakage tests were run)", "A filing counts as fresh for 5 sessions"), "text"),
        node("Labels", "3 targets x 4 horizons", "What the models must predict",
             "<p>Twelve targets: direction, volatility and a 10 % drawdown, each at 5, 10, 20 and 40 sessions.</p>"
             + _ul("Direction: DOWN / SIDEWAYS / UP with a band of 0.5 x the stock's own volatility x sqrt(horizon)",
                   "Volatility: log realised volatility from hourly bars", "Drawdown: price falls more than 10 % below today's close"), "12 targets"),
        node("Time splits", "train / 2021 / validation / test", "Splits with no leakage",
             "<p>Everything is split by time, never at random.</p>" + _ul("Train Oct 2016 to 2020: 51,250 rows", "Early stopping 2021: 2,150 rows",
             "Model selection 2022 to 2023: 4,600 rows", "Locked test Jan 2024 to Aug 2026: 6,550 rows", "A 40-session gap between blocks so labels cannot overlap"),
             "64,550 rows"),
    ], "left")
    l3 = lane(3, "Models", "13 networks plus simple benchmarks", [
        node(f"{n_cv} vision networks", "read the three charts", "Chart-reading networks",
             "<p>One backbone reads the weekly, daily and hourly chart; the three feature vectors are joined and passed to a small head with 12 outputs "
             "(direction, volatility and drawdown at four horizons).</p>" + _ul(*[esc(v) for v in cv.values()]) +
             "<p>Pre-trained on ImageNet and fine-tuned, except the custom CNN, which is trained from scratch as the academic baseline.</p>", f"{n_cv} networks"),
        node(f"{n_nlp} language models", "read the filing", "Text-reading models",
             "<p>Each model reads the filing text and outputs the same 12 numbers.</p>" + _ul("TF-IDF + logistic regression (classic baseline)", "FinBERT (finance)", "SEC-BERT (filings)",
             "RoBERTa-large", "Longformer (long documents)", "DeBERTa-v3-large") +
             "<p>A seventh, FinGPT (frozen, 4-bit), was analysed in the research but is not in the dashboard.</p>", f"{n_nlp} models"),
        node("Benchmarks", "numbers only, no images or text", "What the networks must beat",
             "<p>Simple models on prices alone, plus controls.</p>" + _ul("<b>prior</b>: base rates, no information", "<b>baseline_simple</b>: momentum / HAR volatility / trailing-volatility logistic",
             "<b>tech_gbm</b>: LightGBM on 20 technical indicators", "Filing-event versions: company base rates and no-text models on the same filings",
             "Controls: scratch CNN, random weights, shuffled labels, TF-IDF"), "the bar"),
        node("Training recipe", "same for every network", "How they were trained",
             "<p>Fine-tuning with a warm-up and cosine learning-rate schedule, mixed precision, up to 6 epochs with early stopping on the 2021 block.</p>"
             + _ul("Trained on a cloud GPU (Colab)", "One training seed per network (a stated limitation)", "Most networks peaked after the first epoch"), "Colab GPU"),
    ], "mid")
    l4 = lane(4, "Combine and calibrate", "from 13 networks to one answer", [
        node("Top 3 per target", "chosen on validation", "Specialist selection",
             "<p>For each of the 12 targets, the 3 best chart networks and the 3 best text models on 2022 to 2023 data are averaged. "
             "So different networks serve different targets, and the same recipe is used for every stock (it is not chosen by sector).</p>"
             + _ul("Example: direction at 20 sessions uses ConvNeXt-T, ViT-B/16 and Swin-T", "Drawdown at 20 sessions uses EfficientNet-B0, ViT-B/16 and Swin-T"), "3 + 3"),
        node("Text blend", "weight differs by target", "Blending charts and text",
             "<p>When a fresh filing is available the two averages are blended with a weight chosen on validation. The text share is 0 % for direction at 5, 10 and 40 sessions, "
             "25 % for direction at 20 and for short volatility, 75 % for volatility at 40 sessions and 100 % for drawdown.</p>"
             + _ul("Without a filing text the answer comes from the charts only"), "w per target"),
        node("Calibration", "so 30 % means about 30 %", "Calibration",
             "<p>Raw network outputs are over- or under-confident. A correction fitted on held-out data (temperature for direction, a straight line for volatility, a logistic curve "
             "for drawdown) is applied to each network and again to the final blend.</p>", "deploy recipe"),
        node("Monte Carlo cone", "4,000 price paths", "Turning numbers into a price range",
             f"<p>The 20-session direction sets the drift and the 20-session volatility, multiplied by a width factor of {float(B.get('mc_scale', 1.0)):.2f}, sets the spread of simulated daily returns. "
             "The factor was chosen on validation so the ranges are not too narrow.</p>"
             + _ul("On the locked test the nominal 80 % and 90 % ranges contained 83 % and 91 % of the 20-day returns", "It is a range of what could happen, not a forecast of one price"),
             "4,000 paths"),
    ], "mid")
    l5 = lane(5, "Outputs and testing", "what you see, and how we checked it", [
        node("Direction calls", "DOWN / SIDEWAYS / UP", "Direction output",
             "<p>Probabilities of DOWN, SIDEWAYS and UP at four horizons, plus a call. A call is shown only for horizons where the locked test supported it (all four direction horizons did).</p>", "4 horizons"),
        node("Risk numbers", "volatility and drawdown", "Risk output",
             "<p>Expected annual volatility and the chance of a fall of more than 10 %. The drawdown probability ranks risk well but its level is not better than a constant guess, so read it as relative risk.</p>", "4 + 4"),
        node("Locked test", "2024 to 2026, opened once", "How the study was judged",
             "<p>The test block was sealed with a hashed freeze and opened once, after the design was fixed. Seven hypotheses were pre-registered; later analyses are labelled post-hoc.</p>"
             + _ul("Holm correction for many tests", "Bootstrap confidence intervals", "Strong numbers-only benchmarks, not just chance"), "H1 to H11"),
        node("Findings", "small edge, honest negatives", "What the research found",
             "<p>A small but real direction signal at 5 and 10 sessions. Simple numbers beat charts and text on volatility and drawdown. Most of the text signal is company and calendar "
             "shortcuts, but filing days do move prices about 2.1 times as much, and the words help explain the size of the reaction.</p>", "result"),
    ], "right")
    return '<div class="ar-flow">' + l1 + l2 + l3 + l4 + l5 + "</div>"


# ------------------------------------------------------------------ 2. live pipeline
def live_html(B):
    l1 = lane(1, "You provide", "inputs on the Analyse page", [
        node("Screenshots", "1W, 1D, 1H (optional)", "Chart screenshots",
             "<p>Three TradingView screenshots (weekly, daily, hourly). Paste with Ctrl+V, drag in, or upload. Turn on the Volume indicator for the closest match to how the models were trained.</p>", "optional"),
        node("Ticker", "or company name", "Which stock",
             "<p>Typed, or read from the screenshot header. A company name such as <i>nvidia</i> is matched to its ticker.</p>"),
        node("Filing summary", "10-K, 10-Q or 8-K", "Optional text",
             "<p>A short summary of the latest filing. It switches on the six language models; without it the answer comes from the charts only.</p>", "optional"),
    ], "left")
    l2 = lane(2, "Reading", "no market download needed", [
        node("Price scale", "from the axis numbers", "Reading the prices",
             "<p>An OCR network reads the numbers on the price axis; a straight line (or a log curve) is fitted from pixel height to price. The O/H/L/C legend at the top is used as a check.</p>"
             + _ul("Tested on 66 real screenshots: median error 0.08 %"), "OCR + fit"),
        node("Candles and volume", "found by colour", "Finding every candle",
             "<p>Candle bodies and wicks are found by their colours; open, high, low and close come from where each body and wick starts and ends. Indicators, price tags and markers are ignored.</p>", "pixel reader"),
        node("Dates", "from the time axis", "Dating the candles",
             "<p>Month labels on the time axis pin the date of each daily candle (NYSE trading calendar). The last candle can never be later than the latest completed session.</p>"),
    ], "left")
    l3 = lane(3, "Predicting", "the networks run", [
        node("Redraw", "training format", "Redrawing the charts",
             "<p>The candles read from your picture are redrawn as the three 224 x 224 charts the networks were trained on. The networks never see your broker's colours.</p>", "3 images"),
        node("7 vision networks", "top 3 per target", "Chart networks",
             "<p>All seven run; for each of the 12 targets only the 3 best on validation data are averaged.</p>", "7 networks"),
        node("6 language models", "if text was given", "Text models",
             "<p>Only when you paste a filing summary. Run one at a time on a normal PC to save memory; results are the same.</p>", "6 models"),
        node("Blend and calibrate", "one answer per target", "Final numbers",
             "<p>The chart average and the text average are blended with a weight per target, then calibrated so stated probabilities are honest.</p>"),
    ], "mid")
    l4 = lane(4, "Result page", "what appears", [
        node("Outlook", "calls and probabilities", "Direction and risk",
             "<p>DOWN / SIDEWAYS / UP at 5, 10, 20 and 40 sessions, expected volatility and chance of a fall of more than 10 %, with a badge showing whether the locked test supported each.</p>"),
        node("Price chart", "your candles + cone", "Price chart and Monte Carlo cone",
             "<p>The candles read from your screenshot, joined to 4,000 simulated 20-session price paths. Candlestick patterns are marked for context only; the models do not use them.</p>"),
        node("Flowchart", "which network did what", "How this result was made",
             "<p>The expander on the result page shows which networks were used for the target you pick and what each of them said for this stock.</p>"),
        node("Ask AI", "Google Gemini", "Plain-language explanation",
             "<p>A chat that explains your numbers and the study. It answers only from the study's facts and your result, and it is told not to give investment advice.</p>"),
    ], "right")
    return '<div class="ar-flow ar-live">' + l1 + l2 + l3 + l4 + "</div>"


# ------------------------------------------------------------------ 3. what it predicts
def predicts_html(B):
    sk = {s["task"]: s for s in B.get("deploy_skill", [])}
    rows = ""
    for k in ("dir", "vol", "dd"):
        name, desc, metric = TARGET_INFO[k]
        cells = ""
        for h in (5, 10, 20, 40):
            s = sk.get(f"{k}_{h}")
            if s:
                ok = bool(s.get("supported"))
                verdict = (f'<p><b>Locked test:</b> skill {s["skill"]:+.3f} (95 % interval {s["lo"]:+.3f} to {s["hi"]:+.3f}), Holm p = {s["p_holm"]:.3f}. '
                           + ("<b>An edge over the no-skill benchmark was supported.</b>" if ok else "<b>No edge over the no-skill benchmark was proven</b>, so treat this number with caution.") + "</p>")
            else:
                ok, verdict = False, ""
            cells += node(f"{h} sessions", HZ[h], f"{name} at {h} sessions ({HZ[h]})", f"<p>{esc(desc)}</p><p>Measured by: {esc(metric)}.</p>{verdict}",
                          "", "ar-ok" if ok else "ar-no")
        rows += f'<div class="ar-prow"><div class="ar-ph"><div class="ar-lt">{name}</div><div class="ar-ls">{esc(metric)}</div></div>{cells}</div>'
    out = ('<div class="ar-pred">' + rows + "</div>"
           '<div class="nx-note">Green means the locked 2024 to 2026 test showed the dashboard recipe beating the no-skill benchmark for that target. '
           'It does not mean it beat the simple numbers-only models: on volatility and drawdown those still win. Hover a box for the definition and the numbers.</div>')
    return out
