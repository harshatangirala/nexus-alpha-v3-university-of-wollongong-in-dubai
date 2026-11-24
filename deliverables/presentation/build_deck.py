"""Build the NEXUS Alpha v3 project presentation (35 slides, stock-market theme).

Every number comes from results/tables or the two reports in deliverables/. Charts are native PowerPoint charts, flow charts are
native shapes, and the speaker notes carry a plain-spoken talk track.
Run:  python build_deck.py   ->  NEXUS_Alpha_v3_Presentation.pptx
"""
import json
import os
import random

from lxml import etree
from PIL import Image, ImageFont
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE, XL_TICK_LABEL_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FIG = os.path.join(ROOT, "deliverables", "figures")
EXA = os.path.join(ROOT, "lite", "examples")
OUT = os.path.join(HERE, "NEXUS_Alpha_v3_Presentation.pptx")

REPO = "github.com/harshatangirala/nexus-alpha-v3-university-of-wollongong-in-dubai"
FULL = "buffalo-deposits-graph-legends.trycloudflare.com"
LITE = "nexus-alpha-v3-lite.streamlit.app"

# ------------------------------------------------------------------ palette (a trading terminal at night)
BG, PANEL, PANEL2, LINE = "0B1220", "121C2F", "1A2740", "2A3A58"
TEXT, MUTED = "E8EEF8", "9AA9BF"
UP, DOWN, AMBER, CYAN, VIOLET = "22C55E", "EF4444", "F5B942", "38BDF8", "A78BFA"
FONT = "Calibri"
W, H = 13.333, 7.5
MX = 0.6


def rgb(h):
    return RGBColor.from_string(h)


_F = {}


def _font(bold, pt):
    k = (bold, pt)
    if k not in _F:
        _F[k] = ImageFont.truetype(r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf", int(pt * 10))
    return _F[k]


WARN = []


def fit_lines(text, pt, width_in, bold=False):
    f = _font(bold, pt)
    maxw = width_in * 72 * 10
    lines = 0
    for para in text.split("\n"):
        cur, n = "", 1
        for wd in para.split(" "):
            t = (cur + " " + wd).strip()
            if f.getlength(t) <= maxw or not cur:
                cur = t
            else:
                n += 1
                cur = wd
        lines += n
    return lines


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
BLANK = prs.slide_layouts[6]
SLIDE_NO = [0]


def new_slide(notes=""):
    s = prs.slides.add_slide(BLANK)
    SLIDE_NO[0] += 1
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(BG)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def rect(s, x, y, w, h, fill=PANEL, line=None, rounded=True, radius=0.06, shape=None, name=None):
    shp = s.shapes.add_shape(shape or (MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE), Inches(x), Inches(y), Inches(w), Inches(h))
    if rounded and shape is None:
        shp.adjustments[0] = radius
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(fill)
    if line:
        shp.line.color.rgb = rgb(line)
        shp.line.width = Pt(1)
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    if name:
        shp.name = name
    return shp


def _bullet(p, color):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", "228600")
    pPr.set("indent", "-228600")
    for tag in ("a:buNone", "a:buChar", "a:buClr"):
        for e in pPr.findall("{http://schemas.openxmlformats.org/drawingml/2006/main}" + tag.split(":")[1]):
            pPr.remove(e)
    ns = "http://schemas.openxmlformats.org/drawingml/2006/main"
    clr = etree.SubElement(pPr, f"{{{ns}}}buClr")
    etree.SubElement(clr, f"{{{ns}}}srgbClr").set("val", color)
    etree.SubElement(pPr, f"{{{ns}}}buChar").set("char", "•")


def tb(s, x, y, w, h, paras, size=16, color=TEXT, bold=False, align="l", anchor="t", bullets=False, space=6, italic=False, check=True,
       bcolor=None):
    """paras: str | list of (str | list of (text, opts)). Returns the shape. Warns when the text will not fit."""
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.04)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    if isinstance(paras, str):
        paras = [paras]
    total_h = 0.0
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
        p.space_after = Pt(space if i < len(paras) - 1 else 0)
        runs = [(para, {})] if isinstance(para, str) else para
        plain, psize, pbold = "", size, bold
        for text, o in runs:
            r = p.add_run()
            r.text = text
            f = r.font
            f.name = FONT
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", italic)
            f.color.rgb = rgb(o.get("color", color))
            plain += text
            psize = max(psize, o.get("size", size)) if "size" in o else psize
            pbold = pbold or o.get("bold", False)
        if bullets:
            _bullet(p, bcolor or CYAN)
        wi = w - 0.08 - (0.25 if bullets else 0)
        total_h += fit_lines(plain, psize, wi, pbold) * psize * 1.2 / 72 + (space / 72 if i < len(paras) - 1 else 0)
    if check and total_h > h + 0.02:
        WARN.append(f"slide {SLIDE_NO[0]}: text needs {total_h:.2f}in but box is {h:.2f}in: {str(paras)[:70]}")
    return box


def chip(s, x, y, text, color=CYAN, w=None):
    w = w or (0.28 + 0.095 * len(text))
    r = rect(s, x, y, w, 0.3, fill=PANEL2, radius=0.5)
    tb(s, x, y, w, 0.3, text, size=11, color=color, bold=True, align="c", anchor="m", check=False)
    return w


def header(s, title, section, color=CYAN, size=30):
    chip(s, MX, 0.32, section.upper(), color)
    tb(s, MX, 0.68, W - 2 * MX, 0.95, title, size=size, bold=True, anchor="t")
    n = fit_lines(title, size, W - 2 * MX - 0.08, True)
    if n > 2:
        WARN.append(f"slide {SLIDE_NO[0]}: title wraps to {n} lines")
    tb(s, W - MX - 3.5, 7.08, 3.5, 0.28, f"{SLIDE_NO[0]}", size=10, color=MUTED, align="r", check=False)
    tb(s, MX, 7.08, 6, 0.28, "NEXUS Alpha v3  ·  University of Wollongong in Dubai", size=10, color=MUTED, check=False)


def arrow(s, x1, y1, x2, y2, color=MUTED, width=2.0):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    ns = "http://schemas.openxmlformats.org/drawingml/2006/main"
    te = etree.SubElement(ln, f"{{{ns}}}tailEnd")
    te.set("type", "triangle")
    te.set("w", "med")
    te.set("len", "med")
    return c


def node(s, x, y, w, h, title, sub="", color=CYAN, fill=PANEL, tsize=15, ssize=12):
    rect(s, x, y, w, h, fill=fill, line=color, radius=0.12)
    if sub:
        tb(s, x + 0.08, y + 0.06, w - 0.16, h * 0.42, title, size=tsize, bold=True, color=color, align="c", anchor="m")
        tb(s, x + 0.08, y + h * 0.46, w - 0.16, h * 0.5, sub, size=ssize, color=TEXT, align="c", anchor="m")
    else:
        tb(s, x + 0.08, y + 0.04, w - 0.16, h - 0.08, title, size=tsize, bold=True, color=color, align="c", anchor="m")


def stat(s, x, y, w, h, big, label, color=CYAN, bsize=36, lsize=13):
    rect(s, x, y, w, h, fill=PANEL, line=LINE)
    tb(s, x + 0.1, y + 0.12, w - 0.2, h * 0.5, big, size=bsize, bold=True, color=color, align="c", anchor="m")
    tb(s, x + 0.12, y + h * 0.58, w - 0.24, h * 0.38, label, size=lsize, color=MUTED, align="c", anchor="t")


def card(s, x, y, w, h, title, body, color=CYAN, tsize=17, bsize=14):
    rect(s, x, y, w, h, fill=PANEL, line=LINE)
    tb(s, x + 0.2, y + 0.15, w - 0.4, 0.5, title, size=tsize, bold=True, color=color)
    tb(s, x + 0.2, y + 0.68, w - 0.4, h - 0.8, body, size=bsize, color=TEXT)


def picture(s, path, x, y, w, h):
    """Place an image inside the box (x, y, w, h), keeping its aspect ratio and centring it."""
    im = Image.open(path)
    ar = im.width / im.height
    pw, ph = (w, w / ar) if w / h < ar else (h * ar, h)
    s.shapes.add_picture(path, Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


# ------------------------------------------------------------------ charts
def _axis_style(chart, size=12, grid=True, vfmt=None, vmin=None, vmax=None, major=None):
    chart.font.size = Pt(size)
    chart.font.name = FONT
    chart.font.color.rgb = rgb(MUTED)
    chart.has_title = False
    va, ca = chart.value_axis, chart.category_axis
    va.has_major_gridlines = grid
    if grid:
        va.major_gridlines.format.line.color.rgb = rgb(LINE)
        va.major_gridlines.format.line.width = Pt(0.75)
    va.format.line.fill.background()
    va.tick_labels.font.color.rgb = rgb(MUTED)
    va.tick_labels.font.size = Pt(size)
    if vfmt:
        va.tick_labels.number_format = vfmt
        va.tick_labels.number_format_is_linked = False
    if vmin is not None:
        va.minimum_scale = vmin
    if vmax is not None:
        va.maximum_scale = vmax
    if major:
        va.major_unit = major
    ca.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    ca.format.line.color.rgb = rgb(LINE)
    ca.tick_labels.font.color.rgb = rgb(TEXT)
    ca.tick_labels.font.size = Pt(size)
    ca.has_major_gridlines = False


def _legend(chart, on, size=12):
    chart.has_legend = on
    if on:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(size)
        chart.legend.font.color.rgb = rgb(TEXT)


def bar_chart(s, x, y, w, h, cats, series, colors, kind="col", fmt="0.00", legend=None, labels=True, point_colors=None, size=12,
              vmin=None, vmax=None, stacked=False, gap=60, vfmt=None, label_pos=None, label_size=None, major=None):
    """series: {name: values}. colors: one hex per series. point_colors: hex per category (single series only)."""
    cd = CategoryChartData()
    cd.categories = cats
    for n, v in series.items():
        cd.add_series(n, v)
    t = {("col", False): XL_CHART_TYPE.COLUMN_CLUSTERED, ("bar", False): XL_CHART_TYPE.BAR_CLUSTERED,
         ("col", True): XL_CHART_TYPE.COLUMN_STACKED, ("bar", True): XL_CHART_TYPE.BAR_STACKED}[(kind, stacked)]
    gf = s.shapes.add_chart(t, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    _axis_style(ch, size, vfmt=vfmt or fmt, vmin=vmin, vmax=vmax, major=major)
    _legend(ch, legend if legend is not None else len(series) > 1, size)
    plot = ch.plots[0]
    plot.gap_width = gap
    if not stacked and len(series) > 1:
        plot.overlap = -10
    if stacked:
        plot.overlap = 100
    if kind == "bar":
        ch.category_axis.reverse_order = True
    for ser, c in zip(plot.series, colors):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = rgb(c)
        ser.invert_if_negative = False
    if point_colors:
        for i, c in enumerate(point_colors):
            pt = plot.series[0].points[i]
            pt.format.fill.solid()
            pt.format.fill.fore_color.rgb = rgb(c)
        for dpt in plot.series[0]._element.findall('{http://schemas.openxmlformats.org/drawingml/2006/chart}dPt'):
            if dpt.find('{http://schemas.openxmlformats.org/drawingml/2006/chart}invertIfNegative') is None:
                inv = etree.Element('{http://schemas.openxmlformats.org/drawingml/2006/chart}invertIfNegative')
                inv.set('val', '0')
                dpt.find('{http://schemas.openxmlformats.org/drawingml/2006/chart}idx').addnext(inv)
    if kind == 'bar' and labels:
        ch.value_axis.visible = False
    if labels:
        plot.has_data_labels = True
        dl = plot.data_labels
        dl.number_format = fmt
        dl.number_format_is_linked = False
        dl.font.size = Pt(label_size or max(size - 1, 10))
        dl.font.color.rgb = rgb(TEXT)
        dl.position = label_pos or (XL_LABEL_POSITION.CENTER if stacked else XL_LABEL_POSITION.OUTSIDE_END)
    return ch


def line_chart(s, x, y, w, h, cats, series, colors, fmt="0.00", size=12, vmin=None, vmax=None, legend=True):
    cd = CategoryChartData()
    cd.categories = cats
    for n, v in series.items():
        cd.add_series(n, v)
    gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    _axis_style(ch, size, vfmt=fmt, vmin=vmin, vmax=vmax)
    _legend(ch, legend, size)
    for ser, c in zip(ch.plots[0].series, colors):
        ser.format.line.color.rgb = rgb(c)
        ser.format.line.width = Pt(2.75)
        ser.smooth = False
        ser.marker.style = XL_MARKER_STYLE.CIRCLE
        ser.marker.size = 8
        ser.marker.format.fill.solid()
        ser.marker.format.fill.fore_color.rgb = rgb(c)
        ser.marker.format.line.color.rgb = rgb(c)
    return ch


def table(s, x, y, w, col_w, rows, header_color=CYAN, size=13, row_h=0.42, align=None, hl_col=None):
    """rows[0] is the header. col_w: inches per column. align: list of 'l'/'c'."""
    nr, nc = len(rows), len(rows[0])
    gf = s.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    tbl = gf.table
    tblPr = tbl._tbl.tblPr
    tblPr.set("bandRow", "0")
    tblPr.set("firstRow", "0")
    for e in tblPr.findall("{http://schemas.openxmlformats.org/drawingml/2006/main}tableStyleId"):
        tblPr.remove(e)
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = rgb(PANEL2 if i == 0 else (PANEL if i % 2 else BG))
            c.margin_left = c.margin_right = Inches(0.1)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if (align and align[j] == "c") else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = str(val)
            r.font.name = FONT
            r.font.size = Pt(size)
            r.font.bold = i == 0 or (hl_col is not None and j == hl_col)
            r.font.color.rgb = rgb(header_color if i == 0 else TEXT)
            if i > 0 and fit_lines(str(val), size, col_w[j] - 0.2, r.font.bold) * size * 1.2 / 72 + 0.06 > row_h:
                WARN.append(f"slide {SLIDE_NO[0]}: table cell wraps: {str(val)[:40]}")
    return tbl


def candles(s, x, y, w, h, n=34, seed=3, thick=0.55):
    """A random-walk candlestick motif, drawn from native shapes."""
    rnd = random.Random(seed)
    p, pts = 0.0, []
    for _ in range(n):
        o = p
        c = o + rnd.gauss(0.05, 0.9)
        hi, lo = max(o, c) + abs(rnd.gauss(0, 0.45)), min(o, c) - abs(rnd.gauss(0, 0.45))
        pts.append((o, c, hi, lo))
        p = c
    top, bot = max(q[2] for q in pts), min(q[3] for q in pts)
    sc = h / (top - bot)
    step = w / n
    for i, (o, c, hi, lo) in enumerate(pts):
        col = UP if c >= o else DOWN
        cx = x + i * step + step / 2
        yh, yl = y + (top - hi) * sc, y + (top - lo) * sc
        yo, yc = y + (top - max(o, c)) * sc, y + (top - min(o, c)) * sc
        rect(s, cx - 0.008, yh, 0.016, max(yl - yh, 0.02), fill=col, rounded=False)
        rect(s, cx - step * thick / 2, yo, step * thick, max(yc - yo, 0.03), fill=col, rounded=False)


# ================================================================== SLIDES
ex = {}
for t in ("NVDA", "V", "F", "BAC"):
    with open(os.path.join(EXA, f"{t}.json"), encoding="utf-8") as f:
        ex[t] = json.load(f)


def r20(t):
    return [a for a in ex[t]["table"] if a["horizon (sessions)"] == 20][0]


# ---- 1 title
s = new_slide("Hi everyone. We are a group of five from the Master of Applied AI, and this is our final project for Deep Learning for Visual Data and for NLP. "
              "The short version of the question: can AI really read a candlestick chart or a company filing and tell you something useful about a stock? "
              "We built a careful test to find out, and the honest answer is more interesting than a plain yes or no.")
candles(s, 6.9, 0.9, 6.0, 3.6, n=36, seed=11)
tb(s, MX, 1.0, 6.4, 0.4, "MAAI 901 DEEP LEARNING FOR VISUAL DATA  ·  MAAI 902 NLP AND LLMS", size=12, color=AMBER, bold=True)
tb(s, MX, 1.55, 6.3, 2.4, "Can AI read the stock market?", size=50, bold=True)
tb(s, MX, 3.55, 6.1, 1.2, "Candlestick charts and SEC filings, read by seven vision networks and seven text models, tested on a locked 2024 to 2026 window",
   size=19, color=MUTED)
tb(s, MX, 4.95, 12, 0.4, [[("Krishna Harsha Tangirala (lead)", {"bold": True, "color": TEXT}),
                          ("   ·   Mohammed Shahrukh   ·   Valavala Divya Anu Sri   ·   Sreeya Tirumalasetti   ·   Venkata Saisravan Pasam", {})]],
   size=14, color=MUTED)
tb(s, MX, 5.35, 12, 0.35, "Master of Applied Artificial Intelligence, University of Wollongong in Dubai", size=14, color=MUTED)
rect(s, 0, 6.2, W, 0.62, fill=PANEL, rounded=False)
tape = [("NVDA", "▲ 52% up", UP), ("V", "▼ 39% down", DOWN), ("F", "◆ sideways", AMBER), ("BAC", "▲ 43% up", UP), ("50 stocks", "5 sectors", CYAN),
        ("10,269", "filings", VIOLET), ("193,650", "chart images", CYAN)]
xx = 0.5
for a, b, c in tape:
    tb(s, xx, 6.28, 1.9, 0.46, [[(a + "  ", {"bold": True, "color": TEXT}), (b, {"color": c, "bold": True})]], size=13, anchor="m", check=False)
    xx += 1.78
tb(s, MX, 6.95, 9, 0.3, "Outlook for 20 trading sessions from our dashboard, 23 Nov 2026. Educational research, not financial advice.", size=10, color=MUTED, check=False)

# ---- 2 team and agenda
s = new_slide("Quick orientation. On the left is the team, and I led the project and wrote most of the pipeline. On the right is how the next half hour is organised. "
              "We follow the marking guides of both subjects, so you will see the problem, the data, the models, the results, the ethics and a live demo, in that order.")
header(s, "Who we are and what this talk covers", "Welcome", AMBER)
people = [("Krishna Harsha Tangirala", "Lead author"), ("Mohammed Shahrukh", ""), ("Valavala Divya Anu Sri", ""), ("Sreeya Tirumalasetti", ""),
          ("Venkata Saisravan Pasam", "")]
for i, (n, r) in enumerate(people):
    yy = 1.75 + i * 0.98
    rect(s, MX, yy, 5.2, 0.84, fill=PANEL, line=AMBER if r else LINE)
    rect(s, MX + 0.18, yy + 0.17, 0.5, 0.5, fill=PANEL2, shape=MSO_SHAPE.OVAL)
    tb(s, MX + 0.18, yy + 0.17, 0.5, 0.5, n.split()[0][0], size=18, bold=True, color=AMBER if r else CYAN, align="c", anchor="m", check=False)
    tb(s, MX + 0.85, yy + 0.08, 4.2, 0.4, n, size=17, bold=True, anchor="m")
    tb(s, MX + 0.85, yy + 0.45, 4.2, 0.3, r if r else "Group member", size=12, color=AMBER if r else MUTED, anchor="m")
agenda = [("1", "The problem", "What we tested and why people should be sceptical", AMBER), ("2", "Vision: charts as images", "Data, pipeline, seven networks, results", CYAN),
          ("3", "Language: SEC filings", "Corpus, models, fine-tuning, event study", VIOLET), ("4", "Putting them together", "Fusion, dashboard, live demo", UP),
          ("5", "Doing it responsibly", "Ethics, limits, what comes next", DOWN)]
for i, (n, t, d, c) in enumerate(agenda):
    yy = 1.75 + i * 0.98
    rect(s, 6.4, yy, 6.3, 0.84, fill=PANEL, line=LINE)
    rect(s, 6.58, yy + 0.17, 0.5, 0.5, fill=c, shape=MSO_SHAPE.OVAL)
    tb(s, 6.58, yy + 0.17, 0.5, 0.5, n, size=18, bold=True, color=BG, align="c", anchor="m", check=False)
    tb(s, 7.25, yy + 0.07, 5.3, 0.4, t, size=17, bold=True, anchor="m")
    tb(s, 7.25, yy + 0.45, 5.3, 0.3, d, size=13, color=MUTED, anchor="m")

# ---- 3 problem
s = new_slide("Open any trading app today and you will see 'AI pattern recognition'. Some products also say a language model reads the filings for you. "
              "Those are big claims, and almost nobody shows proof on data the model has never seen. That gap is our project.")
header(s, "Everyone sells an AI that reads charts and filings. Few show proof", "The problem", AMBER)
cl = [("Claim 1", "Charts are predictable", "Retail investors stare at candlesticks every day, and platforms now sell automatic pattern detection as an AI feature.", CYAN),
      ("Claim 2", "Filings contain the next move", "10-Ks, 10-Qs and 8-Ks are public within minutes. Products say a language model can turn the words into a signal.", VIOLET),
      ("Claim 3", "More models means more edge", "Newer networks and bigger language models are supposed to read both sources better than simple numbers.", UP)]
for i, (a, b, c, col) in enumerate(cl):
    xx = MX + i * 4.1
    rect(s, xx, 1.8, 3.9, 3.1, fill=PANEL, line=LINE)
    tb(s, xx + 0.25, 1.95, 3.4, 0.35, a.upper(), size=12, bold=True, color=col)
    tb(s, xx + 0.25, 2.3, 3.4, 0.9, b, size=21, bold=True)
    tb(s, xx + 0.25, 3.3, 3.4, 1.5, c, size=16, color=MUTED)
rect(s, MX, 5.15, W - 2 * MX, 1.55, fill=PANEL2)
tb(s, MX + 0.3, 5.28, 11.5, 0.4, "WHAT WE ASKED", size=12, bold=True, color=AMBER)
tb(s, MX + 0.3, 5.62, 11.6, 1.0, "Do these claims survive a fair test? Same stocks, same targets, same time split, and a final test window that nobody touched until every "
   "choice was frozen. If charts or filings help, we want to know where, by how much, and compared with plain numbers.", size=16)

# ---- 4 traps
s = new_slide("There are three classic ways to fool yourself in this field. First, leaking the future into training. Second, using a language model that already read the news you are testing on. "
              "Third, letting the model learn which company it is looking at instead of what the filing says. We designed around each one.")
header(s, "Three ways a stock model fools you, and how we closed each one", "The problem", AMBER)
traps = [("Leakage", "Label windows from the training period overlap the test period, so the model has already seen the answer.",
          "40-session gap between every block. Test labels locked in a hashed vault.", DOWN),
         ("Look-ahead", "A language model that read the news during pre-training can seem to forecast it.",
          "Test window starts after every model's pre-training cut-off. All filing times are point-in-time.", AMBER),
         ("Shortcuts", "A filing reveals the company and the earnings calendar. Both predict volatility without reading a word.",
          "Per-company and calendar baselines on the same filings. Words must beat them.", VIOLET)]
for i, (t, p, f, col) in enumerate(traps):
    xx = MX + i * 4.1
    rect(s, xx, 1.8, 3.9, 4.9, fill=PANEL, line=LINE)
    rect(s, xx + 0.25, 2.0, 0.55, 0.55, fill=col, shape=MSO_SHAPE.OVAL)
    tb(s, xx + 0.25, 2.0, 0.55, 0.55, str(i + 1), size=20, bold=True, color=BG, align="c", anchor="m", check=False)
    tb(s, xx + 0.95, 2.0, 2.8, 0.55, t, size=22, bold=True, anchor="m")
    tb(s, xx + 0.25, 2.8, 3.4, 1.7, [[("The trap  ", {"bold": True, "color": col}), (p, {})]], size=16)
    rect(s, xx + 0.2, 4.6, 3.5, 1.9, fill=PANEL2)
    tb(s, xx + 0.35, 4.7, 3.2, 1.7, [[("Our fix  ", {"bold": True, "color": UP}), (f, {})]], size=16)

# ---- 5 research questions
s = new_slide("Four research questions drive everything. I will show the short answers right here so you know where we are heading, then spend the rest of the talk on the evidence.")
header(s, "Four research questions, and the short answers", "The problem", AMBER)
rq = [("RQ1", "Do vision networks that read candlestick charts beat uninformed benchmarks and simple numbers?", "Modestly yes on direction. Numbers still win on risk.", CYAN),
      ("RQ2", "Do filings move prices, and can language models forecast from the text once company and calendar are controlled for?", "Filings clearly move prices. Forecasting from text alone, mostly no.", VIOLET),
      ("RQ3", "Does combining charts and text beat the better of the two alone?", "Yes on volatility and drawdown risk (6 of 12 tests).", UP),
      ("RQ4", "Can we serve it in a usable tool that explains itself without overclaiming?", "Yes: a Streamlit dashboard with a grounded AI assistant.", AMBER)]
for i, (a, q, ans, col) in enumerate(rq):
    yy = 1.75 + i * 1.28
    rect(s, MX, yy, W - 2 * MX, 1.12, fill=PANEL, line=LINE)
    rect(s, MX + 0.2, yy + 0.22, 0.95, 0.68, fill=col, radius=0.2)
    tb(s, MX + 0.2, yy + 0.22, 0.95, 0.68, a, size=20, bold=True, color=BG, align="c", anchor="m", check=False)
    tb(s, MX + 1.35, yy + 0.08, 6.0, 0.96, q, size=15, anchor="m")
    rect(s, 8.15, yy + 0.14, 0.02, 0.84, fill=LINE, rounded=False)
    tb(s, 8.35, yy + 0.08, 4.2, 0.96, ans, size=15, bold=True, color=col, anchor="m")

# ---- 6 at a glance
s = new_slide("Here is the project in numbers. Fifty big US stocks across five sectors, about ten years of data, more than sixty thousand decision days, nearly two hundred thousand chart images and ten thousand filings. "
              "Fourteen models in total, twelve targets, and one locked test window that we opened exactly once.")
header(s, "The project at a glance", "Overview", AMBER)
tiles = [("50", "large US stocks, 10 in each of 5 sectors", CYAN), ("64,550", "decision days (stock x date rows)", CYAN),
         ("193,650", "chart images, 3 per row (weekly, daily, hourly)", CYAN), ("10,269", "SEC filings: 554 10-K, 1,700 10-Q, 8,015 8-K", VIOLET),
         ("7 + 7", "vision networks + text models", UP), ("6,550", "rows in the locked test, opened once", AMBER)]
for i, (b, l, c) in enumerate(tiles):
    xx = MX + (i % 3) * 4.1
    yy = 1.8 + (i // 3) * 1.95
    stat(s, xx, yy, 3.9, 1.75, b, l, c, bsize=40)
rect(s, MX, 5.8, W - 2 * MX, 1.0, fill=PANEL2)
tb(s, MX + 0.3, 5.88, 3, 0.3, "WHAT WE PREDICT", size=12, bold=True, color=AMBER)
tgt = [("Direction", "DOWN / SIDEWAYS / UP", UP), ("Volatility", "how jumpy the price will be", CYAN), ("Drawdown", "a fall of more than 10%", DOWN), ("Horizons", "5, 10, 20 and 40 sessions", AMBER)]
for i, (a, b, c) in enumerate(tgt):
    tb(s, MX + 0.3 + i * 3.0, 6.2, 2.9, 0.55, [[(a + "  ", {"bold": True, "color": c}), (b, {"color": TEXT})]], size=13, anchor="m")

# ---- 7 architecture
s = new_slide("This is the whole system on one slide. The top lane turns prices into chart images for seven vision networks. The bottom lane turns SEC filings into text for seven language models. "
              "Each side is calibrated, the best three models per target are averaged, the two sides are blended, and the result feeds the dashboard and a Monte Carlo price cone.")
header(s, "The whole system, from raw data to the dashboard", "Architecture", UP)
bx = [0.6, 3.25, 5.9]
rect(s, 0.45, 1.75, 8.05, 1.85, fill=PANEL2, radius=0.04)
rect(s, 0.45, 3.85, 8.05, 1.85, fill=PANEL2, radius=0.04)
tb(s, 0.6, 1.78, 3, 0.3, "VISION LANE", size=11, bold=True, color=CYAN, check=False)
tb(s, 0.6, 3.88, 3, 0.3, "LANGUAGE LANE", size=11, bold=True, color=VIOLET, check=False)
v = [("Price bars", "weekly, daily, hourly\n50 stocks, 2016 to 2026"), ("Redraw charts", "3 images of 224 x 224\n52 / 63 / 70 candles"), ("7 vision networks", "CNN, ResNet, EffNet, ConvNeXt, ViT, Swin, YOLOv8")]
n_ = [("SEC EDGAR", "10-K, 10-Q, 8-K\n10,269 point-in-time"), ("Clean text", "MD&A or 8-K body\n[form][section] tag"), ("7 text models", "TF-IDF, FinBERT, SEC-BERT, RoBERTa, DeBERTa, Longformer, FinGPT")]
for i in range(3):
    node(s, bx[i], 2.12, 2.3, 1.35, v[i][0], v[i][1], CYAN, tsize=15, ssize=12)
    node(s, bx[i], 4.22, 2.3, 1.35, n_[i][0], n_[i][1], VIOLET, tsize=15, ssize=12)
    if i < 2:
        arrow(s, bx[i] + 2.3, 2.8, bx[i + 1], 2.8, MUTED)
        arrow(s, bx[i] + 2.3, 4.9, bx[i + 1], 4.9, MUTED)
node(s, 9.05, 3.0, 1.75, 1.75, "Top 3 per target", "chosen on 2022 to 2023, calibrated, blended", AMBER, tsize=14, ssize=12)
arrow(s, 8.2, 2.8, 9.05, 3.55, MUTED)
arrow(s, 8.2, 4.9, 9.05, 4.2, MUTED)
node(s, 11.15, 1.9, 1.6, 1.5, "12 outputs", "direction, volatility, drawdown x 4 horizons", UP, tsize=14, ssize=12)
node(s, 11.15, 3.6, 1.6, 1.3, "Price cone", "Monte Carlo, 4,000 paths", UP, tsize=14, ssize=12)
node(s, 11.15, 5.1, 1.6, 1.1, "Ask AI", "grounded Gemini", UP, tsize=14, ssize=12)
arrow(s, 10.8, 3.7, 11.15, 2.9, MUTED)
arrow(s, 10.8, 3.9, 11.15, 4.25, MUTED)
arrow(s, 10.8, 4.2, 11.15, 5.6, MUTED)
tb(s, 0.6, 5.95, 10.2, 0.95, "One shared multi-task head per network: for each horizon, three direction logits, one volatility value and one drawdown logit. "
   "The test window (Jan 2024 to Aug 2026) was sealed with a hashed freeze before anyone looked at it.", size=14, color=MUTED)

# ---- 8 data and splits
s = new_slide("We split by time, never randomly, because random splits let overlapping label windows leak. Train on 2016 to 2020, stop early on 2021, choose models on 2022 and 2023, and test on 2024 to August 2026. "
              "There is a forty-session gap between blocks. Notice the validation block contains the 2022 bear market and the test block is a strong rally.")
header(s, "Splitting by time, with a gap, and one locked test", "Computer vision · data", CYAN)
blocks = [("Train", "Oct 2016 to Dec 2020", 4.25, CYAN), ("Inner", "2021", 1.0, VIOLET), ("Validation", "2022 to 2023", 2.0, AMBER), ("Locked test", "Jan 2024 to Aug 2026", 2.67, DOWN)]
tot = sum(b[2] for b in blocks)
xx = MX
for n, d, wd, c in blocks:
    ww = (W - 2 * MX) * wd / tot
    rect(s, xx + 0.02, 1.8, ww - 0.04, 0.95, fill=c, rounded=False)
    tb(s, xx + 0.05, 1.82, ww - 0.1, 0.9, [[(n, {"bold": True, "size": 16})], [(d, {"size": 12})]], size=12, color=BG, align="c", anchor="m", space=0, check=False)
    xx += ww
tb(s, MX, 2.78, W - 2 * MX, 0.3, "A 40-session purge separates every pair of blocks so no label window crosses a boundary.", size=12, color=MUTED, check=False)
bar_chart(s, MX, 3.5, 6.6, 3.4, ["Train", "Inner (2021)", "Validation", "Locked test"], {"rows": [51250, 2150, 4600, 6550]}, [CYAN], fmt="#,##0",
          point_colors=[CYAN, VIOLET, AMBER, DOWN], legend=False, gap=50, vmin=0, vmax=60000)
tb(s, MX, 3.15, 6.6, 0.3, "Chart rows per block", size=12, bold=True, color=MUTED, check=False)
card(s, 7.5, 3.2, 5.2, 3.6, "Why this matters", [[("Universe. ", {"bold": True, "color": CYAN}), ("50 large US stocks, 10 per sector, so no sector dominates.", {})],
                                                 [("Bars. ", {"bold": True, "color": CYAN}), ("Built from a public 1-minute dataset, five split dates corrected, SHA-256 manifest.", {})],
                                                 [("Regime shift. ", {"bold": True, "color": CYAN}), ("Validation holds the 2022 bear market, the test a rally. Single winners can flip, so we average three.", {})]],
     CYAN, bsize=14)

# ---- 9 chart rendering
s = new_slide("Instead of using broker screenshots for training, we drew every chart ourselves with one renderer. Same colours, same candle counts, no text or axes. "
              "That gives byte-exact reruns, and it gives the networks a single clean format. You can see three training rows here, one per direction class.")
header(s, "Every chart is drawn the same way, three timeframes per decision", "Computer vision · data", CYAN)
picture(s, os.path.join(FIG, "chart_gallery.png"), MX, 1.75, 6.2, 5.2)
facts = [("3 views", "weekly (52 candles), daily (63) and hourly (70), each with a volume strip"), ("224 x 224", "pixels per image, five-colour palette, no axes, dates or text"),
         ("193,650", "images, stored as palette indices: 9.72 GB total"), ("Deterministic", "any row can be redrawn byte for byte from its bars")]
for i, (b, l) in enumerate(facts):
    yy = 1.8 + i * 1.28
    rect(s, 7.2, yy, 5.5, 1.14, fill=PANEL, line=LINE)
    tb(s, 7.35, yy + 0.05, 1.9, 1.04, b, size=20, bold=True, color=CYAN, anchor="m")
    tb(s, 9.3, yy + 0.05, 3.3, 1.04, l, size=13, anchor="m")
tb(s, 7.2, 6.95, 5.5, 0.1, "", size=8, check=False)

# ---- 10 preprocessing
s = new_slide("Preprocessing is a five-step pipeline. The augmentation story is worth a minute: we deliberately avoid flips, because flipping a chart horizontally reverses time, and flipping it vertically turns a rally into a crash. "
              "We only do things that keep a chart meaning what it meant.")
header(s, "Preprocessing and augmentation that respect what a chart means", "Computer vision · pipeline", CYAN)
steps = [("Raw bars", "1-minute data, split-adjusted"), ("Aggregate", "hourly, daily, weekly bars"), ("Render", "3 charts per decision"), ("Store", "8-bit palette indices"), ("On the GPU", "RGB tensors, ImageNet stats")]
for i, (a, b) in enumerate(steps):
    xx = MX + i * 2.45
    node(s, xx, 1.8, 2.1, 1.25, a, b, CYAN, tsize=15, ssize=12)
    if i < 4:
        arrow(s, xx + 2.1, 2.42, xx + 2.45, 2.42)
rect(s, MX, 3.45, 5.95, 3.35, fill=PANEL, line=UP)
tb(s, MX + 0.25, 3.55, 5.4, 0.4, "WHAT WE USE", size=13, bold=True, color=UP)
tb(s, MX + 0.25, 4.0, 5.5, 2.7, ["Small vertical rescale of the price range", "Crop away the oldest bars", "Brightness changes", "Gaussian noise"], size=19, bullets=True, bcolor=UP, space=10)
rect(s, 6.75, 3.45, 5.95, 3.35, fill=PANEL, line=DOWN)
tb(s, 7.0, 3.55, 5.4, 0.4, "WHAT WE AVOID, AND WHY", size=13, bold=True, color=DOWN)
tb(s, 7.0, 4.0, 5.5, 2.7, [[("Horizontal flip. ", {"bold": True}), ("It reverses time, so a rally becomes a decline.", {})],
                           [("Vertical flip. ", {"bold": True}), ("It turns a rally into a crash and flips the meaning of the label.", {})],
                           [("Heavy colour jitter. ", {"bold": True}), ("The five-colour palette carries the up and down signal.", {})]], size=17, bullets=True, bcolor=DOWN, space=12)

# ---- 11 labels
s = new_slide("There are three targets. Direction uses a sideways band that scales with each stock's own volatility, so a quiet stock and a wild one are judged fairly. "
              "Volatility is the log of realised volatility from hourly bars. Drawdown asks whether the price dips more than ten percent. The chart shows how class shares differ between train and validation.")
header(s, "Three targets, labelled against each stock's own volatility", "Computer vision · labels", CYAN)
lab = [("Direction", "DOWN, SIDEWAYS or UP. SIDEWAYS when the log return stays inside 0.5 x σ x √h, where σ is the stock's own trailing daily volatility.", UP),
       ("Volatility", "The log of realised volatility over the window, computed from hourly bars. Scored by R-squared against predicting the average.", CYAN),
       ("Drawdown", "1 if the price falls more than 10% below the decision-day close at any point in the window. Scored by ROC-AUC.", DOWN)]
for i, (a, b, c) in enumerate(lab):
    yy = 1.8 + i * 1.65
    rect(s, MX, yy, 6.4, 1.5, fill=PANEL, line=LINE)
    tb(s, MX + 0.2, yy + 0.08, 6.0, 0.4, a, size=18, bold=True, color=c)
    tb(s, MX + 0.2, yy + 0.5, 6.0, 0.95, b, size=14)
bar_chart(s, 7.2, 2.1, 5.5, 4.2, ["Train", "Validation"], {"DOWN": [22.3, 31.4], "SIDEWAYS": [40.1, 38.7], "UP": [37.6, 29.9]}, [DOWN, AMBER, UP], fmt='0.0"%"',
          stacked=True, kind="col", vmin=0, vmax=100, vfmt='0"%"', gap=55)
tb(s, 7.2, 1.75, 5.5, 0.35, "20-session label shares: validation holds the 2022 bear market", size=13, bold=True, color=MUTED, check=False)
tb(s, 7.2, 6.4, 5.5, 0.5, "Always guessing SIDEWAYS already scores about 42% accuracy, so we score direction with macro-F1.", size=12, color=MUTED)

# ---- 12 leakage control
s = new_slide("Leakage control mattered more than any modelling choice. Six safeguards, from the gap between time blocks to a vault that holds the test labels. "
              "Every time the vault is opened it is logged, and anything we analysed afterwards is labelled post-hoc in the reports.")
header(s, "Six safeguards against fooling ourselves", "Computer vision · rigour", CYAN)
sg = [("40-session purge", "No label window crosses a block boundary.", CYAN), ("Seven automated checks", "Hidden test labels, no repeated stock-date and more, before any training.", CYAN),
      ("Hashed freeze", "Every modelling choice registered and hashed before the test opened.", AMBER), ("Test vault", "Labels deposited once, every opening logged.", AMBER),
      ("Point-in-time filings", "Usable 35 minutes after EDGAR acceptance, entry at the next hourly bar.", VIOLET), ("Post-hoc labelling", "Anything added after opening the test is marked as exploratory.", DOWN)]
for i, (a, b, c) in enumerate(sg):
    xx = MX + (i % 3) * 4.1
    yy = 1.8 + (i // 3) * 2.5
    rect(s, xx, yy, 3.9, 2.3, fill=PANEL, line=LINE)
    rect(s, xx + 0.22, yy + 0.22, 0.5, 0.5, fill=c, shape=MSO_SHAPE.OVAL)
    tb(s, xx + 0.22, yy + 0.22, 0.5, 0.5, str(i + 1), size=18, bold=True, color=BG, align="c", anchor="m", check=False)
    tb(s, xx + 0.85, yy + 0.2, 2.9, 0.6, a, size=17, bold=True, anchor="m")
    tb(s, xx + 0.25, yy + 1.0, 3.4, 1.2, b, size=16, color=MUTED)

# ---- 13 seven networks
s = new_slide("We compare seven vision networks that span the last decade of ideas. A small CNN trained from scratch in the style of Jiang, Kelly and Xiu is the academic baseline. "
              "ResNet adds shortcuts, EfficientNet scales carefully, ConvNeXt modernises convolutions, ViT and Swin use attention, and YOLOv8 is a detection family we reuse as a classifier.")
header(s, "Seven vision networks, from a small CNN to Vision Transformers", "Computer vision · models", CYAN)
nets = ["Custom CNN", "ResNet-50", "EfficientNet-B0", "ConvNeXt-T", "ViT-B/16", "Swin-T", "YOLOv8s-cls"]
bar_chart(s, MX, 2.1, 6.0, 4.6, nets, {"Parameters (millions)": [3.39, 26.68, 5.99, 29.02, 86.99, 28.71, 7.07]}, [CYAN], kind="bar", fmt="0.0", legend=False, gap=45,
          vmin=0, vmax=125, vfmt="0")
tb(s, MX, 1.75, 6.0, 0.35, "Parameters (millions)", size=13, bold=True, color=MUTED, check=False)
bar_chart(s, 6.9, 2.1, 5.8, 4.6, nets, {"GFLOPs per row": [10.22, 24.53, 2.31, 26.73, 105.38, 26.94, 4.58]}, [AMBER], kind="bar", fmt="0.0", legend=False, gap=45,
          vmin=0, vmax=150, vfmt="0")
tb(s, 6.9, 1.75, 5.8, 0.35, "Compute per decision row (three views), GFLOPs", size=13, bold=True, color=MUTED, check=False)

# ---- 14 head architecture
s = new_slide("All seven networks share one wrapper. A single backbone reads the weekly, daily and hourly images in turn, the three feature vectors are joined, and a small multi-task head produces twelve numbers at once. "
              "Sharing the backbone across views keeps the model small and lets the head decide which timeframe to trust.")
header(s, "One multi-task head turns three charts into twelve numbers", "Computer vision · architecture", CYAN)
for i, (a, c) in enumerate([("Weekly chart", CYAN), ("Daily chart", CYAN), ("Hourly chart", CYAN)]):
    node(s, MX, 1.8 + i * 1.18, 1.9, 0.98, a, "224 x 224", c, tsize=14, ssize=12)
    arrow(s, MX + 1.9, 2.29 + i * 1.18, 3.1, 3.4, MUTED)
node(s, 3.1, 2.55, 2.1, 1.7, "Shared backbone", "ResNet, ConvNeXt, ViT, Swin...\nrun once per view", AMBER, tsize=15, ssize=12)
arrow(s, 5.2, 3.4, 5.75, 3.4)
node(s, 5.75, 2.55, 1.9, 1.7, "Concatenate", "three feature vectors, side by side", VIOLET, tsize=15, ssize=12)
arrow(s, 7.65, 3.4, 8.2, 3.4)
node(s, 8.2, 2.55, 2.0, 1.7, "Head", "LayerNorm, dropout 0.2, 512-unit GELU layer", UP, tsize=15, ssize=12)
arrow(s, 10.2, 3.4, 10.7, 3.4)
rect(s, 10.7, 1.85, 2.0, 3.1, fill=PANEL, line=UP, radius=0.08)
tb(s, 10.78, 1.92, 1.84, 0.4, "4 horizons x", size=13, bold=True, color=UP, align="c", check=False)
for i, (a, c) in enumerate([("3 direction logits", UP), ("1 volatility value", CYAN), ("1 drawdown logit", DOWN)]):
    rect(s, 10.82, 2.4 + i * 0.82, 1.76, 0.68, fill=PANEL2, line=c, radius=0.15)
    tb(s, 10.82, 2.4 + i * 0.82, 1.76, 0.68, a, size=12, bold=True, color=c, align="c", anchor="m", check=False)
rect(s, MX, 5.35, W - 2 * MX, 1.4, fill=PANEL2)
tb(s, MX + 0.3, 5.45, 12, 1.25, [[("Loss. ", {"bold": True, "color": AMBER}), ("Cross-entropy for direction, mean squared error for volatility and binary cross-entropy for drawdown, summed.", {})],
                                 [("Why this design. ", {"bold": True, "color": AMBER}), ("One backbone for all three views keeps parameters low, and one head keeps the twelve targets consistent across models.", {})]],
   size=14, space=8)

# ---- 15 training
s = new_slide("Training is deliberately plain so the comparison is fair: the same optimiser, schedule and early stopping for every network, and a short learning-rate probe to pick between two rates. "
              "A finding that surprised us: five of the seven networks were at their best after the first epoch, which tells us these results describe short, early-stopped fine-tuning.")
header(s, "Training every network the same way, and what the curves showed", "Computer vision · training", CYAN)
picture(s, os.path.join(FIG, "cv_training_curves.png"), MX, 1.7, W - 2 * MX, 1.5)
tb(s, MX, 3.2, W - 2 * MX, 0.35, "Training and early-stopping loss per epoch for the seven networks: later epochs fit the training block better, not the held-out block.", size=12, color=MUTED)
tb(s, MX, 3.75, 5.9, 3.1, [[("Optimiser. ", {"bold": True, "color": CYAN}), ("AdamW, weight decay 0.05, 5% warm-up then cosine decay", {})],
                           [("Precision. ", {"bold": True, "color": CYAN}), ("bfloat16 mixed precision, batches of 64 rows (192 images)", {})],
                           [("Length. ", {"bold": True, "color": CYAN}), ("At most six epochs, early stopping on 2021", {})]], size=16, bullets=True, space=12)
tb(s, 6.8, 3.75, 5.9, 2.0, [[("Learning rate. ", {"bold": True, "color": CYAN}), ("A short probe on a training subsample picks between two rates per network", {})],
                            [("Calibration. ", {"bold": True, "color": CYAN}), ("Fitted on 2021 only: temperature (direction), Platt (drawdown), linear (volatility)", {})]], size=16, bullets=True, space=12)
rect(s, 6.8, 5.75, 5.9, 1.0, fill=PANEL2)
tb(s, 7.0, 5.8, 5.5, 0.9, [[("4.2 GPU-hours ", {"bold": True, "color": AMBER, "size": 22}), ("for the full run. 5 of 7 networks peaked at epoch 1.", {"color": MUTED})]], size=13, anchor="m")

# ---- 16 CV results
s = new_slide("This is the confirmatory result for the vision networks on the locked test. Green bars passed the Holm-corrected test; grey did not. Six of twelve passed: direction at five and ten sessions, "
              "short volatility, and drawdown at five, ten and forty sessions. The direction gain is small, about three hundredths of macro-F1 above a shuffled-label null, but it is real.")
header(s, "Charts carry a small but real signal: 6 of 12 tests passed", "Computer vision · results", CYAN)
tasks = ["Dir 5", "Dir 10", "Dir 20", "Dir 40", "Vol 5", "Vol 10", "Vol 20", "Vol 40", "DD 5", "DD 10", "DD 20", "DD 40"]
skill = [0.032, 0.030, 0.027, 0.012, 0.084, 0.032, -0.029, -0.182, 0.105, 0.086, 0.000, 0.040]
ok = [1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1]
pc = [UP if o else (DOWN if v < -0.05 else "5B6B86") for o, v in zip(ok, skill)]
bar_chart(s, MX, 1.95, 8.3, 4.9, tasks, {"skill": skill}, [UP], fmt="+0.000;-0.000;0.000", legend=False, point_colors=pc, gap=40, vmin=-0.2, vmax=0.15, vfmt="0.00", label_size=11, major=0.05)
tb(s, MX, 1.7, 8.3, 0.3, "Skill over the uninformed benchmark. Green = supported after Holm correction", size=13, bold=True, color=MUTED, check=False)
stat(s, 9.3, 1.8, 3.4, 1.5, "+0.032", "direction at 5 sessions, 95% interval 0.019 to 0.043 (ConvNeXt-T)", UP, bsize=32, lsize=12)
stat(s, 9.3, 3.45, 3.4, 1.5, "+0.084", "volatility R-squared at 5 sessions (Swin-T)", CYAN, bsize=32, lsize=12)
stat(s, 9.3, 5.1, 3.4, 1.5, "0.605", "drawdown AUC at 5 sessions (Swin-T); 0.5 is chance", AMBER, bsize=32, lsize=12)

# ---- 17 vs baselines
s = new_slide("A fair test needs strong opponents. Our baselines are plain numbers: momentum, the HAR volatility model and a LightGBM on twenty technical indicators. On direction the charts look better on raw macro-F1, "
              "but that flatters them because the baselines mostly predict one class. Scored against each model's own shuffled null, the charts win only at five sessions. On volatility, the plain numbers win clearly.")
header(s, "Against plain numbers: charts help on direction, lose on volatility", "Computer vision · benchmarks", CYAN)
hz = ["5", "10", "20", "40"]
bar_chart(s, MX, 2.15, 6.1, 3.9, hz, {"Top-3 charts": [0.329, 0.360, 0.369, 0.363], "baseline_simple": [0.243, 0.256, 0.292, 0.287], "tech_gbm": [0.197, 0.226, 0.264, 0.298]},
          [CYAN, AMBER, "7C8CA8"], fmt="0.00", gap=55, vmin=0, vmax=0.45, label_size=11)
tb(s, MX, 1.75, 6.1, 0.35, "Direction, macro-F1, by horizon (sessions)", size=13, bold=True, color=MUTED, check=False)
bar_chart(s, 6.9, 2.15, 5.8, 3.9, hz, {"Top-3 charts": [0.132, 0.062, 0.010, -0.112], "baseline_simple": [0.304, 0.308, 0.317, 0.295], "tech_gbm": [0.371, 0.352, 0.314, 0.292]},
          [CYAN, AMBER, "7C8CA8"], fmt="0.00", gap=55, vmin=-0.2, vmax=0.45, label_size=11)
tb(s, 6.9, 1.75, 5.8, 0.35, "Volatility, R-squared, by horizon (sessions)", size=13, bold=True, color=MUTED, check=False)
rect(s, MX, 6.15, W - 2 * MX, 0.72, fill=PANEL2)
tb(s, MX + 0.25, 6.18, 12, 0.66, [[("Fair reading. ", {"bold": True, "color": AMBER}), ("Each model against its own shuffled null: charts are better at 5 sessions (+0.024, interval 0.007 to 0.039) and level at 10, 20 and 40.", {})]],
   size=14, anchor="m")

# ---- 18 why charts lose
s = new_slide("Why do charts lose on volatility? Because we scale every chart to its own high and low. A stock that moved two percent and one that moved twenty percent can produce nearly the same picture. "
              "The volatility level is exactly what the picture throws away, while a plain trailing-volatility number states it directly. Our ablation backs this up.")
header(s, "Why charts lose on volatility: every chart fills its own frame", "Computer vision · insight", CYAN)
rect(s, MX, 1.8, 3.0, 3.0, fill=PANEL, line=LINE)
rect(s, 3.75, 1.8, 3.0, 3.0, fill=PANEL, line=LINE)
candles(s, MX + 0.2, 2.35, 2.6, 2.2, n=14, seed=5)
candles(s, 3.95, 2.35, 2.6, 2.2, n=14, seed=5)
tb(s, MX + 0.15, 1.85, 2.7, 0.45, "Calm stock: moves 2%", size=14, bold=True, color=CYAN, align="c", check=False)
tb(s, 3.9, 1.85, 2.7, 0.45, "Wild stock: moves 20%", size=14, bold=True, color=AMBER, align="c", check=False)
tb(s, MX, 4.9, 6.15, 0.5, "After per-chart scaling, both pictures look the same to the network.", size=14, color=MUTED, align="c")
rect(s, MX, 5.5, 6.15, 1.3, fill=PANEL2)
tb(s, MX + 0.25, 5.55, 5.7, 1.2, [[("Fix we plan: ", {"bold": True, "color": UP}), ("a fixed-scale chart encoding that keeps the volatility level, tested on a forward window from 1 Dec 2026.", {})]], size=14, anchor="m")
ev = [("0.371 vs 0.132", "5-session volatility R-squared: tech_gbm vs top-3 charts", CYAN), ("-0.135", "volatility R-squared change from vertical-rescale augmentation", DOWN),
      ("0.693 vs 0.620", "drawdown AUC: simple trailing-volatility model vs top-3 charts", AMBER)]
for i, (b, l, c) in enumerate(ev):
    yy = 1.8 + i * 1.7
    rect(s, 7.2, yy, 5.5, 1.5, fill=PANEL, line=LINE)
    tb(s, 7.4, yy + 0.1, 5.1, 0.7, b, size=30, bold=True, color=c, anchor="m")
    tb(s, 7.4, yy + 0.82, 5.1, 0.62, l, size=13, color=MUTED)

# ---- 19 CV insights
s = new_slide("Four more things from the vision side. ConvNeXt-Tiny was the best direction network in every sector. Ensembles of two or three networks beat the best single one in five of twelve tasks and lost nowhere. "
              "Fine-tuning beat frozen features in nine of eighteen tests, so it is not a safe default. And the validation winner, ViT, dropped to third on the test. We also checked the famous chart patterns, and they added nothing.")
header(s, "What else the vision experiments taught us", "Computer vision · insight", CYAN)
tiles = [("Every sector", "ConvNeXt-Tiny was the best direction network in all five sectors", CYAN), ("5 of 12", "ensembles beat the best single network, and lost in none", UP),
         ("9 of 18", "fine-tuned beat frozen; EfficientNet-B0 got worse on all three targets", AMBER), ("1st to 3rd", "ViT-B/16 led on validation, ranked third on the test", DOWN)]
for i, (b, l, c) in enumerate(tiles):
    xx = MX + (i % 2) * 3.3
    yy = 1.8 + (i // 2) * 2.45
    stat(s, xx, yy, 3.15, 2.25, b, l, c, bsize=30, lsize=13)
bar_chart(s, 7.4, 2.15, 5.3, 3.3, ["No double bottom", "Double bottom"], {"avg 20-session return": [1.5, 0.9]}, [CYAN], fmt='0.0"%"', legend=False, gap=70, vmin=0, vmax=2.0,
          point_colors=["7C8CA8", AMBER], vfmt='0.0"%"')
tb(s, 7.4, 1.75, 5.3, 0.35, "Named chart patterns (YOLOv8 detector, weekly charts)", size=13, bold=True, color=MUTED, check=False)
tb(s, 7.4, 5.55, 5.3, 1.3, "Rows with a detected double bottom returned 0.9% on average over 20 sessions, against 1.5% without. So the dashboard labels candlestick patterns as description only.",
   size=13, color=MUTED)

# ---- 20 filings corpus
s = new_slide("Now the language side. Public companies file three main things: a yearly 10-K, quarterly 10-Qs, and an 8-K whenever something material happens. We collected over ten thousand of them for our fifty companies. "
              "Notice Financials file almost twice as many 8-Ks as other sectors, which gives that sector more training examples.")
header(s, "10,269 filings from SEC EDGAR for the same 50 companies", "Language · data", VIOLET)
forms = [("10-K", "Annual report", "554 filings. Median document 381,688 characters.", VIOLET), ("10-Q", "Quarterly report", "1,700 filings. Median document 153,539 characters.", CYAN),
         ("8-K", "Current event", "8,015 filings. Earnings releases, deals, officer changes. Median 8,233 characters.", AMBER)]
for i, (a, b, c, col) in enumerate(forms):
    yy = 1.8 + i * 1.7
    rect(s, MX, yy, 5.0, 1.55, fill=PANEL, line=LINE)
    tb(s, MX + 0.2, yy + 0.1, 1.5, 0.9, a, size=28, bold=True, color=col, anchor="m")
    tb(s, MX + 1.8, yy + 0.12, 3.0, 0.4, b, size=16, bold=True)
    tb(s, MX + 1.8, yy + 0.55, 3.0, 0.95, c, size=13, color=MUTED)
bar_chart(s, 5.9, 2.15, 6.8, 4.7, ["Comm. Services", "Cons. Discretionary", "Financials", "Health Care", "Info. Technology"],
          {"10-K": [110, 111, 110, 110, 113], "10-Q": [339, 342, 340, 340, 339], "8-K": [1419, 1414, 2518, 1383, 1281]}, [VIOLET, CYAN, AMBER], stacked=True, fmt="#,##0",
          vfmt="#,##0", gap=45, vmin=0, vmax=3200, label_size=10)
tb(s, 5.9, 1.75, 6.8, 0.35, "Filings by sector and form", size=13, bold=True, color=MUTED, check=False)

# ---- 21 NLP preprocessing
s = new_slide("Preprocessing on the text side is mostly about time and length. We stamp each filing with the moment EDGAR accepted it, and we only let a model use it thirty-five minutes later, at the next hourly bar. "
              "Then we pull out the section where management explains results, tag every input with its form, and cut it to what each model can read.")
header(s, "Five steps from an EDGAR download to model-ready text", "Language · pipeline", VIOLET)
st5 = [("1  Download", "SEC fair-access rules: declared user agent, 8 requests per second"), ("2  Timestamp", "acceptance time from the SGML header; usable 35 minutes later"),
       ("3  Extract", "MD&A for 10-K and 10-Q; body plus EX-99 for 8-K"), ("4  Tag", "every input starts with its form and section"), ("5  Tokenise", "512 tokens; 4 chunks frozen; 4,096 for Longformer")]
for i, (a, b) in enumerate(st5):
    xx = MX + i * 2.45
    node(s, xx, 1.8, 2.1, 2.0, a, b, VIOLET, tsize=15, ssize=12)
    if i < 4:
        arrow(s, xx + 2.1, 2.8, xx + 2.45, 2.8)
stat(s, MX, 4.2, 3.9, 1.75, "85.0%", "of 10-Ks had an MD&A section found (89.3% of 10-Qs); otherwise the full document", VIOLET, bsize=36, lsize=12)
stat(s, MX + 4.1, 4.2, 3.9, 1.75, "~20,000", "tokens in a median 10-K section (83,366 characters), far above any model window", AMBER, bsize=36, lsize=12)
stat(s, MX + 8.2, 4.2, 3.9, 1.75, "2,165", "filing events in the locked test, 4,149 in training", CYAN, bsize=36, lsize=12)
tb(s, MX, 6.15, 12.1, 0.7, "Labels are measured from the entry price after the 35-minute delay, never from the day's open, which would let a filing look like it predicted a move that happened before it.",
   size=14, color=MUTED)

# ---- 22 text models
s = new_slide("We picked the seven text models so each one changes a single thing. TF-IDF is the classical baseline. FinBERT versus SEC-BERT tests domain pre-training. RoBERTa and DeBERTa test size. "
              "Longformer tests reading eight times more text. And FinGPT, a seven-billion-parameter model, tests whether a generative financial model adds anything.")
header(s, "Seven text models, each chosen to test one idea", "Language · models", VIOLET)
rows = [["Model", "Params (M)", "What it tests"], ["TF-IDF + linear", "0.75", "The classical baseline with few parameters"],
        ["FinBERT", "109.9", "BERT adapted to financial text (domain)"], ["SEC-BERT", "109.5", "Pre-trained on SEC filings (domain, closer to our data)"],
        ["RoBERTa-large", "355.9", "A general model about three times larger (size)"], ["DeBERTa-v3-large", "434.5", "Newer pre-training recipe at large size"],
        ["Longformer-4096", "149.1", "Sparse attention reads 8x more text (length)"], ["FinGPT (Llama-2-7B)", "6,738", "Generative financial LLM, 4-bit, LoRA adapter, frozen"]]
table(s, MX, 1.85, W - 2 * MX, [3.2, 1.6, 7.3], rows, header_color=VIOLET, size=15, row_h=0.6, align=["l", "c", "l"], hl_col=0)

# ---- 23 NLP training
s = new_slide("Fine-tuning five encoders on about four thousand filings taught us humility. All five were best after the first epoch. DeBERTa was unstable, its loss stayed well above the rest and rose again, "
              "which matches what the literature says about big encoders on small noisy data. FinGPT stayed frozen in four-bit form to fit the GPU budget.")
header(s, "Fine-tuning on ~4,000 filings, and DeBERTa struggled", "Language · training", VIOLET)
bar_chart(s, MX, 2.15, 6.3, 4.1, ["Longformer", "FinBERT", "RoBERTa-L", "SEC-BERT", "DeBERTa-v3-L"], {"best early-stopping loss": [1.441, 1.453, 1.468, 1.523, 1.907]}, [VIOLET],
          fmt="0.00", legend=False, gap=50, vmin=0, vmax=2.2, point_colors=[VIOLET, VIOLET, VIOLET, VIOLET, DOWN])
tb(s, MX, 1.75, 6.3, 0.35, "Best early-stopping loss per encoder (lower is better)", size=13, bold=True, color=MUTED, check=False)
tb(s, MX, 6.3, 6.3, 0.55, "Learning rates kept by the probe: 2e-5 (FinBERT, SEC-BERT), 1e-5 (RoBERTa), 3e-5 (Longformer), 1.5e-5 (DeBERTa).", size=12, color=MUTED)
tb(s, 7.2, 1.8, 5.5, 5.0, [[("Fine-tuned route. ", {"bold": True, "color": VIOLET}), ("Encoder and the same multi-task head trained together, AdamW, at most 3 epochs, early stopping on 2021.", {})],
                           [("Frozen route. ", {"bold": True, "color": VIOLET}), ("Mean-pool four 512-token chunks, reduce to 32 principal components, fit regularised linear heads.", {})],
                           [("FinGPT. ", {"bold": True, "color": VIOLET}), ("Llama-2-7B in 4-bit precision with the FinGPT LoRA adapter, frozen. QLoRA fine-tuning is planned future work.", {})],
                           [("Calibration. ", {"bold": True, "color": VIOLET}), ("Same as the vision side, fitted on 2021 only.", {})]],
   size=14, bullets=True, space=10, bcolor=VIOLET)

# ---- 24 event study
s = new_slide("The event study gave our clearest result. On filing days the average absolute two-day price move was about 1.4 normal-day units, against 0.7 on matched days without a filing. "
              "Quarterly reports move prices most. So filings really do matter to the market, which makes the next question fair: can a model forecast what they will do?")
header(s, "Filings roughly double the size of a price reaction", "Language · event study", VIOLET)
bar_chart(s, MX, 2.15, 7.2, 4.6, ["10-Q", "10-K", "8-K", "No filing (control)"], {"mean |abnormal move|": [1.64, 1.52, 1.37, 0.70]}, [VIOLET], fmt="0.00", legend=False, gap=45,
          vmin=0, vmax=2.0, point_colors=[VIOLET, VIOLET, VIOLET, "7C8CA8"])
tb(s, MX, 1.75, 7.2, 0.35, "Mean absolute abnormal two-day move, in normal-day units (locked test)", size=13, bold=True, color=MUTED, check=False)
stat(s, 8.2, 1.8, 4.5, 1.5, "p < 0.001", "Mann-Whitney: filing days move more than matched non-filing days", UP, bsize=30, lsize=12)
stat(s, 8.2, 3.45, 4.5, 1.5, "p = 0.006", "Kruskal-Wallis: reaction size differs by form", CYAN, bsize=30, lsize=12)
stat(s, 8.2, 5.1, 4.5, 1.6, "2.39", "mean move for 8-K item 2.02, results of operations (1,446 filings, development blocks)", AMBER, bsize=30, lsize=12)

# ---- 25 NLP forecasting
s = new_slide("Here is the hard part. Judged only against an uninformed benchmark, our text models show skill on drawdown risk, with Longformer reaching AUCs between 0.61 and 0.65. "
              "But volatility R-squared is negative at every horizon. And when we compare against baselines that know the company and simple price features, the text wins only one of twelve tasks.")
header(s, "Text helps on drawdown only, and fades against fair baselines", "Language · results", VIOLET)
tsk = ["Dir 5", "Dir 10", "Dir 20", "Dir 40", "Vol 5", "Vol 10", "Vol 20", "Vol 40", "DD 5", "DD 10", "DD 20", "DD 40"]
nsk = [0.003, 0.001, 0.003, 0.005, -0.023, -0.040, -0.021, -0.022, 0.151, 0.112, 0.139, 0.128]
bar_chart(s, MX, 1.95, 7.4, 4.9, tsk, {"skill": nsk}, [VIOLET], fmt="+0.00;-0.00;0.00", legend=False, gap=40, vmin=-0.05, vmax=0.2, vfmt="0.00", major=0.05,
          point_colors=[("5B6B86" if abs(v) < 0.01 else (UP if v > 0 else DOWN)) for v in nsk], label_size=11)
tb(s, MX, 1.7, 7.4, 0.3, "Skill of the validation-chosen text model over the uninformed benchmark", size=13, bold=True, color=MUTED, check=False)
tb(s, 8.3, 1.75, 4.4, 0.35, "VS NO-TEXT BASELINES, SAME FILINGS", size=12, bold=True, color=AMBER, check=False)
for i, (b, l, c) in enumerate([("1 of 12", "text better (SEC-BERT, 20-session direction)", UP), ("7 of 12", "no clear difference", "9AA9BF"), ("4 of 12", "text worse (all volatility)", DOWN)]):
    yy = 2.15 + i * 1.5
    rect(s, 8.3, yy, 4.4, 1.35, fill=PANEL, line=LINE)
    tb(s, 8.45, yy + 0.08, 4.1, 0.65, b, size=30, bold=True, color=c, anchor="m")
    tb(s, 8.45, yy + 0.75, 4.1, 0.55, l, size=13, color=MUTED)
tb(s, 8.3, 6.6, 4.4, 0.3, "Baselines: company base rates; momentum + volatility; technical model.", size=11, color=MUTED, check=False)

# ---- 26 shortcut learning
s = new_slide("This slide explains why. When we add text features to a technical model, volatility R-squared goes up by about 0.075. But 0.063 of that comes from the filing-calendar columns alone, with no words at all. "
              "The words add only 0.013 beyond the calendar, and 0.007 beyond calendar and company. The models learned who filed and when. That is shortcut learning. But the words are not useless: they do help explain how big a reaction is.")
header(s, "Most of the text signal is shortcut learning, not reading", "Language · analysis", VIOLET)
bar_chart(s, MX, 2.15, 6.7, 4.0, ["All text features", "Calendar columns only", "Words beyond calendar", "Words beyond calendar + company"],
          {"gain in volatility R-squared": [0.075, 0.063, 0.013, 0.007]}, [VIOLET], fmt="+0.000", legend=False, gap=45, vmin=0, vmax=0.09, point_colors=[VIOLET, AMBER, CYAN, CYAN], size=11, vfmt="0.00")
tb(s, MX, 1.75, 6.7, 0.35, "Where the volatility gain comes from (technical model + text vs technical alone)", size=13, bold=True, color=MUTED, check=False)
tb(s, MX, 6.2, 6.7, 0.65, "Integrated-gradients attributions for the largest reactions were dominated by dates, company names and formatting tokens.", size=12, color=MUTED)
rect(s, 7.6, 1.8, 5.1, 2.35, fill=PANEL, line=DOWN)
tb(s, 7.8, 1.9, 4.7, 0.4, "THE SHORTCUT", size=12, bold=True, color=DOWN)
tb(s, 7.8, 2.3, 4.7, 1.8, "A filing reveals the company. The calendar reveals when earnings are due. Both predict volatility without reading a word.", size=15)
rect(s, 7.6, 4.35, 5.1, 2.45, fill=PANEL, line=UP)
tb(s, 7.8, 4.45, 4.7, 0.4, "BUT THE WORDS DO SOMETHING", size=12, bold=True, color=UP)
tb(s, 7.8, 4.85, 4.7, 1.9, [[("Predicting the size of a reaction: ", {}), ("R-squared 0.181 to 0.235", {"bold": True, "color": UP}), (" with TF-IDF words, and ", {}),
                             ("+0.043 (0.029 to 0.058)", {"bold": True, "color": UP}), (" even after controlling for the company.", {})]], size=14)

# ---- 27 ablations
s = new_slide("The ablations tell us what actually helps. Reading four chunks instead of the first five hundred tokens helped on risk targets. Reading longer helped Longformer. "
              "Domain pre-training helped direction but hurt volatility. Fine-tuning versus frozen was a mixed bag. And averaging language models never beat the best single one.")
header(s, "Ablations: more text and domain data help risk, fine-tuning is mixed", "Language · ablations", VIOLET)
ab = ["4 chunks vs first 512: RoBERTa-L (dd AUC)", "Longformer vs FinBERT, length (dd AUC)", "4 chunks vs first 512: SEC-BERT (dd AUC)", "Fine-tuned vs frozen: Longformer (dd AUC)",
      "SEC-BERT vs FinBERT, domain (direction F1)", "Fine-tuned vs frozen: FinBERT (direction F1)", "SEC-BERT vs FinBERT, domain (vol R-squared)"]
av = [0.140, 0.115, 0.101, 0.084, 0.067, -0.034, -0.148]
bar_chart(s, MX, 1.95, 8.6, 4.85, ab, {"delta": av}, [UP], kind="bar", fmt="+0.000;-0.000", legend=False, gap=40, vmin=-0.2, vmax=0.2, vfmt="0.0",
          point_colors=[UP if v > 0 else DOWN for v in av], size=11, label_size=11)
tb(s, MX, 1.7, 8.6, 0.3, "Validation, 5,000 paired bootstrap draws: change versus the comparison model", size=13, bold=True, color=MUTED, check=False)
stat(s, 9.5, 1.85, 3.2, 1.75, "0 of 12", "tasks where averaging 2 to 3 language models beat the best single one", DOWN, bsize=30, lsize=12)
stat(s, 9.5, 3.8, 3.2, 1.75, "2.92", "TF-IDF had the best mean validation rank of all seven text models", AMBER, bsize=30, lsize=12)
stat(s, 9.5, 5.75, 3.2, 1.05, "SEC-BERT", "ranked first on the test (2.58)", VIOLET, bsize=18, lsize=12)

# ---- 28 fusion
s = new_slide("Now we put charts and text together. Late fusion, where each branch is calibrated and then blended, beat the better single modality in six of twelve tests, including all four volatility horizons. "
              "On this chart each bar counts, out of twelve tasks, where the comparison was better, level or worse. The deployed recipe beat its uninformed benchmark in nine of twelve.")
header(s, "Putting charts and text together: fusion helps where it matters", "Fusion · results", UP)
cmp = ["H1c  charts vs numbers-only", "H2c  text vs no-text (same filings)", "H3c  fusion vs numbers-only", "H9a  chart combos vs best single chart", "H8  technical + charts + text vs technical", "H8f  text beyond calendar + company"]
bar_chart(s, MX, 1.95, 7.9, 4.85, cmp, {"better": [4, 1, 4, 5, 6, 1], "no clear difference": [0, 7, 1, 7, 4, 9], "worse": [8, 4, 7, 0, 2, 2]}, [UP, "5B6B86", DOWN], kind="bar", stacked=True,
          fmt="0;;", gap=40, vmin=0, vmax=12, vfmt="0", size=11, label_size=11)
tb(s, MX, 1.7, 7.9, 0.3, "Locked-test comparisons, tasks out of 12 (Holm-corrected)", size=13, bold=True, color=MUTED, check=False)
stat(s, 8.9, 1.85, 3.8, 1.5, "6 of 12", "fusion beats the better single modality (all 4 volatility horizons)", UP, bsize=30, lsize=12)
stat(s, 8.9, 3.5, 3.8, 1.5, "9 of 12", "dashboard recipe beats its uninformed benchmark", AMBER, bsize=30, lsize=12)
rect(s, 8.9, 5.15, 3.8, 1.65, fill=PANEL2)
tb(s, 9.05, 5.2, 3.5, 1.55, [[("Text share of the blend: ", {"bold": True, "color": CYAN}), ("direction 0 to 25%, volatility 25 to 75%, drawdown 100%.", {})]], size=13, anchor="m")

# ---- 29 scoreboard
s = new_slide("This is the scoreboard against everything. For volatility and drawdown, a technical model that also sees our chart and text outputs is the best forecast in the whole study. "
              "That is the practical message: use charts and text alongside plain numbers, not instead of them.")
header(s, "Add charts and text to plain numbers, do not replace them", "Fusion · results", UP)
ser = {"Technical + charts + text": [0.446, 0.446, 0.405, 0.355], "tech_gbm": [0.371, 0.352, 0.314, 0.292], "baseline_simple": [0.304, 0.308, 0.317, 0.295],
       "Top-3 charts": [0.132, 0.062, 0.010, -0.112], "Top-3 text": [-0.075, -0.090, -0.076, -0.079]}
cols = [UP, "7C8CA8", AMBER, CYAN, VIOLET]
line_chart(s, MX, 2.15, 6.1, 4.1, ["5", "10", "20", "40"], ser, cols, fmt="0.0", vmin=-0.2, vmax=0.5, legend=False)
tb(s, MX, 1.75, 6.1, 0.35, "Volatility, R-squared, by horizon (sessions)", size=13, bold=True, color=MUTED, check=False)
ser2 = {"Technical + charts + text": [0.710, 0.687, 0.639, 0.575], "tech_gbm": [0.701, 0.649, 0.620, 0.602], "baseline_simple": [0.693, 0.666, 0.633, 0.590],
        "Top-3 charts": [0.620, 0.590, 0.537, 0.518], "Top-3 text": [0.612, 0.601, 0.588, 0.586]}
line_chart(s, 6.9, 2.15, 5.8, 4.1, ["5", "10", "20", "40"], ser2, cols, fmt="0.0", vmin=0.45, vmax=0.75, legend=False)
tb(s, 6.9, 1.75, 5.8, 0.35, "Drawdown, ROC-AUC, by horizon (sessions)", size=13, bold=True, color=MUTED, check=False)
xx = MX
for n, c in zip(ser.keys(), cols):
    w_ = 0.5 + 0.092 * len(n)
    rect(s, xx, 6.5, 0.22, 0.22, fill=c, shape=MSO_SHAPE.OVAL)
    tb(s, xx + 0.28, 6.43, w_, 0.36, n, size=12, color=TEXT, anchor="m", check=False)
    xx += w_ + 0.45

# ---- 30 dashboard
s = new_slide("The dashboard is where all of this becomes usable. You paste three TradingView screenshots, or type a ticker. The app reads prices straight from the pixels, redraws the charts in the exact training format, "
              "runs the networks, blends in text if you add a filing, and draws a Monte Carlo price cone. The Ask AI tab explains the result using only our verified findings.")
header(s, "From a screenshot to an explained forecast", "Dashboard", AMBER)
fl = [("Paste screenshots", "1W, 1D, 1H from TradingView", CYAN), ("Read the pixels", "OCR on the price axis plus a candle reader, checked against the legend", CYAN),
      ("Redraw and predict", "training-format charts into 7 networks, text blended if a filing is given", AMBER),
      ("Calibrate and simulate", "12 outputs, 4,000-path Monte Carlo cone", UP), ("Ask AI explains", "Gemini, grounded in the study's facts and your numbers", UP)]
for i, (a, b, c) in enumerate(fl):
    yy = 1.75 + i * 1.03
    node(s, MX, yy, 5.6, 0.88, a, b, c, tsize=14, ssize=12)
    if i < 4:
        arrow(s, MX + 2.8, yy + 0.88, MX + 2.8, yy + 1.03)
picture(s, os.path.join(EXA, "NVDA_model_input.png"), 6.6, 1.8, 6.1, 2.3)
tb(s, 6.6, 4.15, 6.1, 0.55, "What the networks saw for NVDA: weekly, daily and hourly charts redrawn from our screenshots", size=12, color=MUTED)
stat(s, 6.6, 4.8, 2.95, 1.9, "0.08%", "median price error reading 66 real screenshots", UP, bsize=30, lsize=12)
stat(s, 9.75, 4.8, 2.95, 1.9, "82.6% / 90.8%", "of realised 20-session returns fell inside the 80% / 90% cone", CYAN, bsize=22, lsize=12)

# ---- 31 demo results
s = new_slide("Here are four live examples from our own screenshots on the twenty-third of November. Nvidia leans up, Visa leans down, Ford is sideways and Bank of America leans up. "
              "These are chances, not certainties: the direction edge in the study is small, so we show them as a nudge. The dashboard also gives a ninety percent price range for each.")
header(s, "Live demo: four stocks through the full system", "Dashboard · demo", AMBER)
names = {"NVDA": "NVIDIA", "V": "Visa", "F": "Ford Motor", "BAC": "Bank of America"}
hdr = ["Stock", "Last close", "20-session call", "Expected vol", "P(fall > 10%)", "90% price range"]
rows = [hdr]
for t in ("NVDA", "V", "F", "BAC"):
    x, q = r20(t), ex[t]["quantiles"]
    rows.append([f"{t}  {names[t]}", f"${ex[t]['p0']:,.2f}", x["call"], f"{x['expected vol (ann.)']:.0%}", f"{x['P(drawdown > 10%)']:.0%}", f"${q[0][-1]:,.0f} to ${q[4][-1]:,.0f}"])
table(s, MX, 1.85, 7.4, [1.95, 0.95, 1.15, 0.9, 0.95, 1.5], [["Stock", "Last close", "Call (20 sessions)", "Expected vol", "P(fall > 10%)", "90% price range"]] + rows[1:], header_color=AMBER, size=12, row_h=0.62, align=["l", "c", "c", "c", "c", "c"], hl_col=0)
bar_chart(s, 8.2, 2.05, 4.5, 3.3, list(names.keys()),
          {"DOWN": [r20(t)["P(DOWN)"] * 100 for t in names], "SIDEWAYS": [r20(t)["P(SIDEWAYS)"] * 100 for t in names], "UP": [r20(t)["P(UP)"] * 100 for t in names]},
          [DOWN, AMBER, UP], stacked=True, fmt='0"%"', vfmt='0"%"', vmin=0, vmax=100, gap=45, size=11, label_size=10)
tb(s, 8.2, 1.7, 4.5, 0.35, "20-session direction probabilities", size=13, bold=True, color=MUTED, check=False)
rect(s, MX, 5.1, 7.4, 1.65, fill=PANEL2)
tb(s, MX + 0.25, 5.15, 6.9, 1.55, [[("How to read it. ", {"bold": True, "color": AMBER}), ("The call is the most likely of three outcomes, with the odds beside it. Visa shows the highest chance of a 10% fall (28%). "
                                                                                              "No filing text was added, so these come from the charts only.", {})]], size=13, anchor="m")
tb(s, 8.2, 5.5, 4.5, 1.3, "Computed with the full frozen models from screenshots taken on 24 November 2026. Educational research, not financial advice.", size=12, color=MUTED)

# ---- 32 tools
s = new_slide("A quick tour of the tools. We used PyTorch and timm for the vision networks, Hugging Face for the language models, scikit-learn and LightGBM for the baselines, "
              "Colab for the GPUs, and Streamlit with Gemini for the dashboard and assistant.")
header(s, "The tools we used, and the job each one did", "Tools", AMBER)
tools = [("PyTorch + timm", "Vision networks, training loop, pretrained backbones", CYAN), ("Ultralytics YOLOv8", "YOLOv8s-cls backbone and the zero-shot pattern detector", CYAN),
         ("Hugging Face", "Transformers and tokenizers for FinBERT, SEC-BERT, RoBERTa, DeBERTa, Longformer", VIOLET), ("bitsandbytes + LoRA", "4-bit loading of Llama-2-7B with the FinGPT adapter", VIOLET),
         ("scikit-learn", "TF-IDF, linear heads, PCA and calibration helpers", AMBER), ("LightGBM", "Technical-indicator baseline and fusion models", AMBER),
         ("Google Colab", "A100 and L4 GPUs for the 4.2-hour confirmatory run", UP), ("RapidOCR", "Reads the price axis from screenshots", UP),
         ("Streamlit + Plotly", "The dashboard, interactive charts and flow diagrams", CYAN), ("Gemini API", "The Ask AI assistant, grounded in verified results", VIOLET),
         ("SEC EDGAR", "Source of 10,269 filings under fair-access rules", AMBER), ("GitHub + Drive", "Public code, notebook, tables and datasets", UP)]
for i, (a, b, c) in enumerate(tools):
    xx = MX + (i % 4) * 3.05
    yy = 1.8 + (i // 4) * 1.7
    rect(s, xx, yy, 2.9, 1.55, fill=PANEL, line=LINE)
    tb(s, xx + 0.15, yy + 0.08, 2.6, 0.4, a, size=15, bold=True, color=c)
    tb(s, xx + 0.15, yy + 0.52, 2.6, 1.0, b, size=13, color=TEXT)

# ---- 33 ethics
s = new_slide("Ethics is part of the design, not an appendix. For each risk we name what could go wrong and what we did about it. The two I would stress: the shortcut learning, which is a form of bias toward companies, "
              "and the AI assistant, which can sound sure of itself. We ground it in verified results and tell users their questions go to Google.")
header(s, "Responsible AI: each risk and what we did about it", "Ethics", DOWN)
eth = [("Look-ahead bias", "A model that saw the future looks smart.", "Test starts after every model's cut-off; point-in-time stamps; hashed vault.", AMBER),
       ("Bias and shortcuts", "Models learn which company filed, and Financials had twice the 8-Ks.", "Company and calendar baselines; results by sector; out-of-universe warning.", VIOLET),
       ("Privacy and data rights", "Filings are public records. The price dataset has no declared licence.", "Fair-access rules followed; licence caveat stated; academic use only; no weights shared.", CYAN),
       ("Misuse of forecasts", "A confident number can be read as advice.", "Direction call shown only where the test supported it; drawdown labelled relative risk.", DOWN),
       ("The AI assistant", "A fluent answer can sound more certain than the evidence.", "Grounded prompt, no buy or sell advice, caveats stated; users told questions go to Google.", UP),
       ("Environmental cost", "GPU training uses energy.", "About 4.2 GPU-hours total; later analyses reused cached predictions.", "7C8CA8")]
for i, (a, r_, m, c) in enumerate(eth):
    xx = MX + (i % 3) * 4.1
    yy = 1.8 + (i // 3) * 2.5
    rect(s, xx, yy, 3.9, 2.35, fill=PANEL, line=LINE)
    tb(s, xx + 0.2, yy + 0.1, 3.5, 0.4, a, size=16, bold=True, color=c)
    tb(s, xx + 0.2, yy + 0.55, 3.5, 0.75, [[("Risk  ", {"bold": True, "color": MUTED}), (r_, {})]], size=13)
    tb(s, xx + 0.2, yy + 1.3, 3.5, 1.0, [[("Response  ", {"bold": True, "color": UP}), (m, {})]], size=13)

# ---- 34 limits and next
s = new_slide("To be straight about the limits: fifty large survivors, one training seed per model, one test period that was a rally, and some analyses added after we opened the test, which we label post-hoc. "
              "What we learned matters more than the scoreboard. And we have a pre-registered follow-up with a forward test starting December first.")
header(s, "What we learned, where it stops, and what comes next", "Wrap-up", AMBER)
rect(s, MX, 1.8, 4.0, 5.0, fill=PANEL, line=LINE)
tb(s, MX + 0.2, 1.9, 3.6, 0.4, "WHAT WE LEARNED", size=13, bold=True, color=UP)
tb(s, MX + 0.2, 2.35, 3.6, 4.4, ["Controls matter more than model size.", "TF-IDF is a serious competitor on small data.", "Large encoders need seeds and careful schedules.",
                                 "A grounded assistant is only as good as its facts."], size=16, bullets=True, bcolor=UP, space=14)
rect(s, 4.8, 1.8, 3.9, 5.0, fill=PANEL, line=LINE)
tb(s, 5.0, 1.9, 3.5, 0.4, "WHERE IT STOPS", size=13, bold=True, color=DOWN)
tb(s, 5.0, 2.35, 3.5, 4.4, ["50 large surviving US stocks", "One training seed per model", "One test period, a strong rally", "FinGPT used frozen, DeBERTa under-trained",
                            "No trading costs", "Post-hoc analyses are labelled"], size=16, bullets=True, bcolor=DOWN, space=11)
rect(s, 8.9, 1.8, 3.8, 5.0, fill=PANEL, line=AMBER)
tb(s, 9.1, 1.9, 3.4, 0.4, "NEXT: v3.1 PROTOCOL", size=13, bold=True, color=AMBER)
nx = ["Fixed-scale charts", "Five seeds per model", "Layer-wise learning-rate decay", "QLoRA for FinGPT", "Minute-stamped news", "Forward test from 1 Dec 2026", "Wider universe, with delisted firms"]
for i, n_ in enumerate(nx):
    yy = 2.4 + i * 0.62
    rect(s, 9.1, yy + 0.07, 0.3, 0.3, fill=AMBER, shape=MSO_SHAPE.OVAL)
    tb(s, 9.1, yy + 0.07, 0.3, 0.3, str(i + 1), size=11, bold=True, color=BG, align="c", anchor="m", check=False)
    tb(s, 9.5, yy, 3.1, 0.45, n_, size=14, anchor="m")

# ---- 35 team and links
s = new_slide("To close: we built a fair test, found a small real signal in charts, a clear market effect from filings, and learned that numbers plus charts plus text work best together. "
              "Everything is public: the code, the notebook, the datasets and the dashboards. Thank you, and we are happy to take questions and run the demo live.")
candles(s, 7.9, 0.45, 4.9, 1.75, n=28, seed=21)
tb(s, MX, 0.7, 6.8, 1.2, "Thank you", size=50, bold=True)
tb(s, MX, 1.75, 6.8, 0.5, "Questions, and a live demo", size=20, color=AMBER, bold=True)
tb(s, MX, 2.5, 12.1, 0.85, [[("Krishna Harsha Tangirala (lead author)", {"bold": True, "color": TEXT}), ("  ·  Mohammed Shahrukh  ·  Valavala Divya Anu Sri  ·  Sreeya Tirumalasetti  ·  Venkata Saisravan Pasam", {})]],
   size=14, color=MUTED)
lk = [("Code, notebook, results", REPO, CYAN), ("CV dataset (public Drive)", "drive.google.com/file/d/1Y7wvfpJcjnmXLP79hHTih-jj5pt-muu-", CYAN),
      ("NLP dataset (public Drive)", "drive.google.com/drive/folders/1nczPvpd_wkMgF5N7uFBfAQMp-Ja96TRI", VIOLET), ("Lite dashboard (public, always on)", LITE, UP),
      ("Full dashboard (heavy, 13 networks)", FULL, AMBER)]
for i, (a, b, c) in enumerate(lk):
    yy = 3.55 + i * 0.7
    rect(s, MX, yy, W - 2 * MX, 0.6, fill=PANEL, line=LINE)
    tb(s, MX + 0.2, yy, 4.0, 0.6, a, size=14, bold=True, color=c, anchor="m")
    tb(s, MX + 4.3, yy, 7.7, 0.6, b, size=13, color=TEXT, anchor="m")
tb(s, MX, 7.08, 12, 0.3, "The full dashboard loads 13 networks (2.9 GB), so it runs on our machine during demos; the lite version is public. Educational research, not financial advice.",
   size=10, color=MUTED, check=False)

prs.save(OUT)
print("saved", OUT, "slides:", len(prs.slides))
for w_ in WARN:
    print("WARN", w_)
