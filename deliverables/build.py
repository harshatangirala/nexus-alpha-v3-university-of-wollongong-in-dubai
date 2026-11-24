"""Turns the LaTeX deliverables into PDFs without a TeX installation.

The .tex files are the masters (upload them to Overleaf). This script:
  1. orders the bibliography by first citation (IEEE style) in the .tex itself and checks every \\cite has a \\bibitem;
  2. counts the words of the body (references, tables and figure files excluded);
  3. converts the LaTeX subset used in these files to HTML (KaTeX for maths) and prints it to PDF with Chrome.

Usage (repository root):  python deliverables/build.py   [needs: pip install playwright; Google Chrome installed]
"""
import html
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = [("CV_Report", "report"), ("NLP_Report", "report"), ("IEEE_Paper", "ieee")]


# ------------------------------------------------------------------ bibliography order and checks
def order_bibliography(path):
    src = open(path, encoding="utf-8").read()
    m = re.search(r"\\begin\{thebibliography\}\{\d+\}(.*?)\\end\{thebibliography\}", src, re.S)
    if not m:
        return src, []
    items = re.split(r"(?=\\bibitem\{)", m.group(1).strip())
    items = [i.strip() for i in items if i.strip().startswith("\\bibitem")]
    bib = {re.match(r"\\bibitem\{([^}]+)\}", i).group(1): i for i in items}
    body = src[: m.start()]
    order = []
    for c in re.findall(r"\\cite\{([^}]+)\}", body):
        for k in [x.strip() for x in c.split(",")]:
            if k not in order:
                order.append(k)
    missing = [k for k in order if k not in bib]
    unused = [k for k in bib if k not in order]
    if missing:
        raise SystemExit(f"{os.path.basename(path)}: citations without a reference: {missing}")
    new = "\n\n".join(bib[k] for k in order) + ("\n\n" + "\n\n".join(bib[k] for k in unused) if unused else "")
    out = src[: m.start()] + "\\begin{thebibliography}{99}\n\n" + new + "\n\n\\end{thebibliography}" + src[m.end():]
    if out != src:
        open(path, "w", encoding="utf-8").write(out)
    return out, unused


# ------------------------------------------------------------------ word count
def body_words(src):
    b = src[src.index("\\begin{document}"): src.index("\\begin{thebibliography}") if "\\begin{thebibliography}" in src else None]
    b = re.sub(r"(?<!\\)%.*", "", b)
    b = re.sub(r"\\begin\{(table\*?|tabular)\}.*?\\end\{\1\}", " ", b, flags=re.S)
    b = re.sub(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}", " ", b)
    b = re.sub(r"\\(label|ref|cite|url|href)\{[^}]*\}", " ", b)
    b = re.sub(r"\$[^$]*\$", " x ", b)
    b = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", b)
    b = re.sub(r"[{}\\&~]", " ", b)
    return len([w for w in b.split() if re.search(r"[A-Za-z0-9]", w)])


# ------------------------------------------------------------------ LaTeX subset -> HTML
def braced(s, i):
    """Return (content, end) for the brace group starting at s[i] == '{'."""
    depth, j = 0, i
    while j < len(s):
        if s[j] == "{" and (j == 0 or s[j - 1] != "\\"):
            depth += 1
        elif s[j] == "}" and s[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return s[i + 1: j], j + 1
        j += 1
    raise ValueError("unbalanced braces near: " + s[i: i + 60])


def replace_cmd(s, cmd, fn, nargs=1):
    out, i = [], 0
    pat = "\\" + cmd
    while True:
        k = s.find(pat, i)
        if k < 0:
            out.append(s[i:])
            return "".join(out)
        nxt = s[k + len(pat): k + len(pat) + 1]
        if nxt.isalpha():
            out.append(s[i: k + len(pat)])
            i = k + len(pat)
            continue
        out.append(s[i:k])
        j = k + len(pat)
        opt = None
        if j < len(s) and s[j] == "[":
            e = s.index("]", j)
            opt, j = s[j + 1: e], e + 1
        args = []
        for _ in range(nargs):
            while j < len(s) and s[j] in " \n":
                j += 1
            a, j = braced(s, j)
            args.append(a)
        out.append(fn(*args) if opt is None else fn(*args, opt=opt) if "opt" in fn.__code__.co_varnames else fn(*args))
        i = j


class Doc:
    def __init__(self, src, kind):
        self.src, self.kind = src, kind
        self.labels, self.cites = {}, {}
        self.math = []
        self.sec = [0, 0, 0]

    # numbering of labels (sections, figures, tables, equations) in document order
    def scan_labels(self, body):
        sec = [0, 0, 0]
        fig = tab = eq = 0
        tokens = re.finditer(r"\\(section|subsection|subsubsection)(\*?)\{|\\begin\{(figure\*?|table\*?|equation)\}|\\label\{([^}]+)\}", body)
        last = None
        for t in tokens:
            if t.group(1):
                if t.group(2):
                    continue
                lvl = ["section", "subsection", "subsubsection"].index(t.group(1))
                sec[lvl] += 1
                for z in range(lvl + 1, 3):
                    sec[z] = 0
                last = self.secnum(sec, lvl)
            elif t.group(3):
                env = t.group(3).rstrip("*")
                if env == "figure":
                    fig += 1
                    last = str(fig)
                elif env == "table":
                    tab += 1
                    last = self.roman(tab) if self.kind == "ieee" else str(tab)
                else:
                    eq += 1
                    last = str(eq)
            elif t.group(4):
                self.labels[t.group(4)] = last

    @staticmethod
    def roman(n):
        vals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
        r = ""
        for v, s in vals:
            while n >= v:
                r += s
                n -= v
        return r

    def secnum(self, sec, lvl):
        if self.kind == "ieee":
            return [self.roman(sec[0]), f"{self.roman(sec[0])}-{chr(64 + sec[1])}" if sec[1] else "", ""][lvl] or self.roman(sec[0])
        return ".".join(str(x) for x in sec[: lvl + 1])

    def inline(self, s):
        # protect maths first
        def keep(m):
            self.math.append(m.group(0))
            return f"\x00{len(self.math) - 1}\x00"
        s = re.sub(r"\$[^$]+\$", keep, s)
        s = re.sub(r"\\\\(\[[^\]]*\])?", "\x01", s)
        s = re.sub(r"(?<!\\)%.*", "", s)
        s = html.escape(s, quote=False).replace("\x01", "<br>")
        s = s.replace("``", "\u201c").replace("''", "\u201d").replace("---", "\u2014").replace("--", "\u2013")
        s = s.replace("~", "\u00a0").replace("\\%", "%").replace("\\&amp;", "&amp;").replace("\\_", "_").replace("\\#", "#")
        s = s.replace("\\$", "$").replace("\\,", "\u2009").replace("\\ ", " ").replace("\\@", "")
        s = s.replace("\\ldots", "\u2026").replace("\\dots", "\u2026").replace("\\times", "\u00d7")
        s = replace_cmd(s, "textbf", lambda a: f"<b>{a}</b>")
        s = replace_cmd(s, "emph", lambda a: f"<i>{a}</i>")
        s = replace_cmd(s, "textit", lambda a: f"<i>{a}</i>")
        s = replace_cmd(s, "texttt", lambda a: f"<code>{a}</code>")
        s = replace_cmd(s, "textsc", lambda a: f"<span class=sc>{a}</span>")
        s = replace_cmd(s, "url", lambda a: f'<a href="{a}">{a}</a>')
        s = replace_cmd(s, "href", lambda a, b: f'<a href="{a}">{b}</a>', 2)
        s = replace_cmd(s, "ref", lambda a: self.labels.get(a, "??"))
        s = replace_cmd(s, "label", lambda a: "")
        s = replace_cmd(s, "footnote", lambda a: f" ({a})")
        s = replace_cmd(s, "cite", lambda a: "[" + ", ".join(str(self.cites[k.strip()]) for k in a.split(",")) + "]")
        s = re.sub(r"\\noindent\s*", "", s)
        s = re.sub(r"\\(smallskip|medskip|bigskip|centering|small|footnotesize|par|newline)\b\s*", "", s)
        s = re.sub(r"\x00(\d+)\x00", lambda m: html.escape(self.math[int(m.group(1))], quote=False), s)
        return s

    def tabular(self, t):
        spec_m = re.match(r"\s*\{([^}]*)\}", t)
        t = t[spec_m.end():] if spec_m else t
        rows = re.split(r"\\\\", t)
        out = ["<table>"]
        rule_next = ""
        for r in rows:
            r = r.strip()
            for rule in ("\\toprule", "\\midrule", "\\bottomrule", "\\hline"):
                if r.startswith(rule):
                    rule_next = "mid" if rule in ("\\midrule", "\\hline") else rule_next
                    r = r[len(rule):].strip()
            if not r or r in ("\\bottomrule", "\\hline"):
                continue
            cells = [c.strip() for c in re.split(r"(?<!\\)&", r)]
            tag = "th" if len(out) == 1 else "td"
            cls = ' class="rule"' if rule_next == "mid" else ""
            rule_next = ""
            out.append(f"<tr{cls}>" + "".join(f"<{tag}>{self.inline(c)}</{tag}>" for c in cells) + "</tr>")
        out.append("</table>")
        return "".join(out)

    def block(self, b):
        """Convert block-level structure of a body chunk."""
        out = []
        pos = 0
        pat = re.compile(r"\\begin\{(figure\*?|table\*?|itemize|enumerate|equation|abstract|IEEEkeywords|center|quote)\}"
                         r"|\\(section|subsection|subsubsection|paragraph)(\*?)\{")
        sec = self.sec
        while True:
            m = pat.search(b, pos)
            text = b[pos: m.start() if m else len(b)]
            for para in re.split(r"\n\s*\n", text):
                if para.strip():
                    out.append(f"<p>{self.inline(para.strip())}</p>")
            if not m:
                break
            if m.group(1):
                env = m.group(1)
                end = b.index(f"\\end{{{env}}}", m.end())
                inner = b[m.end(): end]
                pos = end + len(f"\\end{{{env}}}")
                base = env.rstrip("*")
                wide = " wide" if env.endswith("*") else ""
                if base == "figure":
                    imgs = re.findall(r"\\includegraphics(?:\[([^\]]*)\])?\{([^}]+)\}", inner)
                    cap = re.search(r"\\caption\{", inner)
                    capt, _ = braced(inner, cap.end() - 1) if cap else ("", 0)
                    lab = re.search(r"\\label\{([^}]+)\}", inner)
                    num = self.labels.get(lab.group(1), "") if lab else ""
                    im = "".join(f'<img src="{src}">' for _, src in imgs)
                    name = "Fig." if self.kind == "ieee" else "Figure"
                    out.append(f'<figure class="fig{wide}">{im}<figcaption><b>{name} {num}.</b> {self.inline(capt)}</figcaption></figure>')
                elif base == "table":
                    cap = re.search(r"\\caption\{", inner)
                    capt, _ = braced(inner, cap.end() - 1) if cap else ("", 0)
                    lab = re.search(r"\\label\{([^}]+)\}", inner)
                    num = self.labels.get(lab.group(1), "") if lab else ""
                    tabm = re.search(r"\\begin\{tabular\*?\}(.*?)\\end\{tabular\*?\}", inner, re.S)
                    name = "TABLE" if self.kind == "ieee" else "Table"
                    out.append(f'<div class="tbl{wide}"><div class="tcap"><b>{name} {num}.</b> {self.inline(capt)}</div>'
                               f'{self.tabular(tabm.group(1)) if tabm else ""}</div>')
                elif base in ("itemize", "enumerate"):
                    items = [x.strip() for x in re.split(r"\\item\s*", inner) if x.strip()]
                    tag = "ul" if base == "itemize" else "ol"
                    out.append(f"<{tag}>" + "".join(f"<li>{self.inline(x)}</li>" for x in items) + f"</{tag}>")
                elif base == "equation":
                    lab = re.search(r"\\label\{([^}]+)\}", inner)
                    num = self.labels.get(lab.group(1), "") if lab else ""
                    tex = re.sub(r"\\label\{[^}]+\}", "", inner).strip()
                    out.append(f'<div class="eq"><span class="math">$${html.escape(tex, quote=False)}$$</span><span class="eqn">({num})</span></div>')
                elif base == "abstract":
                    lead = "<i>Abstract</i>\u2014" if self.kind == "ieee" else ""
                    out.append(f'<div class="abstract"><p>{lead}{self.inline(inner.strip())}</p></div>' if self.kind == "ieee"
                               else f'<div class="abstract"><h2 class="nonum">Executive summary</h2><p>{self.inline(inner.strip())}</p></div>')
                elif base == "IEEEkeywords":
                    out.append(f'<div class="keywords"><p><i>Index Terms</i>\u2014{self.inline(inner.strip())}</p></div>')
                elif base in ("center", "quote"):
                    out.append(f'<div class="{base}">{self.block(inner)}</div>')
            else:
                kind, star = m.group(2), m.group(3)
                title, pos = braced(b, m.end() - 1)
                if kind == "paragraph":
                    out.append(f'<p class="runin"><b>{self.inline(title)}</b>')
                    # the paragraph text that follows is merged by the browser into the same <p> visually via CSS
                    continue
                lvl = ["section", "subsection", "subsubsection"].index(kind)
                if star:
                    out.append(f'<h{lvl + 2} class="nonum">{self.inline(title)}</h{lvl + 2}>')
                    continue
                sec[lvl] += 1
                for z in range(lvl + 1, 3):
                    sec[z] = 0
                if self.kind == "ieee":
                    num = self.roman(sec[0]) + "." if lvl == 0 else (chr(64 + sec[1]) + "." if lvl == 1 else f"{sec[2]})")
                    t = self.inline(title).upper() if lvl == 0 else self.inline(title)
                else:
                    num = ".".join(str(x) for x in sec[: lvl + 1])
                    t = self.inline(title)
                out.append(f'<h{lvl + 2}><span class="num">{num}</span> {t}</h{lvl + 2}>')
        return "\n".join(out)

    def render(self):
        src = self.src
        body = src[src.index("\\begin{document}") + len("\\begin{document}"): src.index("\\end{document}")]
        bibm = re.search(r"\\begin\{thebibliography\}\{\d+\}(.*?)\\end\{thebibliography\}", body, re.S)
        bibitems = re.findall(r"\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem\{|$)", bibm.group(1), re.S) if bibm else []
        for i, (k, _) in enumerate(bibitems, 1):
            self.cites[k] = i
        main = body[: bibm.start()] if bibm else body
        tail = body[bibm.end():] if bibm else ""
        self.scan_labels(main + tail)
        title = braced(src, src.index("\\title{") + 6)[0]
        authors = []
        for m in re.finditer(r"\\IEEEauthorblockN\{", src):
            name, end = braced(src, m.end() - 1)
            am = re.match(r"\s*\\IEEEauthorblockA\{", src[end:])
            aff = braced(src, end + am.end() - 1)[0] if am else ""
            authors.append((name, aff))
        if not authors:
            am = re.search(r"\\author\{", src)
            raw = braced(src, am.end() - 1)[0] if am else ""
            raw = re.sub(r"\s*\\and\s*", ", ", re.sub(r"\\\\\[[^\]]*\]", r"\\\\", raw))
            authors = [(raw, "")] if raw else []
        main = main.replace("\\maketitle", "")
        for cmd in ("\\title{", "\\author{"):
            k = main.find(cmd)
            if k >= 0:
                main = main[:k] + main[braced(main, k + len(cmd) - 1)[1]:]
        pre = re.search(r"%\s*HTML-FRONT-MATTER\s*(.*?)%\s*END-HTML-FRONT-MATTER", src, re.S)
        front = ""
        if pre:
            lines = [re.sub(r"^%\s?", "", l) for l in pre.group(1).strip().splitlines()]
            front = "<div class=frontlinks>" + "".join(f"<div>{self.inline(l)}</div>" for l in lines if l.strip()) + "</div>"
        auth_html = "".join(f'<div class=author><div class=aname>{self.inline(n)}</div><div class=aaff>{self.inline(a)}</div></div>'
                            for n, a in authors)
        head = f'<header><h1 class="title">{self.inline(title)}</h1><div class=authors>{auth_html}</div>{front}</header>'
        own_heading = main.rstrip().endswith("\\section{References}")
        content = self.block(main)
        refs = ""
        if bibitems:
            refs = ("" if own_heading else '<h2 class="nonum refs">References</h2>') + '<ol class="refs">' + "".join(
                f"<li value={i}>{self.inline(t.strip())}</li>" for i, (_, t) in enumerate(bibitems, 1)) + "</ol>"
        if tail.strip():
            refs += self.block(tail)
        return head, content, refs


CSS_COMMON = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: 'Times New Roman', Times, serif; color: #000; }
a { color: #0b3d91; text-decoration: none; word-break: break-all; }
table { border-collapse: collapse; width: 100%; font-size: 8.6pt; margin: 2px 0 4px; }
th { border-top: 1.2px solid #000; border-bottom: 0.8px solid #000; padding: 2px 4px; text-align: left; font-weight: bold; }
td { padding: 1.6px 4px; text-align: left; vertical-align: top; }
tr.rule td { border-top: 0.6px solid #000; }
table tr:last-child td { border-bottom: 1.2px solid #000; }
figure { margin: 6px 0 8px; break-inside: avoid; text-align: center; }
figure img { max-width: 100%; }
figcaption, .tcap { font-size: 8.6pt; text-align: left; margin-top: 3px; line-height: 1.25; }
.tbl { break-inside: avoid; margin: 6px 0 8px; }
.eq { display: flex; align-items: center; justify-content: center; margin: 4px 0; }
.eq .math { flex: 1; text-align: center; }
.eq .eqn { font-size: 9.5pt; }
code { font-family: 'Courier New', monospace; font-size: 0.92em; }
.frontlinks { font-size: 8.8pt; margin: 6px 0 10px; line-height: 1.35; text-align: center; }
ol.refs { padding-left: 2.2em; font-size: 8.4pt; line-height: 1.25; }
ol.refs li { margin-bottom: 2px; }
"""

CSS_REPORT = CSS_COMMON + """
body { font-size: 11pt; line-height: 1.42; }
h1.title { font-size: 18pt; text-align: center; margin: 0 0 6px; line-height: 1.2; }
.authors { display: flex; justify-content: center; gap: 30px; flex-wrap: wrap; margin-bottom: 2px; }
.author { text-align: center; } .aname { font-weight: bold; } .aaff { font-size: 9.5pt; }
h2 { font-size: 13pt; margin: 14px 0 4px; } h3 { font-size: 11.5pt; margin: 10px 0 3px; } h4 { font-size: 11pt; margin: 8px 0 2px; font-style: italic; }
h2 .num, h3 .num { margin-right: 6px; }
p { margin: 0 0 6px; text-align: justify; }
.abstract { border: 0.8px solid #888; padding: 6px 10px 2px; margin: 6px 0 10px; background: #f7f7f7; }
.abstract h2 { margin-top: 2px; }
figure img { max-height: 92mm; }
"""

CSS_IEEE = CSS_COMMON + """
@page { size: A4; margin: 17mm 14mm 20mm 14mm; }
body { font-size: 9.6pt; line-height: 1.2; }
header { text-align: center; margin-bottom: 8px; }
h1.title { font-size: 21pt; font-weight: normal; margin: 0 0 8px; line-height: 1.15; }
.authors { display: flex; justify-content: center; gap: 18px; flex-wrap: wrap; font-size: 9.5pt; }
.author { text-align: center; max-width: 660px; } .aname { font-size: 10.5pt; } .aaff { font-size: 8.6pt; line-height: 1.35; }
main { column-count: 2; column-gap: 6mm; text-align: justify; hyphens: auto; }
p { margin: 0; text-indent: 1em; }
h2 + p, h3 + p, .abstract p, .keywords p, figure + p, .tbl + p, .eq + p { text-indent: 0; }
h2 { font-size: 9.6pt; font-weight: normal; text-align: center; margin: 9px 0 4px; letter-spacing: 0.3px; font-variant: small-caps; }
h2 .num { margin-right: 5px; }
h3 { font-size: 9.6pt; font-weight: normal; font-style: italic; margin: 7px 0 2px; }
h4 { font-size: 9.6pt; font-weight: normal; font-style: italic; margin: 5px 0 1px; }
.abstract p, .keywords p { font-weight: bold; font-size: 8.9pt; text-indent: 0; margin-bottom: 4px; }
.keywords p { font-weight: bold; }
.wide { column-span: all; }
figure.wide img { max-width: 100%; max-height: 78mm; }
figure img { max-height: 70mm; }
ul, ol { margin: 2px 0 4px; padding-left: 1.4em; }
h2.refs { margin-top: 8px; }
"""


def build_pdf(name, kind):
    tex = os.path.join(HERE, name, f"{name}.tex")
    src, unused = order_bibliography(tex)
    words = body_words(src)
    doc = Doc(src, kind)
    head, content, refs = doc.render()
    css = CSS_IEEE if kind == "ieee" else CSS_REPORT
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{name}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
 onload="renderMathInElement(document.body,{{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}]}});document.body.dataset.ready=1"></script>
<style>{css}</style></head><body>{head}<main>{content}{refs}</main></body></html>"""
    out_html = os.path.join(HERE, name, f"{name}.html")
    open(out_html, "w", encoding="utf-8").write(page)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        pg = b.new_page()
        pg.goto("file:///" + out_html.replace("\\", "/"))
        pg.wait_for_function("document.body.dataset.ready == '1'", timeout=60000)
        pg.wait_for_timeout(500)
        pg.pdf(path=os.path.join(HERE, name, f"{name}.pdf"), format="A4", print_background=True, prefer_css_page_size=True,
               display_header_footer=True, header_template="<span></span>",
               footer_template='<div style="font-size:8px;width:100%;text-align:center;font-family:Times New Roman">'
                               '<span class="pageNumber"></span></div>')
        npages = len(re.findall(rb"/Type\s*/Page[^s]", open(os.path.join(HERE, name, f"{name}.pdf"), "rb").read()))
        b.close()
    os.remove(out_html)
    print(f"{name}: {words} body words, {npages} pages, unused references: {unused or 'none'}")


if __name__ == "__main__":
    which = sys.argv[1:] or [d for d, _ in DOCS]
    for d, k in DOCS:
        if d in which:
            build_pdf(d, k)
