#!/usr/bin/env python
"""Render a docs/ Markdown file to PDF, typesetting its LaTeX math.

Covers the Markdown the technical docs use: headings, paragraphs, bullet lists,
pipe tables, fenced code blocks, **bold**, `code`, links, inline ``$...$`` and
display ``$$...$$`` math (an equation's ``\\tag{n}`` is printed as its number
at the right margin), and images ``![caption](path)`` (path relative to the
Markdown file; the caption is printed under the image). Formulas are rendered
with matplotlib's mathtext (STIX fonts), so no TeX installation or browser is
needed. Two-row ``cases`` environments are supported; sizing commands
(``\\big``, ``\\textstyle``, ...) are dropped and ``\\text`` / ``\\operatorname``
become upright text; ``\\underbrace`` and other environments are not supported.

    pip install matplotlib reportlab   # not in requirements.txt
    python scripts/md_to_pdf.py docs/age_training_scenario_D.md            # -> .pdf beside it
    python scripts/md_to_pdf.py docs/x.md --out /tmp/x.pdf
"""
import argparse
import hashlib
import os
import re
import tempfile
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["savefig.transparent"] = True   # formulas sit on tinted cells too
from matplotlib import font_manager, mathtext  # noqa: E402
from matplotlib.image import imread  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    XPreformatted,
)

DPI = 300
BODY_PT = 10
INK = colors.HexColor("#17202a")
MUTED = colors.HexColor("#5b6876")
ACCENT = colors.HexColor("#0b6e8a")
RULE = colors.HexColor("#d4dbe2")
TINT = colors.HexColor("#eef3f6")


def _register_fonts():
    ttf = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    for name, file in (("Body", "DejaVuSans.ttf"), ("Body-Bold", "DejaVuSans-Bold.ttf"),
                       ("Body-Oblique", "DejaVuSans-Oblique.ttf"), ("Mono", "DejaVuSansMono.ttf")):
        pdfmetrics.registerFont(TTFont(name, os.path.join(ttf, file)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold",
                                  italic="Body-Oblique", boldItalic="Body-Bold")


def _styles():
    base = dict(fontName="Body", fontSize=BODY_PT, leading=BODY_PT * 1.45, textColor=INK,
                alignment=TA_LEFT)
    return {
        "title": ParagraphStyle("title", **{**base, "fontName": "Body-Bold", "fontSize": 20,
                                            "leading": 25, "spaceAfter": 10}),
        "h2": ParagraphStyle("h2", **{**base, "fontName": "Body-Bold", "fontSize": 14,
                                      "leading": 18, "spaceBefore": 14, "spaceAfter": 6,
                                      "textColor": ACCENT, "keepWithNext": 1}),
        "h3": ParagraphStyle("h3", **{**base, "fontName": "Body-Bold", "fontSize": 11.5,
                                      "leading": 15, "spaceBefore": 9, "spaceAfter": 4,
                                      "keepWithNext": 1}),
        "body": ParagraphStyle("body", **{**base, "spaceAfter": 6}),
        "caption": ParagraphStyle("caption", **{**base, "fontSize": 8.6, "leading": 11.5, "textColor": MUTED,
                                                "spaceBefore": 4, "alignment": 1}),
        "bullet": ParagraphStyle("bullet", **{**base, "leftIndent": 14, "bulletIndent": 3,
                                              "spaceAfter": 3}),
        "cell": ParagraphStyle("cell", **{**base, "fontSize": 8.6, "leading": 11.5}),
        "cellh": ParagraphStyle("cellh", **{**base, "fontName": "Body-Bold", "fontSize": 8.6,
                                            "leading": 11.5}),
        "code": ParagraphStyle("code", fontName="Mono", fontSize=7.6, leading=10, textColor=INK),
    }


class MathRenderer:
    """LaTeX -> PNG via mathtext, cached by content; returns size and depth in points."""

    def __init__(self, workdir):
        self.workdir = workdir
        self.fontprop = font_manager.FontProperties(size=BODY_PT + 1)

    @staticmethod
    def to_mathtext(tex, display):
        tex = re.sub(r"\\le(?![a-zA-Z])", r"\\leq", tex)
        tex = re.sub(r"\\ge(?![a-zA-Z])", r"\\geq", tex)
        tex = tex.replace(r"\tfrac", r"\frac")
        tex = re.sub(r"\\(?:[bB]ig{1,2}[lrm]?|textstyle|displaystyle)(?![a-zA-Z])", "", tex)
        tex = re.sub(r"\\[lr]vert(?![a-zA-Z])", "|", tex)
        tex = re.sub(r"\\[lr]Vert(?![a-zA-Z])", r"\\|", tex)
        tex = re.sub(r"\\(?:text|operatorname)\{([^{}]*)\}",
                     lambda m: r"\mathrm{%s}" % m.group(1).replace(" ", r"\ "), tex)
        tex = re.sub(r"\\begin\{cases\}(.*?)\\end\{cases\}", MathRenderer._cases, tex, flags=re.S)
        if display:
            tex = re.sub(r"\\frac(?![a-zA-Z])", r"\\dfrac", tex)
        return " ".join(tex.split())

    @staticmethod
    def _cases(match):
        rows = [r.strip() for r in re.split(r"\\\\", match.group(1)) if r.strip()]
        if len(rows) != 2:
            raise ValueError("md_to_pdf supports two-row cases environments only")
        cells = [r.split("&") for r in rows]
        lines = [r"%s,\quad %s" % (c[0].strip().rstrip(","), c[1].strip()) for c in cells]
        return r"\left\lbrace\genfrac{}{}{0}{0}{%s}{%s}\right." % (lines[0], lines[1])

    def render(self, tex, display=False):
        mt = self.to_mathtext(tex, display)
        key = hashlib.sha1(f"{display}:{mt}".encode()).hexdigest()[:16]
        path = os.path.join(self.workdir, f"m_{key}.png")
        prop = font_manager.FontProperties(size=BODY_PT + (2.5 if display else 1))
        # Thin spaces on both sides keep the tight crop from clipping edge glyphs.
        depth = mathtext.math_to_image(rf"$\,{mt}\,\,$", path, prop=prop, dpi=DPI)
        h_px, w_px = imread(path).shape[:2]
        # math_to_image lays out at 72 dpi, so its depth is already in points.
        return path, w_px * 72 / DPI, h_px * 72 / DPI, depth or 0


class Converter:
    INLINE_MATH = re.compile(r"(?<![\\$])\$(?!\$)(.+?)(?<!\\)\$")

    def __init__(self, workdir, width, base_dir=".", max_image_height=13.5 * cm):
        self.math = MathRenderer(workdir)
        self.st = _styles()
        self.width = width
        self.base_dir, self.max_image_height = base_dir, max_image_height

    def inline(self, text):
        """Markdown inline markup -> reportlab paragraph XML."""
        slots = []

        def stash(xml):
            slots.append(xml)
            return f"\x00{len(slots) - 1}\x00"

        def math_img(m):
            path, w, h, depth = self.math.render(m.group(1))
            return stash(f'<img src="{path}" width="{w:.2f}" height="{h:.2f}" valign="{-depth:.2f}"/>')

        text = self.INLINE_MATH.sub(math_img, text)
        text = re.sub(r"`([^`]+)`", lambda m: stash(f'<font name="Mono" size="{BODY_PT - 1}">'
                                                    f"{escape(m.group(1))}</font>"), text)
        # anchors <a id="x"></a> (reference list) become PDF destinations; [t](#x) links to them
        text = re.sub(r'<a id="([^"]+)"></a>', lambda m: stash(f'<a name="{escape(m.group(1))}"/>'), text)

        def link(m):
            label, href = m.group(1), m.group(2)
            if href.startswith("#"):
                return stash(f'<a href="{escape(href)}" color="#0b6e8a">{escape(label)}</a>')
            if href.startswith("http"):
                return stash(f'<link href="{escape(href)}" color="#0b6e8a">{escape(label)}</link>')
            return stash(f'<font name="Mono" size="{BODY_PT - 1}">{escape(label)}</font>')
        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, text)
        text = escape(text)
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        while "\x00" in text:   # slots can nest (e.g. `code` inside a link)
            text = re.sub("\x00(\\d+)\x00", lambda m: slots[int(m.group(1))], text)
        return text

    def table(self, rows):
        header, body = rows[0], rows[2:]
        n = len(header)
        plain = [[re.sub(r"\$[^$]*\$", "xxxxx", re.sub(r"[`*]", "", c)) for c in r]
                 for r in [header] + body]
        # Every column is at least as wide as its longest word (so words never
        # split; a bracketed group such as an interval "[+0.56, +0.76]" counts as
        # one word and is kept on one line); the remaining width goes to columns in
        # proportion to text length.
        # A short cell (up to 10 characters, e.g. "98.5 M") is one word as well.
        words = lambda c: [c] if len(c) <= 10 else (re.findall(r"\[[^\]]*\]|\S+", c) or [""])  # noqa: E731
        floor = [max(pdfmetrics.stringWidth(w, "Body-Bold", 8.6) for r in plain for w in words(r[i])) + 9
                 for i in range(n)]
        keep = lambda c: (c.replace(" ", "\u00a0") if len(c) <= 10 else  # noqa: E731
                          re.sub(r"\[[^\]]*\]", lambda m: m.group(0).replace(" ", "\u00a0"), c))
        header = [keep(c) for c in header]
        body = [[keep(c) for c in r] for r in body]
        length = [max(len(r[i]) for r in plain) + 2 for i in range(n)]
        spare = self.width - sum(floor)
        if spare > 0:
            widths = [f + spare * l / sum(length) for f, l in zip(floor, length)]
        else:
            widths = [f * self.width / sum(floor) for f in floor]
        data = [[Paragraph(self.inline(c), self.st["cellh"]) for c in header]]
        data += [[Paragraph(self.inline(c), self.st["cell"]) for c in r] for r in body]
        t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), TINT),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
            ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        return [t, Spacer(1, 8)]

    def display_math(self, tex):
        tag = re.search(r"\\tag\{([^{}]*)\}", tex)
        if tag:
            tex = tex.replace(tag.group(0), "")
        path, w, h, _ = self.math.render(tex, display=True)
        room = self.width - (60 if tag else 12)
        scale = min(1.0, room / w)
        img = Image(path, width=w * scale, height=h * scale, hAlign="CENTER")
        if not tag:
            return [Spacer(1, 3), img, Spacer(1, 7)]
        row = Table([[img, Paragraph(f"({escape(tag.group(1))})", self.st["body"])]],
                    colWidths=[self.width - 48, 48])
        row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (0, 0), "CENTER"),
                                 ("ALIGN", (1, 0), (1, 0), "RIGHT"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                 ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
        return [Spacer(1, 3), row, Spacer(1, 7)]

    def image(self, caption, path, max_height):
        from reportlab.lib.utils import ImageReader
        iw, ih = ImageReader(path).getSize()
        scale = min(self.width / iw, max_height / ih)
        out = [Spacer(1, 4), Image(path, width=iw * scale, height=ih * scale, hAlign="CENTER")]
        if caption:
            out.append(Paragraph(self.inline(caption), self.st["caption"]))
        return [KeepTogether(out), Spacer(1, 8)]

    def code(self, lines):
        pre = XPreformatted(escape("\n".join(lines)), self.st["code"])
        box = Table([[pre]], colWidths=[self.width])
        box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), TINT),
                                 ("BOX", (0, 0), (-1, -1), 0.4, RULE),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 8),
                                 ("TOPPADDING", (0, 0), (-1, -1), 6),
                                 ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        return [box, Spacer(1, 8)]

    def convert(self, md):
        lines, story, i = md.splitlines(), [], 0
        while i < len(lines):
            line = lines[i]
            s = line.strip()
            if not s:
                i += 1
            elif s.startswith("```"):
                j = i + 1
                while not lines[j].strip().startswith("```"):
                    j += 1
                story += self.code(lines[i + 1:j])
                i = j + 1
            elif s.startswith("$$"):
                j, buf = i, []
                body = s[2:]
                if body.endswith("$$") and len(body) >= 2:
                    buf, j = [body[:-2]], i
                else:
                    buf = [body]
                    j = i + 1
                    while "$$" not in lines[j]:
                        buf.append(lines[j])
                        j += 1
                    buf.append(lines[j].split("$$")[0])
                story += self.display_math(" ".join(buf))
                i = j + 1
            elif re.match(r"!\[(.*)\]\(([^)]+)\)$", s):
                m = re.match(r"!\[(.*)\]\(([^)]+)\)$", s)
                story += self.image(m.group(1), os.path.join(self.base_dir, m.group(2)), self.max_image_height)
                i += 1
            elif s.startswith("#"):
                level = len(s) - len(s.lstrip("#"))
                style = {1: "title", 2: "h2"}.get(level, "h3")
                story.append(Paragraph(self.inline(s[level:].strip()), self.st[style]))
                i += 1
            elif s.startswith("|"):
                rows = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                    i += 1
                story += self.table(rows)
            elif s.startswith("- "):
                while i < len(lines) and lines[i].strip().startswith("- "):
                    item = [lines[i].strip()[2:]]
                    i += 1
                    while i < len(lines) and lines[i].startswith("  ") and lines[i].strip():
                        item.append(lines[i].strip())
                        i += 1
                    story.append(Paragraph(self.inline(" ".join(item)), self.st["bullet"],
                                           bulletText="•"))
                story.append(Spacer(1, 4))
            else:
                buf = []
                while i < len(lines) and lines[i].strip() and not re.match(
                        r"\s*(#|\||- |```|\$\$|!\[)", lines[i]):
                    buf.append(lines[i].strip())
                    i += 1
                para = Paragraph(self.inline(" ".join(buf)), self.st["body"])
                # a "**Table n.** caption" paragraph stays on the page of its table's first rows
                k = i
                while k < len(lines) and not lines[k].strip():
                    k += 1
                if buf[0].startswith("**Table") and k < len(lines) and lines[k].strip().startswith("|"):
                    rows = []
                    while k < len(lines) and lines[k].strip().startswith("|"):
                        rows.append([c.strip() for c in lines[k].strip().strip("|").split("|")])
                        k += 1
                    table = self.table(rows)
                    table[0].keepWithNext = 0
                    para.keepWithNext = 1
                    story += [para] + table
                    i = k
                else:
                    story.append(para)
        return story


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("markdown")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.splitext(args.markdown)[0] + ".pdf"
    _register_fonts()
    with open(args.markdown, encoding="utf-8") as fh:
        md = fh.read()
    title = re.search(r"^#\s+(.+)$", md, re.M).group(1)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Body", 7.5)
        canvas.setFillColor(MUTED)
        short = title if len(title) <= 80 else title[:77].rsplit(" ", 1)[0] + " ..."
        canvas.drawString(2 * cm, 1.2 * cm, f"LaBraM · {short}")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=2 * cm, title=title,
                            author="LaBraM brain-age project")
    with tempfile.TemporaryDirectory() as tmp:
        story = Converter(tmp, doc.width, os.path.dirname(os.path.abspath(args.markdown))).convert(md)
        doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
