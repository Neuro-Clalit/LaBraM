#!/usr/bin/env python
"""Build the brain-age technical report from one source into LaTeX, Markdown and PDF.

Source: ``docs/tech_report/brain_age_tech_report.src.md`` -- Markdown with LaTeX
math, written in a small fixed subset:

* ``# Title``, ``## 1. Section``, ``### 1.1. Subsection``;
* paragraphs, ``- `` bullets, ``**bold**``, `` `code` ``, inline ``$...$``;
* display math on one line, ``$$ ... \\tag{n} $$`` (mathtext-compatible LaTeX);
* citations ``[@key]`` / ``[@key1; @key2]``, keys of ``refs.bib``;
* a table = a paragraph starting ``**Table n.**`` (its caption) followed by a
  pipe table;
* a figure = ``![**Figure n.** caption](path)`` (path relative to the source).

Outputs next to the source:

* ``brain_age_tech_report.tex`` -- LaTeX (article, natbib, booktabs), citing
  ``refs.bib`` with ``\\citep``; compile with ``pdflatex`` + ``bibtex`` or on Overleaf;
* ``brain_age_tech_report.md`` -- the same text with numbered citations and a
  reference list formatted from ``refs.bib``;
* ``brain_age_tech_report.pdf`` -- from that Markdown through
  ``scripts/md_to_pdf.py`` (matplotlib mathtext + reportlab; no TeX needed).

    python scripts/build_tech_report.py
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.join(HERE, "..", "docs", "tech_report", "brain_age_tech_report.src.md")

UNICODE_TEX = {"−": "$-$", "–": "--", "±": "$\\pm$", "×": "$\\times$", "≤": "$\\le$", "≥": "$\\ge$",
               "≈": "$\\approx$", "µ": "$\\mu$", "²": "$^2$", "→": "$\\rightarrow$", "ρ": "$\\rho$",
               "δ": "$\\delta$", "α": "$\\alpha$", "λ": "$\\lambda$", "σ": "$\\sigma$", "’": "'", "“": "``", "”": "''"}


# ------------------------------------------------------------------ bibliography
def parse_bib(path):
    """{key: {type, field: value}} for the simple entries of refs.bib."""
    text = open(path, encoding="utf-8").read()
    out = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        fields = {"type": m.group(1).lower()}
        for f in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*(?=\n\s*\w+\s*=|\n?$)", m.group(3), re.S):
            fields[f.group(1).lower()] = " ".join(f.group(2).split())
        out[m.group(2).strip()] = fields
    return out


def detex(s):
    """BibTeX text -> plain Unicode for the Markdown reference list."""
    for a, b in ((r'{\"o}', "ö"), (r"{\'e}", "é"), (r"{\'a}", "á"), (r"{\~n}", "ñ"), (r"{\o}", "ø")):
        s = s.replace(a, b)
    return re.sub(r"[{}]", "", s)


def format_authors(author):
    names = [n.strip() for n in detex(author).split(" and ")]
    out = []
    for n in names:
        if n == "others":
            out.append("et al.")
            continue
        last, _, first = n.partition(",")
        out.append(f"{first.strip()} {last.strip()}".strip())
    return ", ".join(out).rstrip(".")


def format_reference(e):
    parts = []
    if "author" in e:
        parts.append(format_authors(e["author"]) + ".")
    parts.append(detex(e["title"]) + ".")
    venue = e.get("journal") or e.get("booktitle") or e.get("howpublished")
    if venue:
        v = detex(venue)
        if "volume" in e:
            v += f" {e['volume']}" + (f"({e['number']})" if "number" in e else "")
        if "pages" in e:
            v += f":{e['pages'].replace('--', '–')}" if "volume" in e else f", pp. {e['pages'].replace('--', '–')}"
        parts.append(v + ",")
    parts.append(e.get("year", "") + ".")
    if "doi" in e:
        parts.append(f"doi:{e['doi']}.")
    if "note" in e:
        parts.append(detex(e["note"]) + ".")
    return " ".join(parts)


# ------------------------------------------------------------------ Markdown with numbers
CITE = re.compile(r"\[@([^\]]+)\]")


def cite_keys(m):
    return [k.strip().lstrip("@") for k in m.group(1).split(";")]


def to_markdown(src, bib):
    order = []
    for m in CITE.finditer(src):
        for k in cite_keys(m):
            if k not in bib:
                raise SystemExit(f"citation {k!r} is not in refs.bib")
            if k not in order:
                order.append(k)
    number = {k: i + 1 for i, k in enumerate(order)}
    md = CITE.sub(lambda m: "[" + ", ".join(str(number[k]) for k in cite_keys(m)) + "]", src)
    refs = ["", "## References", ""] + [f"[{number[k]}] {format_reference(bib[k])}\n" for k in order]
    unused = sorted(set(bib) - set(order))
    return md.rstrip("\n") + "\n" + "\n".join(refs), order, unused


# ------------------------------------------------------------------ LaTeX
def tex_text(s):
    """Plain text (no math, no code) -> LaTeX."""
    s = s.replace("\\", "\\textbackslash{}")
    for ch in "%&#_":
        s = s.replace(ch, "\\" + ch)
    for a, b in UNICODE_TEX.items():
        s = s.replace(a, b)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    return s


def tex_inline(s):
    out = []
    for part in re.split(r"(\$[^$]+\$|`[^`]+`|\[@[^\]]+\])", s):
        if not part:
            continue
        if part.startswith("$"):
            out.append(part)
        elif part.startswith("`"):
            out.append("\\texttt{" + tex_text(part[1:-1]).replace("\\textbf", "") + "}")
        elif part.startswith("[@"):
            out.append("\\citep{" + ",".join(k.strip().lstrip("@") for k in part[1:-1].split(";")) + "}")
        else:
            out.append(tex_text(part))
    return "".join(out).replace(" \\citep", "~\\citep")


def tex_table(rows, caption):
    header, body = rows[0], rows[2:]
    n = len(header)
    lines = ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{tex_inline(caption)}}}"]
    tab = [f"\\begin{{tabular}}{{{'l' * n}}}", "\\toprule",
           " & ".join("\\textbf{%s}" % tex_inline(c) for c in header) + " \\\\", "\\midrule"]
    tab += [" & ".join(tex_inline(c) for c in r) + " \\\\" for r in body]
    tab += ["\\bottomrule", "\\end{tabular}"]
    lines += ["\\resizebox{\\textwidth}{!}{%"] + tab + ["}"] if n > 4 else tab
    return lines + ["\\end{table}", ""]


def next_block_is_table(lines, i):
    while i < len(lines) and not lines[i].strip():
        i += 1
    return i < len(lines) and lines[i].strip().startswith("|")


def to_tex(src):
    lines, out, i = src.splitlines(), [], 0
    title = "Technical report"
    in_list = False

    def close_list():
        nonlocal in_list
        if in_list:
            out.append("\\end{itemize}")
            in_list = False

    while i < len(lines):
        s = lines[i].strip()
        if not s:
            close_list()
            out.append("")
            i += 1
        elif s.startswith("# "):
            title = s[2:]
            i += 1
        elif s.startswith("#"):
            close_list()
            level = len(s) - len(s.lstrip("#"))
            name = re.sub(r"^\d+(\.\d+)*\.\s*", "", s[level:].strip())
            out.append({2: "\\section", 3: "\\subsection"}.get(level, "\\paragraph") + "{" + tex_inline(name) + "}")
            i += 1
        elif s.startswith("$$"):
            close_list()
            body = s.strip("$").strip()
            out += ["\\begin{equation}", body, "\\end{equation}"]
            i += 1
        elif s.startswith("!["):
            close_list()
            m = re.match(r"!\[(.*)\]\(([^)]+)\)$", s)
            cap = re.sub(r"^\*\*Figure \d+\.\*\*\s*", "", m.group(1))
            out += ["\\begin{figure}[t]", "\\centering",
                    f"\\includegraphics[width=\\textwidth,height=0.62\\textheight,keepaspectratio]{{{m.group(2)}}}",
                    f"\\caption{{{tex_inline(cap)}}}", "\\end{figure}", ""]
            i += 1
        elif s.startswith("**Table") and next_block_is_table(lines, i + 1):
            close_list()
            cap = re.sub(r"^\*\*Table \d+\.\*\*\s*", "", s)
            i += 1
            while not lines[i].strip():
                i += 1
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            out += tex_table(rows, cap)
        elif s.startswith("|"):
            raise SystemExit(f"line {i + 1}: a table needs a '**Table n.** caption' paragraph before it")
        elif s.startswith("- "):
            if not in_list:
                out.append("\\begin{itemize}")
                in_list = True
            item = [s[2:]]
            i += 1
            while i < len(lines) and lines[i].startswith("  ") and lines[i].strip():
                item.append(lines[i].strip())
                i += 1
            out.append("\\item " + tex_inline(" ".join(item)))
        else:
            close_list()
            buf = []
            while i < len(lines) and lines[i].strip() and not re.match(r"\s*(#|\||- |\$\$|!\[)", lines[i]):
                buf.append(lines[i].strip())
                i += 1
            if not buf:
                raise SystemExit(f"line {i + 1}: cannot convert {s[:60]!r}")
            out.append(tex_inline(" ".join(buf)))
    close_list()
    pre = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=2.3cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[numbers,sort&compress]{natbib}
\usepackage[hidelinks]{hyperref}
\setlength{\parskip}{0.5em}
\setlength{\parindent}{0pt}
% Generated by scripts/build_tech_report.py from brain_age_tech_report.src.md -- edit the source, not this file.
% Table and figure numbers in the text are written out ("Table 3"); floats are numbered in the same order.
"""
    return (pre + "\\title{" + tex_inline(title) + "}\n\\author{LaBraM brain-age project}\n\\date{Working draft}\n"
            "\\begin{document}\n\\maketitle\n\n" + "\n".join(out)
            + "\n\n\\bibliographystyle{unsrtnat}\n\\bibliography{refs}\n\\end{document}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    src_path = os.path.abspath(args.src)
    base = src_path[:-len(".src.md")]
    src = open(src_path, encoding="utf-8").read()
    bib = parse_bib(os.path.join(os.path.dirname(src_path), "refs.bib"))
    md, order, unused = to_markdown(src, bib)
    open(base + ".md", "w", encoding="utf-8").write(md)
    open(base + ".tex", "w", encoding="utf-8").write(to_tex(src))
    print(f"{len(order)} references cited; not cited: {unused or 'none'}")
    print(base + ".md\n" + base + ".tex")
    if not args.no_pdf:
        subprocess.run([sys.executable, os.path.join(HERE, "md_to_pdf.py"), base + ".md"], check=True)


if __name__ == "__main__":
    main()
