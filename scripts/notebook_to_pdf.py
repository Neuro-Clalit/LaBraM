#!/usr/bin/env python
"""Export an analysis notebook to PDF next to it, the way a reader sees it.

Uses nbconvert's ``webpdf`` exporter: the notebook's HTML view is printed by
headless Chromium, so MathJax typesets every ``$$ ... \\tag{s.k} $$`` formula
with its number, Markdown/pandas tables keep their ``Table s.k`` headings and
figures keep their ``Figure s.k`` titles. The ``pdf_report`` template
(``scripts/nbconvert_templates``) adds print CSS so wide tables fit the page.
By default code cells are hidden (``--no-input``) and cells tagged ``pdf-skip``
are dropped.

    pip install "nbconvert[webpdf]" pypdfium2      # not in requirements.txt
    python -m playwright install chromium          # once; on a bare Ubuntu host also
    sudo python -m playwright install-deps chromium  # (system libraries: libatk, libgbm, ...)

    python scripts/notebook_to_pdf.py notebooks/x.ipynb                  # -> notebooks/x.pdf
    python scripts/notebook_to_pdf.py notebooks/x.ipynb --execute        # run all, save, export
    python scripts/notebook_to_pdf.py notebooks/x.ipynb --with-code --pages /tmp/pages
"""
import argparse
import os
import subprocess
import sys

SKIP_TAG = "pdf-skip"
TEMPLATE = "pdf_report"   # scripts/nbconvert_templates/pdf_report: print CSS (compact tables, margins)


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook")
    ap.add_argument("--execute", action="store_true",
                    help="Restart & Run All in place first (saves the outputs into the notebook)")
    ap.add_argument("--with-code", action="store_true", help="keep the code cells in the PDF")
    ap.add_argument("--timeout", type=int, default=3600, help="per-cell timeout for --execute, s")
    ap.add_argument("--pages", metavar="DIR",
                    help="also render every PDF page to DIR/page_NNN.png for a visual check (needs pypdfium2)")
    args = ap.parse_args()

    nb = os.path.abspath(args.notebook)
    nbconvert = [sys.executable, "-m", "nbconvert"]
    if args.execute:
        run(nbconvert + ["--to", "notebook", "--execute", "--inplace",
                         f"--ExecutePreprocessor.timeout={args.timeout}", nb])

    templates = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nbconvert_templates")
    cmd = nbconvert + ["--to", "webpdf", nb,
                       f"--TemplateExporter.extra_template_basedirs={templates}", "--template", TEMPLATE,
                       "--TagRemovePreprocessor.enabled=True",
                       f"--TagRemovePreprocessor.remove_cell_tags={SKIP_TAG}"]
    if not args.with_code:
        cmd.append("--no-input")
    run(cmd)
    pdf = os.path.splitext(nb)[0] + ".pdf"

    try:
        import pypdfium2 as pdfium
    except ImportError:
        print(f"wrote {pdf}")
        return
    doc = pdfium.PdfDocument(pdf)
    print(f"wrote {pdf}: {len(doc)} pages, {os.path.getsize(pdf) / 1e6:.1f} MB")
    if args.pages:
        os.makedirs(args.pages, exist_ok=True)
        for i, page in enumerate(doc):
            page.render(scale=1.0).to_pil().save(os.path.join(args.pages, f"page_{i + 1:03d}.png"))
        print(f"rendered {len(doc)} pages to {args.pages}/")


if __name__ == "__main__":
    main()
