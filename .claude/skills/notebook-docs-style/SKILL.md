---
name: notebook-docs-style
description: House style for analysis notebooks (notebooks/*.ipynb) and Markdown docs (docs/*.md) — every loss, metric and derived value written as a numbered LaTeX formula; every equation, table, figure and subfigure numbered, titled visibly and referenced from the text; at most two subplots side by side; notebooks exported to a PDF next to them (scripts/notebook_to_pdf.py). Load before writing or editing a notebook or a doc that reports results, metrics, losses or plots, or when asked for a PDF of a notebook.
---

# Notebook and docs style

Applies to every analysis notebook under `notebooks/` and every Markdown doc under
`docs/`. Reviewers read these as papers: a reader must be able to find the exact
definition of any number on the page, and to point at any panel by name.

## 1. Formulas: every loss, metric and derived value

* Each loss, metric, statistic or derived column a cell prints (a table column, an
  axis, a correction, a schedule) gets its definition as a LaTeX display formula
  in the Markdown next to it, **before** it is used. Name every symbol once, with
  its unit (years, Hz, µV, z-scored), in a "where" line or a symbol table.
* Write the formula the code computes, not the textbook one: same reduction
  (mean vs. sum), same unit (years vs. z-score), same set (windows vs. recordings),
  same estimator (sample vs. population std, `np.polyfit` OLS). Cite the function
  that implements it (`labram/eval/age_analysis.py::regression_summary`).
* For a table whose columns are metrics, give one numbered formula per column and
  a "column → equation" key, e.g. `mae` (2.1), `median_ae` (2.2).
* Display math is `$$ ... $$` on its own lines; inline symbols are `$...$`.
  Stick to the KaTeX/MathJax common subset (`\operatorname`, `\tfrac`, `\tag`,
  `\mathbb`, `\lvert ... \rvert`, `\begin{aligned}`) so it renders in VS Code,
  JupyterLab and GitHub alike. No `\label`/`\eqref` (GitHub ignores them); refer
  to an equation by its printed number.

## 2. Numbering: equations, tables, figures, subfigures

Numbers are **per section**, `‹section›.‹k›`, so adding an item renumbers only its
own section. Docs without numbered sections use their `##` order (1, 2, …).

| item | how it is numbered | how the text refers to it |
|---|---|---|
| equation | `\tag{2.3}` inside the `$$` block | "Eq. (2.3)" |
| table | its own **subsection heading** directly above it, one level below the enclosing heading: `### Table 2.1. <short title>`, then the description (columns, units, equations) as text under the heading. In a notebook the heading is a Markdown cell right before the code cell that displays the table (split the code cell so each table has its own cell); the displayed frame has **no** caption inside it (`numbered_table(df, "2.1", "…")` only formats it), so the title is never repeated in the table | "Table 2.1" |
| figure | like a table: its own **subsection heading** directly above it, `### Figure 3.1. <short title>`, with the panel description (what is plotted, units, split, equations) as text under it. In a notebook it is a Markdown cell right before the cell that draws the figure; the figure itself has **no** title (no suptitle). A figure drawn in a loop or by a helper gets its heading from `figure_heading(number, title, note, level)`, which displays it in the output right above the figure (`AgeExplorer.show/compare_at_age(..., number=..., heading_level=...)` do this) | "Figure 3.1" |
| subfigure | every panel's title starts with `(a)`, `(b)`, … left-to-right, top-to-bottom (`panel_labels(axes)`) | "Figure 3.1(b)" |

* Every numbered item is referenced at least once from the prose, and every
  reference resolves. No "the plot below", "top-left panel", "the table above".
* A figure produced in a loop gets consecutive numbers (`8.1 … 8.4`, one per
  loop item), each with its own displayed heading; the Markdown names the range.
* The text under a figure heading says what is plotted, the unit, the split/cohort and the
  `n`; the interpretation goes in the prose that cites it.

## 3. Figures: at most two subplots side by side

* No figure row holds more than **two** axes (`plt.subplots(r, c)` with
  `c <= 2`; a gridspec with at most two columns per row, a panel may span both).
  Three or more panels stack into more rows. Colorbars and legends are not axes
  for this rule.
* Size rows at about `figsize=(14, 4.5 * rows)` so two panels per row stay legible
  in a notebook at 110 dpi and in a PDF export.
* The figure helpers in `labram/eval/age_plots.py` (`AgeExplorer.show`,
  `AgeExplorer.compare_at_age`) follow this rule; keep new helpers to it too
  (`assert_max_columns(fig)` checks a figure).

## 4. Making the numbers visible in a notebook

The numbering only helps if a reader sees it in the rendered notebook, on GitHub
and in the PDF. Concretely (worked example: `notebooks/tuab_data_preparation.ipynb`):

* **Figure.** A Markdown heading cell `### Figure 4.2. <short title>` (one level
  below the enclosing heading) with the description under it — what is plotted,
  unit, cohort, n, one clause per panel: (a) …, (b) … — directly before the code
  cell. The figure has no suptitle. End every drawing cell the same way:

  ```python
  from labram.eval.age_plots import assert_max_columns, panel_labels

  def finish(fig, axes=None):
      fig.tight_layout()                       # lay out first: panel_labels sorts by position
      if axes is not None and len(axes) > 1:
          panel_labels(axes)                   # "(a) <title>", "(b) <title>", reading order
      assert_max_columns(fig)                  # house rule of section 3
      plt.show()
  ```

  A figure drawn in a loop calls `figure_heading("10.3", "<title>", note, level=4)`
  right before it is drawn, so its heading appears in the output above it; the
  static Markdown cell before the loop names the range ("Figures 10.1–10.6").
  Give every panel a short `ax.set_title(...)` *before* `finish` (it is
  prefixed, not replaced). Pass only data axes to `panel_labels` — never colorbar
  axes. A single-panel figure gets no letter. Set titles without `pad=`:
  `panel_labels` re-sets the title and drops it, so put in-axes annotations
  (patch indices, band names) inside the axes (`transform=ax.get_xaxis_transform()`,
  `y=0.97, va="top"`), not above them.
* **More than two panels.** Use `fig.add_gridspec(rows, 2)` and let the odd last
  panel span both columns (`gs[r, :]`); `assert_max_columns` counts it as one.
* **Table.** A Markdown heading cell `### Table 2.2. <title>` (one level below the
  enclosing heading) with the column key under it (column → Eq.), then a code cell
  that ends with `display(numbered_table(df, "2.2", "<title>"))`. Turn printed
  summaries into a `DataFrame` so they render as tables; keep `print` for log lines.
  Markdown tables (literature comparisons, step summaries) get the same heading.
* **Equation.** `$$ … \tag{s.k} $$`. Split anything wider than ~90 rendered
  characters into `\begin{aligned} … \\ … \end{aligned}` with one `\tag` after
  `\end{aligned}`, otherwise MathJax runs it off the PDF page and the tag is lost.
* **Cross-references** in prose and in code comments that implement a formula
  (`# Eq. (2.5)`), so a reader can go from a number on the page to its code.

## 5. PDF export of a notebook

Every analysis notebook ships a PDF next to it (`notebooks/<name>.pdf`), linked
from the first Markdown cell (`📄 **[PDF version](<name>.pdf)**`) and regenerated
whenever the saved outputs change. Use `scripts/notebook_to_pdf.py`:

```bash
# once per machine (not in requirements.txt)
.venv/bin/pip install "nbconvert[webpdf]" pypdfium2
.venv/bin/python -m playwright install chromium
sudo .venv/bin/python -m playwright install-deps chromium   # bare Ubuntu: libatk, libgbm, ...

.venv/bin/python scripts/notebook_to_pdf.py notebooks/<name>.ipynb --execute --pages /tmp/pages
```

* It runs nbconvert's `webpdf` exporter (headless Chromium prints the HTML view),
  so MathJax typesets the `\tag` numbers and Markdown/pandas tables keep their
  headings. No TeX install is needed, and Unicode (µ, →, emoji) just works.
* `--execute` does *Restart & Run All* in place first; without it the **saved**
  outputs are exported (save the notebook before exporting from inside Jupyter).
* Code is hidden (`--no-input`; `--with-code` keeps it). Cells tagged `pdf-skip`
  are dropped: tag the export/IFrame cells and anything meaningless on paper
  (`cell.metadata["tags"] = ["pdf-skip"]`).
* The `pdf_report` template (`scripts/nbconvert_templates/pdf_report/index.html.j2`)
  adds print CSS: compact 8 px tables so 10-column comparison tables fit the page,
  headings kept with what follows, figures not split across pages. Its rules sit
  in `html_head_css` with `!important` because JupyterLab's stylesheet is emitted
  after `extra_css`; the file must be called `index.html.j2` (that is what the
  webpdf exporter loads).
* **Look at the PDF before committing.** `--pages DIR` renders every page to PNG;
  check that no equation, table or figure is cut at the right edge, and that the
  `Figure s.k` / `Table s.k` headings and `(s.k)` tags are visible. A cut equation
  → split it (section 4); a cut table → shorten cell text or the column count.
* A notebook may also show its PDF inline in a `pdf-skip` cell:
  `IFrame("<name>.pdf", width="100%", height=800)` (JupyterLab; VS Code does not
  render it — the Markdown link works everywhere).

## 6. Checklist before committing a notebook or doc

1. Every metric/loss column and axis has a numbered formula above it.
2. Every equation, table, figure and panel has a number or letter, and the text
   cites each one; every table and every figure sits under its own
   `Table s.k. <title>` / `Figure s.k. <title>` heading, and neither carries a
   title inside it.
3. No figure row has more than two axes.
4. Notebook: *Restart & Run All* succeeds and the saved outputs show the
   numbered captions (`.venv/bin/jupyter nbconvert --to notebook --execute
   --inplace notebooks/<name>.ipynb`).
5. Prose numbers written by hand (e.g. "test MAE 8.45") match the executed outputs.
6. Notebook: the PDF is regenerated from the committed outputs
   (`scripts/notebook_to_pdf.py`), linked from the first cell, and its pages were
   looked at (`--pages`): nothing cut off at the page edge.
