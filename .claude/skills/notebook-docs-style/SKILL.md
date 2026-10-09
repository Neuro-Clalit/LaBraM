---
name: notebook-docs-style
description: House style for analysis notebooks (notebooks/*.ipynb) and Markdown docs (docs/*.md) — every loss, metric and derived value written as a numbered LaTeX formula; every equation, table, figure and subfigure numbered and referenced from the text; at most two subplots side by side. Load before writing or editing a notebook or a doc that reports results, metrics, losses or plots.
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
| table | caption **above** it: `**Table 2.1.** what it shows.` In a notebook the displayed frame carries the caption itself (`numbered_table(df, "2.1", "…")`, a pandas Styler caption) | "Table 2.1" |
| figure | `**Figure 3.1.** …` caption in the Markdown, and the same `Figure 3.1` as the figure's suptitle (`figure_caption(fig, "3.1", "…")`) | "Figure 3.1" |
| subfigure | every panel's title starts with `(a)`, `(b)`, … left-to-right, top-to-bottom (`panel_labels(axes)`) | "Figure 3.1(b)" |

* Every numbered item is referenced at least once from the prose, and every
  reference resolves. No "the plot below", "top-left panel", "the table above".
* A figure produced in a loop gets consecutive numbers (`8.1 … 8.4`, one per
  loop item); the Markdown names the range.
* A Markdown caption says what is plotted, the unit, the split/cohort and the
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

## 4. Checklist before committing a notebook or doc

1. Every metric/loss column and axis has a numbered formula above it.
2. Every equation, table, figure and panel has a number or letter, and the text
   cites each one.
3. No figure row has more than two axes.
4. Notebook: *Restart & Run All* succeeds and the saved outputs show the
   numbered captions (`.venv/bin/jupyter nbconvert --to notebook --execute
   --inplace notebooks/<name>.ipynb`).
5. Prose numbers written by hand (e.g. "test MAE 8.45") match the executed outputs.
