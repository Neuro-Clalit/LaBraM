#!/usr/bin/env python
"""Bar chart of zero-shot brain-age MAE per external cohort (adults), from
``external_cohort_table.py``'s CSV: (a) every recording / session, then one panel per
label-free 20 % rejection score (``--reject-score``, default ``total`` and
``learned``; a single model such as E2L has only ``spread``). Bars are the models in a
fixed colour order, whiskers the 95 % subject-bootstrap interval, and the dashed
line over each cohort its predict-the-cohort-mean MAE (the floor to beat).
Every panel also shows the best published MAE per cohort (``SOTA``) as two
hatched bars in their own colours: a model trained on that cohort
(within-dataset CV) and a zero-shot one. The published values are on all
recordings of their cohort, also in the rejection panels. Temple / Harvard are the in-domain test split
(our models trained on the rest of those corpora), the others are zero-shot.

    python scripts/external_cohort_bars.py \\
        --table docs/data/external_eval/tdbrain/external_cohort_table.csv \\
        --out docs/figures/tdbrain/cohort_mae_bars.png
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

COHORTS = [("temple", "Temple (TUH)"), ("harvard", "Harvard (HEEDB)"), ("tdbrain", "TDBRAIN"),
           ("lemon", "LEMON"), ("chbp", "CHBP"), ("mipdb", "MIPDB")]
# best published MAE per cohort: (kind, MAE, source); docs/tdbrain_onboarding.md Tables 12.1, 13.1-13.4
SOTA = {
    "temple": [("within", 9.45, "Gemein 2024, TUAB eval (normal 6.60 / abnormal 12.85, weighted)")],
    "tdbrain": [("within", 5.96, "Khayretdinova 2022, ages 5-88"),
                ("zero-shot", 8.36, "Hajhassani 2026, healthy 20-90")],
    "lemon": [("within", 7.75, "Engemann 2022"), ("zero-shot", 9.91, "Hajhassani 2026")],
    "chbp": [("within", 6.48, "Engemann 2022")],
}
MODELS = [("E2L", "E2L"), ("P3ens", "P3 ensemble"), ("P4ens", "P4 ensemble")]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3, fixed order
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--table", required=True)
    ap.add_argument("--reject-score", nargs="+", default=["total", "learned"], choices=["spread", "total", "learned"],
                    help="one panel per rejection score (for the ensembles; E2L always uses spread)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from labram.eval.age_plots import assert_max_columns, panel_labels
    t = pd.read_csv(args.table)
    t = t[t["group"] == "all"]
    present = set(t["cohort"])
    cohorts = [(c, n) for c, n in COHORTS if c in present]
    panels = [("none", "every recording / session")] + [
        (f"reject 20% ({sc})", f"reject 20 % by confidence (ensembles: {sc}; E2L: spread)") for sc in args.reject_score]
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 4.75 * len(panels)), sharey=True, squeeze=False)
    axes = axes[:, 0]
    width = 0.165
    pub = [("within", "published, trained on the cohort", "#eda100"), ("zero-shot", "published, zero-shot", "#e87ba4")]
    def pick(c, m, rej):
        """Row of (cohort, model) under ``rej``; a single model has only the spread rejection."""
        r = t[(t.cohort == c) & (t.model == m) & (t.rejection == rej)]
        if r.empty and rej != "none":
            r = t[(t.cohort == c) & (t.model == m) & (t.rejection == "reject 20% (spread)")]
        return r.iloc[0]

    for ax, (rej, title) in zip(axes, panels):
        x = np.arange(len(cohorts))
        for j, ((m, label), color) in enumerate(zip(MODELS, COLORS)):
            rows = [pick(c, m, rej) for c, _ in cohorts]
            mae = np.array([r.mae for r in rows])
            err = np.array([[r.mae - r.mae_lo for r in rows], [r.mae_hi - r.mae for r in rows]])
            xs = x + (j - 2) * width
            ax.bar(xs, mae, width - 0.03, color=color, label=label, zorder=2)
            ax.errorbar(xs, mae, yerr=err, fmt="none", ecolor=INK, elinewidth=1, capsize=2.5, zorder=3)
            for xi, v in zip(xs, mae):
                ax.text(xi, 0.25, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color="white",
                        fontweight="bold", zorder=4)
        # best published MAE per cohort as two more bars (hatched: not ours, and on all recordings)
        for j, (kind, label, color) in enumerate(pub):
            pts = [(i, v) for i, (c, _) in enumerate(cohorts) for k, v, _src in SOTA.get(c, []) if k == kind]
            xs = np.array([i for i, _ in pts]) + (j + 1) * width
            vals = [v for _, v in pts]
            ax.bar(xs, vals, width - 0.03, color=color, hatch="///", edgecolor="white", linewidth=0, label=label, zorder=2)
            for xi, v in zip(xs, vals):
                ax.text(xi, 0.25, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK, fontweight="bold",
                        zorder=4, bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85))
        for i, (c, _) in enumerate(cohorts):
            r = pick(c, MODELS[-1][0], rej)
            ax.hlines(r.mae_predict_mean, i - 2.5 * width, i + 2.5 * width, colors=MUTED, linestyles="--",
                      lw=1.4, zorder=5, label="predict the cohort mean" if i == 0 else None)
        n = {c: pick(c, MODELS[-1][0], rej) for c, _ in cohorts}
        unit = {"tdbrain": "sessions"}
        domain = {"temple": "in-domain", "harvard": "in-domain"}
        ax.set_xticks(x, [f"{name}\n{domain.get(c, 'zero-shot')}, n = {int(n[c].n):,} {unit.get(c, 'rec.')}\nage {n[c].age_mean:.0f} y"
                          for c, name in cohorts], fontsize=9, color=INK)
        ax.set_title(title, fontsize=11, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(MUTED)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(colors=MUTED)
    for ax in axes:
        ax.set_ylabel("MAE (years), adults ≥ 18", color=INK)
    axes[0].set_ylim(0, 24)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right", ncol=3)
    fig.tight_layout()
    panel_labels(list(axes))
    assert_max_columns(fig)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(args.out)


if __name__ == "__main__":
    main()
