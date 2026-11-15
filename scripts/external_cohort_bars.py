#!/usr/bin/env python
"""Bar chart of zero-shot brain-age MAE per external cohort (adults), from
``external_cohort_table.py``'s CSV: (a) every recording / session, (b) after the
label-free 20 % rejection by window-prediction spread. Bars are the models in a
fixed colour order, whiskers the 95 % subject-bootstrap interval, and the dashed
line over each cohort its predict-the-cohort-mean MAE (the floor to beat).

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

COHORTS = [("tdbrain", "TDBRAIN"), ("lemon", "LEMON"), ("chbp", "CHBP"), ("mipdb", "MIPDB")]
MODELS = [("E2L", "E2L"), ("P3ens", "P3 ensemble"), ("P4ens", "P4 ensemble")]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3, fixed order
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from labram.eval.age_plots import assert_max_columns, panel_labels
    t = pd.read_csv(args.table)
    t = t[t["group"] == "all"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    width = 0.26
    for ax, (rej, title) in zip(axes, [("none", "every recording / session"),
                                        ("reject 20%", "reject 20 % by prediction spread")]):
        x = np.arange(len(COHORTS))
        for j, ((m, label), color) in enumerate(zip(MODELS, COLORS)):
            rows = [t[(t.cohort == c) & (t.model == m) & (t.rejection == rej)].iloc[0] for c, _ in COHORTS]
            mae = np.array([r.mae for r in rows])
            err = np.array([[r.mae - r.mae_lo for r in rows], [r.mae_hi - r.mae for r in rows]])
            xs = x + (j - 1) * width
            ax.bar(xs, mae, width - 0.03, color=color, label=label, zorder=2)
            ax.errorbar(xs, mae, yerr=err, fmt="none", ecolor=INK, elinewidth=1, capsize=2.5, zorder=3)
            for xi, v in zip(xs, mae):
                ax.text(xi, 0.25, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color="white",
                        fontweight="bold", zorder=4)
        for i, (c, _) in enumerate(COHORTS):
            r = t[(t.cohort == c) & (t.model == "E2L") & (t.rejection == rej)].iloc[0]
            ax.hlines(r.mae_predict_mean, i - 1.5 * width, i + 1.5 * width, colors=MUTED, linestyles="--",
                      lw=1.4, zorder=5, label="predict the cohort mean" if i == 0 else None)
        n = {c: t[(t.cohort == c) & (t.model == "E2L") & (t.rejection == rej)].iloc[0] for c, _ in COHORTS}
        unit = {"tdbrain": "sessions"}
        ax.set_xticks(x, [f"{name}\nn = {int(n[c].n):,} {unit.get(c, 'rec.')}\nage {n[c].age_mean:.0f} y"
                          for c, name in COHORTS], fontsize=9, color=INK)
        ax.set_title(title, fontsize=11, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(MUTED)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(colors=MUTED)
    axes[0].set_ylabel("MAE (years), adults ≥ 18", color=INK)
    axes[0].set_ylim(0, 20)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    panel_labels(list(axes))
    assert_max_columns(fig)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(args.out)


if __name__ == "__main__":
    main()
