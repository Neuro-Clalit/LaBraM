#!/usr/bin/env python
"""Learning curves of the long warm-start MIL runs (X1-X6) against the 4-epoch ones (W4, L1, W7).

Reads ``docs/data/mil_x/fit_curves.csv`` (job, epoch, Val, Test, TrainEval, gap;
recording MAE in years, parsed from the training logs) and draws val, test,
train-eval and the gap val - train-eval per epoch.

    python scripts/mil_x_fit_plot.py --csv docs/data/mil_x/fit_curves.csv \\
        --out docs/figures/mil_x_fit_curves.png
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from labram.eval.age_plots import assert_max_columns, panel_labels  # noqa: E402

STYLE = {"X1": ("C0", "-"), "X3": ("C1", "-"), "X2 (SageMaker)": ("C2", "-"), "X2 (local)": ("C2", ":"),
         "X4": ("C3", "-"), "X5": ("C4", "-"), "X6": ("C5", "-"),
         "W4": ("C0", "--"), "L1": ("C2", "--"), "W7": ("C1", "--")}
LABEL = {"X1": "X1 gated + retention", "X3": "X3 gated hq + retention", "X2 (SageMaker)": "X2 ungated (SageMaker)",
         "X2 (local)": "X2 ungated (local)", "X4": "X4 ungated hq", "X5": "X5 bags 8-32 (running)",
         "X6": "X6 bags 8-64 (running)", "W4": "W4 (4 epochs)", "L1": "L1 (4 epochs)", "W7": "W7 (4 epochs)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = pd.read_csv(args.csv)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panels = [("Val", "val recording MAE (y)"), ("TrainEval", "train-eval recording MAE (y)"),
              ("gap", "gap: val − train-eval (y)"), ("Test", "test recording MAE (y)")]
    for ax, (col, ylabel) in zip(axes.flat, panels):
        for job, (c, ls) in STYLE.items():
            g = d[d.job == job].sort_values("epoch")
            if g.empty:
                continue
            ax.plot(g.epoch, g[col], ls=ls, color=c, marker="o", ms=3, lw=1.4, label=LABEL[job])
        ax.axvspan(-0.5, 3.5, color="0.85", alpha=0.4, lw=0)
        ax.set_xlabel("epoch (shaded: encoder frozen in the X runs)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split(" (")[0])
        if col in ("Val", "Test"):
            ax.set_ylim(top=min(ax.get_ylim()[1], 6.95))
    axes[0][0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    panel_labels(list(axes.flat))
    assert_max_columns(fig)
    fig.savefig(args.out, dpi=120)
    print(args.out)


if __name__ == "__main__":
    main()
