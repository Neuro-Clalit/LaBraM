#!/usr/bin/env python
"""Figures of docs/fm_encoder_summary.md, from the aggregate tables in docs/data/fm_encoders/.

Figure 3.1  paired MAE difference against P4 per cohort, (a) CBraMod, (b) REVE
Figure 5.1  brain-age gap per pathology group on the Harvard recordings, adjusted for age band
            and sex against the healthy reference, (a) in years, (b) in units of the reference
            group's spread; P4, CBraMod and REVE

    python scripts/fm_summary_figures.py
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from labram.eval.age_plots import assert_max_columns, panel_labels  # noqa: E402

INK, MUTED, GRID = "#191919", "#666666", "#e4e3df"
# reference palette, categorical slots 1-3 (fixed per model in every figure)
COLOR = {"REVE": "#257de0", "CBraMod": "#f47230", "P4": "#00b894"}
THRESHOLD = 0.38            # years: single-pair threshold of the decision rule
COHORTS = [("test", "Test, all"), ("test_harvard", "Test, Harvard"), ("test_temple", "Test, Temple"),
           ("lemon", "LEMON"), ("dortmund", "Dortmund"), ("tdbrain", "TDBRAIN"), ("srm", "SRM"), ("chbp", "CHBP")]
LABEL = {"clinical_seizure": "clinical seizures", "electrographic_seizure": "electrographic seizures",
         "epilepsy": "epilepsy", "stroke": "stroke", "depression": "depression", "dementia": "dementia",
         "ms": "multiple sclerosis", "other": "other diagnosis", "headache_disorders": "headache / migraine",
         "sleep_disorders": "sleep disorders", "intracranial_and_spinal_tumors": "CNS tumours",
         "movement_and_cerebellar_disorders": "movement disorders"}


def style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", length=0, labelsize=12, colors=INK)
    ax.tick_params(axis="x", labelsize=11, colors=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)


def delta_per_cohort(data, out):
    tables = {"CBraMod": pd.read_csv(os.path.join(data, "cb5_vs_p4.csv")).set_index("cohort"),
              "REVE": pd.read_csv(os.path.join(data, "rv5_vs_p4.csv")).set_index("cohort")}
    lo = min(t.loc[[c for c, _ in COHORTS], "lo"].min() for t in tables.values())
    hi = max(t.loc[[c for c, _ in COHORTS], "hi"].max() for t in tables.values())
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    y = np.arange(len(COHORTS))[::-1]
    for ax, (model, t) in zip(axes, tables.items()):
        r = t.loc[[c for c, _ in COHORTS]]
        ax.axvline(0, color=MUTED, lw=1.2, zorder=1)
        for x in (-THRESHOLD, THRESHOLD):
            ax.axvline(x, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.hlines(y, r["lo"], r["hi"], color=COLOR[model], lw=2, zorder=2)
        ax.scatter(r["d"], y, s=64, color=COLOR[model], edgecolor="white", linewidth=1.5, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{name} ({int(n):,})" for (_, name), n in zip(COHORTS, r["n"])])
        ax.set_xlim(lo - 0.25, hi + 0.25)
        ax.set_xlabel("Δ MAE against P4 (years)", color=INK, fontsize=12)
        ax.text(THRESHOLD, 1.0, " +0.38", transform=ax.get_xaxis_transform(), va="top", ha="left",
                fontsize=10.5, color=MUTED)
        ax.set_title(f"{model} ({'CB5_L1' if model == 'CBraMod' else 'RV5_L1'})", fontsize=13, color=INK)
        style(ax)
    fig.tight_layout()
    panel_labels(list(axes))
    assert_max_columns(fig)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def pathology_gaps(data, out):
    p = pd.read_csv(os.path.join(data, "pathology_fold0.csv"))
    p = p[(p.source == "pooled") & (p.family == "harvard") & (p.kind == "pathology") & (p.suppressed == False)]  # noqa: E712
    p = p.sort_values("gapadj_P4")
    models = ["P4", "CBraMod", "REVE"]
    offset = {"P4": 0.22, "CBraMod": 0.0, "REVE": -0.22}
    y = np.arange(len(p))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 7.2), sharey=True)
    for ax, scaled in zip(axes, (False, True)):
        ax.axvline(0, color=MUTED, lw=1.2, zorder=1)
        for m in models:
            yy = y + offset[m]
            if scaled:
                ax.scatter(p[f"gapadj_sd_{m}"], yy, s=48, color=COLOR[m], edgecolor="white", linewidth=1.2,
                           zorder=3, label=m)
            else:
                ax.hlines(yy, p[f"gapadj_lo_{m}"], p[f"gapadj_hi_{m}"], color=COLOR[m], lw=2, zorder=2)
                ax.scatter(p[f"gapadj_{m}"], yy, s=48, color=COLOR[m], edgecolor="white", linewidth=1.2,
                           zorder=3, label=m)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{LABEL.get(g, g)} ({int(n):,})" for g, n in zip(p["group"], p["n"])])
        if scaled:
            ax.set_xlabel("adjusted gap / reference spread", color=INK, fontsize=12)
            ax.set_title("Scaled, Eq. (5.4)", fontsize=13, color=INK)
        else:
            ax.set_xlabel("adjusted brain-age gap (years)", color=INK, fontsize=12)
            ax.set_title("Years, 95 % intervals, Eq. (5.3)", fontsize=13, color=INK)
        style(ax)
    axes[1].legend(frameon=False, fontsize=11.5, loc="lower right", title=None)
    fig.tight_layout()
    panel_labels(list(axes))
    assert_max_columns(fig)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="docs/data/fm_encoders")
    ap.add_argument("--out", default="docs/figures/fm_encoders")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    delta_per_cohort(args.data, os.path.join(args.out, "delta_per_cohort.png"))
    pathology_gaps(args.data, os.path.join(args.out, "pathology_gap_harvard.png"))
    print("written:", os.listdir(args.out))


if __name__ == "__main__":
    main()
