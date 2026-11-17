#!/usr/bin/env python
"""Predicted against true age with the units each rejection score sets aside.

For one cohort and one ensemble the three default label-free scores of
``rejection_scores.py`` (``spread``, ``total``, ``learned``: fitted on
validation folds of the training cohort only) each reject the ``--reject``
share of units with the highest score. The figure has one panel per score, the
kept units in grey and the rejected ones in the score's colour, and a last
panel coloured by how many of the scores reject the unit. The table
(``--out``) gives, per score, the MAE and mean age of the kept and of the
rejected units, and the overlap of the rejected sets (shared units, Jaccard
index).

Per-unit points are drawn, so the cohort must be one whose terms allow that
(``OPEN``: the CC0 / open cohorts). The ages only place the points and score
the two groups; the rejection itself never reads them.

    python scripts/rejection_scatter.py \\
        --set dortmund:<cohort>:<eval json>:<windows dir> --model P4ens \\
        --val-windows <val parquets> --out docs/data/external_eval/dortmund/rejection_overlap.csv \\
        --figure docs/figures/dortmund/dortmund_rejection_scatter.png
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from external_cohort_table import REJECT_LABEL, load_meta  # noqa: E402

OPEN = {"dortmund", "lemon", "chbp", "mipdb"}        # per-unit points may be published
COLORS = {"spread": "#2a78d6", "total": "#eb6834", "learned": "#1baf7a"}     # categorical slots 1-3
TITLE = {"spread": "spread of the window predictions", "total": "total uncertainty (spread + fold disagreement)",
         "learned": "learned score (fitted on validation folds)"}
COUNT_COLORS = {1: "#86b6ef", 2: "#3987e5", 3: "#104281"}                    # one hue, light to dark
KEPT, INK, MUTED, GRID = "#b0afa8", "#1f1f1e", "#6b6a64", "#e4e3df"


def rejected_sets(units, scores, reject):
    """{score: boolean Series} -- True for the ``reject`` share with the highest score."""
    out = {}
    for name in scores:
        col = REJECT_LABEL[name]
        if col in units:
            out[name] = units[col] > units[col].quantile(1 - reject)
    return out


def summary(units, rej):
    rows = []
    err = (units["pred"] - units["age"]).abs()
    for name, mask in rej.items():
        rows.append({"score": name, "kept": int((~mask).sum()), "rejected": int(mask.sum()),
                     "mae_kept": err[~mask].mean(), "mae_rejected": err[mask].mean(), "mae_all": err.mean(),
                     "age_kept": units.loc[~mask, "age"].mean(), "age_rejected": units.loc[mask, "age"].mean()})
    for a, b in itertools.combinations(rej, 2):
        both, either = (rej[a] & rej[b]).sum(), (rej[a] | rej[b]).sum()
        rows.append({"score": f"{a} & {b}", "rejected": int(both), "jaccard": both / either if either else np.nan})
    if len(rej) > 2:
        count = sum(m.astype(int) for m in rej.values())
        for k in range(len(rej) + 1):
            m = count == k
            rows.append({"score": f"rejected by {k} of {len(rej)} scores", "rejected": int(m.sum()),
                         "mae_rejected": err[m].mean() if m.any() else np.nan,
                         "age_rejected": units.loc[m, "age"].mean() if m.any() else np.nan})
    return pd.DataFrame(rows)


def panel(ax, title, lo, hi):
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls=":", zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("true age (y)")
    ax.set_ylabel("predicted age (y)")
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    ax.grid(color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def figure(units, rej, path, name, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    err = (units["pred"] - units["age"]).abs()
    lo = 5 * np.floor(min(units["age"].min(), units["pred"].min()) / 5) - 5
    hi = 5 * np.ceil(max(units["age"].max(), units["pred"].max()) / 5) + 5
    n_panels = len(rej) + (1 if len(rej) > 1 else 0)
    rows = (n_panels + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(12.5, 6.2 * rows), squeeze=False)
    axes = axes.ravel()
    for ax, tag, (score, mask) in zip(axes, "abcdef", rej.items()):
        panel(ax, f"({tag}) {TITLE[score]}", lo, hi)
        ax.scatter(units.loc[~mask, "age"], units.loc[~mask, "pred"], s=10, color=KEPT, alpha=0.55, lw=0, zorder=2,
                   label=f"kept ({int((~mask).sum())}): MAE {err[~mask].mean():.2f} y")
        ax.scatter(units.loc[mask, "age"], units.loc[mask, "pred"], s=16, color=COLORS[score], alpha=0.9, lw=0,
                   zorder=3, label=f"rejected ({int(mask.sum())}): MAE {err[mask].mean():.2f} y")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    if len(rej) > 1:
        ax = axes[len(rej)]
        count = sum(m.astype(int) for m in rej.values())
        panel(ax, f"({'abcdef'[len(rej)]}) how many of the {len(rej)} scores reject the unit", lo, hi)
        ax.scatter(units.loc[count == 0, "age"], units.loc[count == 0, "pred"], s=10, color=KEPT, alpha=0.55, lw=0,
                   zorder=2, label=f"none ({int((count == 0).sum())}): MAE {err[count == 0].mean():.2f} y")
        for k in range(1, len(rej) + 1):
            m = count == k
            if m.any():
                ax.scatter(units.loc[m, "age"], units.loc[m, "pred"], s=16, color=COUNT_COLORS[min(k, 3)], alpha=0.9,
                           lw=0, zorder=2 + k, label=f"{k} ({int(m.sum())}): MAE {err[m].mean():.2f} y")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    fig.suptitle(f"{name}, {model}: units set aside by each label-free rejection score", fontsize=12, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=170)
    print(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", required=True, metavar="NAME:COHORT:EVAL.json:WINDOWS_DIR[:FAMILY]")
    ap.add_argument("--model", default="P4ens")
    ap.add_argument("--scores", nargs="+", default=["spread", "total", "learned"], choices=sorted(REJECT_LABEL))
    ap.add_argument("--val-windows", default="", metavar="PARQUET[,PARQUET...]",
                    help="validation-fold window parquets of the model's fold models (needed for 'learned')")
    ap.add_argument("--reject", type=float, default=0.2)
    ap.add_argument("--min-age", type=float, default=18.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figure", required=True)
    args = ap.parse_args()
    name, cdir, ev, wdir, *family = args.set.split(":")
    if name not in OPEN:
        raise SystemExit(f"{name}: per-unit points are only drawn for the open cohorts {sorted(OPEN)} "
                         "(data use agreements of the others allow aggregates only)")
    import rejection_scores as rs
    meta = load_meta(cdir, family[0] if family else None, args.min_age)
    members = json.load(open(ev))[args.model].get("members", [args.model])
    val_model = rs.fit_val_model(args.val_windows.split(","), args.min_age)[0] if args.val_windows else None
    units = rs.add_cohort_scores(rs.cohort_units(meta, wdir, members, val_model))
    rej = rejected_sets(units, args.scores, args.reject)
    table = summary(units, rej)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    table.to_csv(args.out, index=False)
    with pd.option_context("display.width", 220):
        print(table.round(3).to_string(index=False))
    figure(units, rej, args.figure, name.capitalize(), args.model)


if __name__ == "__main__":
    main()
