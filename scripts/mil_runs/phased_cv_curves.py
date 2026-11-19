#!/usr/bin/env python
"""Figures 13.2-13.4 of docs/mil_stage1_results.md: CV metrics over epochs.

For every model directory ``<root>/<MODEL>/fold_<k>/cv/fold_<k>/log.txt`` the
per-epoch recording MAE of the train-eval sample, the fold's val split and the
held-out test split, overall and per data cohort (``{split}_{group}_mae``,
``evaluation.per_dataset_metrics``), plus LEMON (``test_lemon_mae``). Each point
is the mean over the folds that logged the epoch, the error bar the sample SD
(ddof = 1). Writes one per-recipe figure (8 panels: all, Harvard normal /
abnormal / unlabelled, Temple normal / abnormal / all, LEMON), a recipe
comparison figure and the long-format CSV of every plotted point.

    PYTHONPATH=. python scripts/mil_runs/phased_cv_curves.py \\
        --root /data/runs/heedb/mil_phased_cv --models P3_L1=P3 P4_L1=P4 L1=P1 \\
        --figdir docs/figures --csv docs/data/mil_phased_cv/cv_curves.csv
"""
import argparse
import glob
import json
import os
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from labram.eval.age_plots import assert_max_columns, panel_labels  # noqa: E402

PANELS = [("all", "all recordings"), ("harvard_normal", "Harvard normal"),
          ("harvard_abnormal", "Harvard abnormal"), ("harvard_unlabelled", "Harvard unlabelled"),
          ("temple_normal", "Temple normal"), ("temple_abnormal", "Temple abnormal"),
          ("temple", "Temple, all"), ("lemon", "LEMON (zero-shot, test only)")]
SPLITS = [("train_eval", "train-eval", "tab:green"), ("val", "val", "tab:blue"), ("test", "test", "tab:red")]
RECIPE_COLORS = {"P1": "tab:gray", "P3": "tab:blue", "P4": "tab:orange"}


def metric_key(split, group):
    if group == "lemon":
        return "test_lemon_mae" if split == "test" else None
    return f"{split}_mae" if group == "all" else f"{split}_{group}_mae"


def load(root, model, label):
    rows = []
    for d in sorted(glob.glob(f"{root}/{model}/fold_*/cv/fold_*")):
        fold = int(re.search(r"fold_(\d)$", d).group(1))
        for line in open(f"{d}/log.txt"):
            r = json.loads(line)
            for split, _, _ in SPLITS:
                for group, _ in PANELS:
                    k = metric_key(split, group)
                    if k and r.get(k) is not None:
                        rows.append({"recipe": label, "model": model, "fold": fold, "epoch": r["epoch"],
                                     "split": split, "group": group, "mae": r[k]})
    return pd.DataFrame(rows)


def phases(root, model):
    import yaml
    cfg = yaml.safe_load(open(glob.glob(f"{root}/{model}/fold_0/cv/fold_0/run_config.yaml")[0]))
    mil = cfg.get("mil", {})
    w = mil.get("window_epochs", 0)
    return w, w + mil.get("freeze_encoder_epochs", 0)


def shade(ax, w, a):
    ax.axvspan(-0.5, w - 0.5, color="0.93", zorder=0)
    ax.axvspan(w - 0.5, a - 0.5, color="#fde7c8", zorder=0)


def summary(df):
    g = df.groupby(["recipe", "split", "group", "epoch"])["mae"]
    return g.agg(mean="mean", sd=lambda x: x.std(ddof=1), n="count").reset_index()


def recipe_figure(s, label, w, a, path):
    fig = plt.figure(figsize=(14, 4.5 * 4))
    gs = fig.add_gridspec(4, 2)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(len(PANELS))]
    for ax, (group, title) in zip(axes, PANELS):
        shade(ax, w, a)
        for split, name, color in SPLITS:
            q = s[(s.recipe == label) & (s.split == split) & (s.group == group)]
            if q.empty:
                continue
            ax.errorbar(q.epoch, q["mean"], yerr=q.sd.fillna(0), color=color, marker="o", ms=3.5,
                        capsize=3, lw=1.4, label=name)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("MAE (years)")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    panel_labels(axes)
    assert_max_columns(fig)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def compare_figure(s, labels, path):
    spec = [("test", "all", "test, all recordings"), ("val", "all", "val (own fold), all recordings"),
            ("train_eval", "all", "train-eval, all recordings"), ("test", "lemon", "LEMON (zero-shot)")]
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    axes = list(axs.flat)
    for ax, (split, group, title) in zip(axes, spec):
        for label in labels:
            q = s[(s.recipe == label) & (s.split == split) & (s.group == group)]
            ax.errorbar(q.epoch, q["mean"], yerr=q.sd.fillna(0), color=RECIPE_COLORS.get(label), marker="o",
                        ms=3.5, capsize=3, lw=1.4, label=label)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("MAE (years)")
        ax.grid(alpha=0.3)
    axes[0].set_ylim(5.9, 7.4)       # zoom on the end of training; the window
    axes[1].set_ylim(6.4, 7.9)       # phase starts far above (Figures 13.2-13.3)
    axes[3].set_ylim(4.5, 6.5)
    for ax in axes:
        ax.set_xlim(2.5, 16.5)
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    panel_labels(axes)
    assert_max_columns(fig)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--models", nargs="+", required=True, metavar="DIR=LABEL")
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    pairs = [m.split("=", 1) for m in args.models]
    df = pd.concat([load(args.root, d, lab) for d, lab in pairs], ignore_index=True)
    s = summary(df)
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    s.round(4).to_csv(args.csv, index=False)
    for d, lab in pairs:
        w, a = phases(args.root, d)
        recipe_figure(s, lab, w, a, f"{args.figdir}/mil_cv_curves_{lab.lower()}.png")
    compare_figure(s, [lab for _, lab in pairs], f"{args.figdir}/mil_cv_curves_compare.png")
    last = s[s.epoch == s.groupby("recipe").epoch.transform("max")]
    print(last.pivot_table(index=["split", "group"], columns="recipe", values="mean").round(2).to_string())


if __name__ == "__main__":
    main()
