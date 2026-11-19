#!/usr/bin/env python
"""Explanatory figures of the technical report (``docs/tech_report/``).

* ``fig_pipeline.png``: (a) the LaBraM pre-training that is reused unchanged
  (neural tokenizer, masked EEG modelling) and (b) the recording-level
  brain-age pipeline of this work built on its encoder (Sections 4.1-4.5).
* ``fig_attention_example.png``: attention pooling over the windows of one
  recording of an open cohort (LEMON, CC BY), P4 ensemble: window predictions
  and relative attention weights over time, weight against the deviation from
  the pooled prediction, and the EEG of the highest- and the lowest-weighted
  window. Ensemble window quantities as Eq. (15) of the report: ``p_i`` the
  fold mean of the window predictions, ``w_i = (n/K) sum_k alpha_ik``; the
  pooled prediction is the fold mean of ``sum_i alpha_ik p_ik``.

    python scripts/tech_report_figures.py --out-dir docs/tech_report \\
        --windows /data/runs/heedb/external_eval/windows/lemon_all_models \\
        --cohort /data/datasets/EEG-public/LEMON/cohorts/lemon_v1 --recording sub-032490_lemon-rs --trim-windows 6
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"
SPACING = 80                                                                 # uV between EEG traces
CLIP = SPACING / 2                                                           # uV, display clip: traces never overlap
STYLE = {  # fill, edge
    "data": ("#efeeea", "#6b6a64"),
    "labram": ("#dbe8f7", "#2a78d6"),
    "ours": ("#fbe3d6", "#eb6834"),
    "note": ("#ffffff", "#6b6a64"),
}


def box(ax, x, y, w, h, title, body, kind):
    from matplotlib.patches import FancyBboxPatch
    fill, edge = STYLE[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fill, ec=edge,
                                lw=1.6, ls="--" if kind == "note" else "-", zorder=2))
    ax.text(x + w / 2, y + h - 1.3, title, ha="center", va="top", fontsize=9.4, fontweight="bold", color=INK, zorder=3)
    ax.text(x + w / 2, y + h - 4.4, body, ha="center", va="top", fontsize=8.4, color=INK, linespacing=1.3, zorder=3)


def arrow(ax, x0, y0, x1, y1, label=None, color=INK, rad=0.0):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13, lw=1.4, color=color,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))
    if label:
        ax.text((x0 + x1) / 2 + 0.8, (y0 + y1) / 2, label, fontsize=8, color=color, va="center", ha="left",
                style="italic")


def pipeline(path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 7.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")
    W, H, X = 16.4, 13.0, [1.5, 21.6, 41.7, 61.8, 81.9]
    ya, yb, yc = 46.5, 24.5, 3.0
    ax.text(0.5, 64.5, "(a) LaBraM pre-training (Jiang et al. 2024): reused, not retrained", fontsize=11,
            fontweight="bold", color=INK, va="top")
    a = [("unlabelled EEG", "2,535 h from 16 sources\n(BCI, emotion, TUH subsets,\nSiena, own recordings)", "data"),
         ("1 s patches", "every channel cut into\n200-sample patches,\nplus channel and time\nembeddings", "labram"),
         ("neural tokenizer", "VQ codebook of 8,192 codes,\nlearned by reconstructing\neach patch's Fourier\namplitude and phase", "labram"),
         ("masked EEG modelling", "50 % of the patches masked;\nthe encoder predicts their\ncodes (as BEiT v2)", "labram"),
         ("pre-trained encoder $f_\\theta$", "12 transformer blocks,\nwidth 200, 10 heads,\n5.8 M parameters", "labram")]
    for x, (t, b, k) in zip(X, a):
        box(ax, x, ya, W, H, t, b, k)
    for x0, x1 in zip(X[:-1], X[1:]):
        arrow(ax, x0 + W + 0.4, ya + H / 2, x1 - 0.4, ya + H / 2)

    ax.text(0.5, 42.0, "(b) This work: recording-level brain age (Sections 4.1–4.5)", fontsize=11,
            fontweight="bold", color=INK, va="top")
    b = [("routine EEG recording", "Temple + Harvard: 103,117\nrecordings of 71,656\npatients (train / val / test\nsplit by patient)", "data"),
         ("one preprocessing", "0.1–75 Hz band-pass,\nline removal at 50 and\n60 Hz, 200 Hz, µV;\nsame for every test cohort", "ours"),
         ("10 s windows", "training: a bag of K = 8\nrandom windows of one\nrecording; test: all\nscored windows", "ours"),
         ("common average ref.", "19 channels of the 10–20\nsystem; missing ones left\nout of the token sequence\n(LaBraM++ input)", "ours"),
         ("fine-tuned encoder $f_\\theta$", "one embedding per window\n$h_i \\in \\mathbb{R}^{200}$ and its\nown prediction $p_i = g(h_i)$", "labram")]
    for x, (t, b_, k) in zip(X, b):
        box(ax, x, yb, W, H, t, b_, k)
    for x0, x1 in zip(X[:-1], X[1:]):
        arrow(ax, x0 + W + 0.4, yb + H / 2, x1 - 0.4, yb + H / 2)
    arrow(ax, X[4] + W / 2, ya - 0.4, X[4] + W / 2, yb + H + 0.4, label="initial weights", color="#2a78d6")

    c = [("training", "17 epochs, one run: window\nphase → attention only →\njoint; Huber loss on the\nz-scored age, Eq. (6)", "note"),
         ("age and confidence", "ensemble age $\\hat y$, gap\n$\\hat y - y$; 20 % least\nconfident set aside by\n$u_r$, Eq. (9)", "ours"),
         ("5-fold ensemble", "mean of 5 fold models,\nEq. (7); their disagreement\n$d_r$ and the window spread\n$s_r$ measure confidence", "ours"),
         ("fold model $k$", "one model per fold of the\npatient-grouped 5-fold CV,\nbest epoch on its own\nvalidation fold", "ours"),
         ("attention pooling", "$\\alpha_i = \\mathrm{softmax}(w^\\top \\tanh(V h_i))$\n$\\hat y_k = g(\\sum_i \\alpha_i h_i)$, Eq. (5)\none prediction per\nrecording", "ours")]
    for x, (t, b_, k) in zip(X, c):
        box(ax, x, yc, W, H, t, b_, k)
    arrow(ax, X[4] + W / 2, yb - 0.4, X[4] + W / 2, yc + H + 0.4)
    for x0, x1 in zip(X[4:1:-1], X[3:0:-1]):
        arrow(ax, x0 - 0.4, yc + H / 2, x1 + W + 0.4, yc + H / 2)
    # legend
    from matplotlib.patches import Patch
    handles = [Patch(fc=STYLE[k][0], ec=STYLE[k][1], label=lab) for k, lab in
               (("data", "data"), ("labram", "LaBraM (architecture / weights)"), ("ours", "this work"),
                ("note", "training schedule"))]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.005), ncol=4, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(path)


def attention_example(path, wdir, cohort, stem, trim, model_glob="P4_L1_f*_test.parquet"):
    import matplotlib.pyplot as plt
    from labram.data.tuh_metadata import TUH_CHANNELS
    from labram.eval.age_plots import assert_max_columns, panel_labels
    files = sorted(glob.glob(os.path.join(wdir, model_glob)))
    d = pd.concat([pd.read_parquet(f).assign(fold=k) for k, f in enumerate(files)])
    d = d[d.recording == stem].copy()
    if d.empty:
        raise SystemExit(f"{stem} not in {wdir}")
    d["a"] = d.alpha / d.groupby("fold").alpha.transform("sum")
    pooled = (d.a * d.window_pred).groupby(d.fold).sum().mean()             # Eq. (7) of the fold models' Eq. (5)
    w = d.groupby("slot").agg(p=("window_pred", "mean"), a=("a", "mean"), minute=("minute", "first")).reset_index()
    n, K = len(w), len(files)
    w["wrel"] = w.a * n                                                      # Eq. (15): 1 = uniform weight
    w["dev"] = (w.p - pooled).abs()                                          # Eq. (16)
    age = float(d.age.iloc[0])
    plain = w.p.mean()
    meta = {r["stem"]: r for r in json.load(open(os.path.join(cohort, "manifest.json")))["recordings"]}[stem]
    x = np.load(os.path.join(cohort, "recordings", f"{stem}.npy"), mmap_mode="r")
    mask = np.asarray(meta["channel_mask"], bool)
    hi, lo = int(w.loc[w.wrel.idxmax(), "slot"]), int(w.loc[w.wrel.idxmin(), "slot"])

    plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "xtick.labelsize": 12,
                         "ytick.labelsize": 12, "legend.fontsize": 12})
    fig = plt.figure(figsize=(14, 15))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.9, 1.25])
    ax_a = fig.add_subplot(gs[0, :])
    sc = ax_a.scatter(w.minute + trim / 6, w.p, c=w.wrel, s=18 + 40 * w.wrel, cmap="viridis", zorder=3,
                      edgecolor="white", linewidth=0.4)
    ax_a.axhline(age, color=INK, ls=":", lw=1.6, label=f"true age (bin midpoint) {age:.1f}")
    ax_a.axhline(pooled, color="#eb6834", lw=2.2, label=f"attention-pooled prediction {pooled:.1f}")
    ax_a.axhline(plain, color=MUTED, lw=1.6, ls="--", label=f"plain mean of the windows {plain:.1f}")
    ax_a.set_xlabel("time in the recording (min)")
    ax_a.set_ylabel("window prediction $p_i$ (y)")
    ax_a.set_title(f"window predictions of one LEMON recording, P4 ensemble ({n} windows of 10 s)", fontsize=14)
    ax_a.set_ylim(w.p.min() - 6, w.p.max() + 1.5)                         # room for the legend below the points
    ax_a.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, fontsize=12, loc="lower left", ncol=3)
    cb = fig.colorbar(sc, ax=ax_a, pad=0.01)
    cb.set_label("relative attention weight $w_i$")

    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.bar(w.minute + trim / 6, w.wrel, width=1 / 6 * 0.9, color="#2a78d6", zorder=2)
    ax_b.axhline(1.0, color=MUTED, ls="--", lw=1.2, label="uniform weight")
    ax_b.set_xlabel("time in the recording (min)")
    ax_b.set_ylabel("relative attention weight $w_i$")
    ax_b.set_title("attention weight per window", fontsize=14)
    ax_b.legend(frameon=False, fontsize=12)

    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.scatter(w.dev, w.wrel, s=16, color="#1baf7a", zorder=3)
    rho = w[["dev", "wrel"]].corr("spearman").iloc[0, 1]
    ax_c.set_xlabel("deviation from the pooled prediction $|p_i - \\hat y|$ (y)")
    ax_c.set_ylabel("relative attention weight $w_i$")
    ax_c.set_title(f"weight against deviation (Spearman {rho:+.2f})", fontsize=14)

    names = list(TUH_CHANNELS)
    show = [c for c in ("FP1", "F3", "C3", "P3", "O1", "FP2", "F4", "C4", "P4", "O2", "T3", "T4")
            if c in names and mask[names.index(c)]]
    axes_eeg = []
    for j, (slot, label) in enumerate(((hi, "highest"), (lo, "lowest"))):
        ax = fig.add_subplot(gs[2, j])
        k = slot + trim                                                      # scored slot -> window of the file
        seg = np.asarray(x[k * 2000:(k + 1) * 2000], dtype=np.float32)[:, mask]
        seg = seg - seg.mean(axis=1, keepdims=True)                          # common average reference, as the input
        present = [c for c, m in zip(names, mask) if m]
        t = np.arange(len(seg)) / 200.0
        peak = float(np.abs(seg).max())
        for i, ch in enumerate(show):                                        # display clipped at +-CLIP uV
            ax.plot(t, np.clip(seg[:, present.index(ch)], -CLIP, CLIP) - i * SPACING, color=INK, lw=0.6)
        ax.set_yticks([-i * SPACING for i in range(len(show))], [c.capitalize() for c in show], fontsize=11)
        ax.set_xlabel("time in the window (s)")
        r = w[w.slot == slot].iloc[0]
        note = f", peak {peak:,.0f} µV" if peak > 4 * CLIP else ""
        ax.set_title(f"{label} attention weight, minute {r.minute + trim / 6:.1f}\n$w_i$ = {r.wrel:.2f}, $p_i$ = {r.p:.1f} y{note}",
                     fontsize=13)
        ax.plot([9.6, 9.6], [-len(show) * SPACING + 15, -len(show) * SPACING + 65], color="#eb6834", lw=2)
        ax.text(9.5, -len(show) * SPACING + 40, "50 µV", ha="right", va="center", fontsize=11, color="#eb6834")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        axes_eeg.append(ax)
    for ax in (ax_a, ax_b, ax_c):
        ax.grid(color=GRID, lw=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    panel_labels([ax_a, ax_b, ax_c, *axes_eeg])
    assert_max_columns(fig)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path, f"n={n} pooled={pooled:.2f} plain={plain:.2f} age={age} rho={rho:.2f} hi={hi} lo={lo}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", default="docs/tech_report")
    ap.add_argument("--windows", required=True, help="saved windows of the cohort (eval_age_by_cohort --save-windows)")
    ap.add_argument("--cohort", required=True, help="cohort folder with manifest.json and recordings/*.npy")
    ap.add_argument("--recording", required=True)
    ap.add_argument("--trim-windows", type=int, default=0, help="windows trimmed at the start when the cohort was scored")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    os.makedirs(args.out_dir, exist_ok=True)
    pipeline(os.path.join(args.out_dir, "fig_pipeline.png"))
    attention_example(os.path.join(args.out_dir, "fig_attention_example.png"), args.windows, args.cohort,
                      args.recording, args.trim_windows)


if __name__ == "__main__":
    main()
