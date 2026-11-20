#!/usr/bin/env python
"""Explanatory figures of the technical report (``docs/tech_report/``).

* ``fig_pipeline.png``: (a) the LaBraM pre-training that is reused unchanged
  (neural tokenizer, masked EEG modelling) and (b) the recording-level
  brain-age pipeline of this work built on its encoder (Sections 4.1-4.5).
* ``fig_attention_example.png``: attention pooling over the windows of one
  recording of an open cohort (LEMON, CC BY), P4 ensemble: window predictions
  and relative attention weights over time, weight against the deviation from
  the pooled prediction, and the EEG of the highest- and the lowest-weighted
  window. Ensemble window quantities as Eq. (20) of the report: ``p_i`` the
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


def pipeline(path, labram_png="labram.png"):
    """Figure 5 of the report: this work's fine-tuning and inference path, drawn in the visual
    language of the original LaBraM figure (temporal encoder, temporal & spatial embedding,
    transformer blocks), with the tensor shape after every step for one recording. Figure 4 is the
    original figure itself (Jiang et al. 2024; labram.png of the authors' repository, MIT licence),
    copied next to it as fig_labram_original.png."""
    import shutil
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle
    rng = np.random.default_rng(3)
    TRACE = "#3d7ab8"
    LB = {"temporal": ("#fde9dc", "#f0a582"), "embed": ("#f2f2f2", "#8c8c8c"), "block": ("#e3f1dc", "#93c47d"),
          "head": ("#dfe7f6", "#8ea9db")}
    OURS = ("#fce4d6", "#eb6834")

    def trace(n, smooth=6):
        v = np.convolve(rng.standard_normal(n + 4 * smooth), np.ones(smooth) / smooth, mode="same")[2 * smooth:-2 * smooth]
        v += 0.6 * np.sin(np.linspace(0, n / 9, n) + rng.uniform(0, 6))
        return v / (np.abs(v).max() + 1e-9)

    def box(ax, x, y, w, h, fc, ec, ls="-", lw=1.6, r=0.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, ls=ls, lw=lw))

    def mini(ax, x, y, w, h, n=60, ec="#9a9a9a", fc="white", rows=1):
        ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=0.8))
        t = np.linspace(x + 0.15, x + w - 0.15, n)
        for q in range(rows):
            yc = y + h * (q + 0.5) / rows
            ax.plot(t, yc + 0.36 * h / rows * trace(n, 3), color=TRACE, lw=0.6)

    def arrow(ax, x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5, mutation_scale=14))

    def shape(ax, x, y, txt, **kw):
        ax.text(x, y, txt, ha="center", va="top", fontsize=10.5, color="#a33a0f", **kw)

    shutil.copyfile(labram_png, os.path.join(os.path.dirname(path), "fig_labram_original.png"))
    fig, ax = plt.subplots(figsize=(14, 9.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")
    ax.text(0.5, 60.5, "Fine-tuning", fontsize=13, fontweight="bold", color=INK, va="top", family="serif")
    # 1. preprocessed recording
    chans = ["Fp1", "C3", "O1", "Fp2", "C4", "O2"]
    x0, x1, ytop = 3.6, 17.5, 54.0
    ax.add_patch(Rectangle((x0, ytop - 15.5), x1 - x0, 16.5, fc="white", ec="#9a9a9a", lw=0.9))
    for k, c in enumerate(chans):
        yy = ytop - 1.4 - k * 2.6
        ax.plot(np.linspace(x0 + 0.2, x1 - 0.2, 400), yy + 0.9 * trace(400, 4), color=TRACE, lw=0.6)
        ax.text(x0 - 0.3, yy, c, ha="right", va="center", fontsize=9, family="serif")
    for xx in np.linspace(x0, x1, 6)[1:-1]:
        ax.plot([xx, xx], [ytop - 15.5, ytop + 1.0], color=INK, lw=0.7, ls=":")
    ax.text((x0 + x1) / 2, ytop + 1.6, "preprocessed recording", ha="center", fontsize=11)
    shape(ax, (x0 + x1) / 2, ytop - 16.0, "19 × (200 Hz × duration)\nEqs. (1)–(4)")
    arrow(ax, x1 + 0.6, 46.5, x1 + 3.4, 46.5)
    # 2. bag of K windows
    for k in range(3):
        mini(ax, 22.0 + 1.0 * k, 41.0 + 1.6 * (2 - k), 7.4, 7.0, n=70, rows=4)
    ax.text(26.6, 53.0, "bag of K = 8 windows", ha="center", fontsize=11)
    shape(ax, 26.6, 40.4, "K × 19 × 2,000\n(test: all scored windows)")
    arrow(ax, 32.2, 46.5, 35.0, 46.5)
    # 3. CAR + patches
    for r in range(4):
        for c in range(4):
            if r == 2 and c == 2:
                continue
            mini(ax, 35.8 + c * 3.2, 51.0 - r * 2.9, 2.8, 2.4, n=18)
    ax.text(43.6, 46.4, "⋱", fontsize=14, ha="center", va="center")
    ax.text(42.0, 54.4, "CAR, ÷ 100, 1 s patches", ha="center", fontsize=11)
    shape(ax, 42.0, 41.3, "19 × 10 × 200\nper window, Eq. (5)")
    arrow(ax, 48.8, 46.5, 51.6, 46.5)
    # 4. encoder f_theta (LaBraM architecture, pre-trained weights, fine-tuned)
    box(ax, 52.2, 32.6, 46.6, 25.0, "none", "#2f5597", ls=(0, (5, 3)), lw=1.6, r=2.5)
    ax.text(75.5, 58.6, "encoder $f_\\theta$: LaBraM base, initialised from pre-training (Figure 4), fine-tuned\n12 blocks · width 200 · 10 heads · 5.8 M parameters", ha="center", fontsize=10.5, color="#2f5597", linespacing=1.3)
    cols = [("Temporal Encoder", "temporal", "-", "190×200"), ("Temporal & Spatial\nEmbedding", "embed", (0, (3, 2)), "191×200\n(+[CLS])"),
            ("Transformer Block 1", "block", "-", ""), ("Transformer Block 2", "block", "-", ""),
            (None, None, None, None), ("Transformer Block 12", "block", "-", "191×200"), ("[CLS] token", "head", "-", "$h_i \\in \\mathbb{R}^{200}$")]
    xs = [53.4, 60.0, 67.0, 72.6, 78.2, 81.2, 89.0]
    for (lbl, kind, ls, shp), xx in zip(cols, xs):
        if lbl is None:
            ax.text(xx + 0.4, 46.3, "……", fontsize=14, ha="center", va="center")
            continue
        w = 4.6 if kind != "head" else 4.2
        fc, ec = LB[kind]
        box(ax, xx, 37.6, w, 17.8, fc, ec, ls=ls, r=0.9)
        ax.text(xx + w / 2, 46.5, lbl, rotation=-90, ha="center", va="center", fontsize=11, family="serif", linespacing=1.0)
        if shp:
            ax.text(xx + w / 2, 37.1, shp, ha="center", va="top", fontsize=10, color="#a33a0f")
    # row 2: window embeddings -> attention pooling -> head -> folds -> ensemble
    arrow(ax, 91.1, 32.2, 91.1, 28.9)
    ax.text(0.5, 30.5, "Inference & pooling", fontsize=13, fontweight="bold", color=INK, va="top", family="serif")
    # embeddings matrix
    for k in range(8):
        for c in range(10):
            ax.add_patch(Rectangle((85.6 + c * 1.15, 25.6 - k * 1.35), 1.05, 1.15, fc=plt.cm.Blues(0.25 + 0.6 * rng.random()), ec="white", lw=0.3))
    ax.text(91.4, 27.8, "window embeddings", ha="center", fontsize=11)
    shape(ax, 91.4, 14.3, "K × 200")
    arrow(ax, 85.0, 21.0, 82.4, 21.0)
    # attention pooling (this work)
    box(ax, 59.2, 11.4, 22.6, 16.0, *OURS, lw=2.0, r=1.2)
    ax.text(70.5, 26.0, "attention pooling", ha="center", va="top", fontsize=11.5, fontweight="bold")
    al = np.array([0.9, 1.6, 0.4, 1.2, 2.1, 0.3, 1.0, 1.5])
    for k, v in enumerate(al):
        ax.add_patch(Rectangle((63.2 + k * 1.9, 19.4), 1.4, v * 1.75, fc="#eb6834" if v >= 1 else "#f4b183", ec="none"))
    ax.text(70.5, 19.0, "$\\alpha_i = \\mathrm{softmax}_i(\\mathbf{w}^\\top \\tanh(\\mathbf{V} h_i))$\n"
            "$\\mathbf{V} \\in \\mathbb{R}^{128 \\times 200}$;  $\\sum_i \\alpha_i h_i \\in \\mathbb{R}^{200}$\nEq. (10)",
            ha="center", va="top", fontsize=10.2, linespacing=1.5)
    arrow(ax, 58.6, 21.0, 56.0, 21.0)
    # head g
    fc, ec = LB["head"]
    box(ax, 50.0, 13.0, 5.4, 14.0, fc, ec, r=0.9)
    ax.text(52.7, 20.0, "Age Head $g$", rotation=-90, ha="center", va="center", fontsize=11, family="serif")
    shape(ax, 52.7, 12.5, "200 → 1\n$\\hat y_k = \\mu + \\sigma \\hat z_k$")
    arrow(ax, 49.4, 21.0, 46.6, 21.0)
    # five fold models
    for k in range(5):
        box(ax, 34.6 + 0.7 * k, 13.8 + 1.9 * (4 - k), 9.0, 4.6, "#ffffff", "#eb6834", lw=1.4, r=0.5)
    ax.text(41.9, 16.1, "fold model 5", ha="center", va="center", fontsize=10)
    ax.text(39.8, 28.0, "5 fold models (patient-grouped CV)", ha="center", fontsize=11)
    shape(ax, 39.8, 13.3, "5 predictions $\\hat y_{r,k}$\nper recording")
    arrow(ax, 34.0, 21.0, 32.8, 21.0)
    # ensemble and confidence
    box(ax, 2.0, 11.4, 30.2, 16.0, *OURS, lw=2.0, r=1.2)
    ax.text(17.1, 26.0, "ensemble age and confidence", ha="center", va="top", fontsize=11.5, fontweight="bold")
    ax.text(17.1, 22.6, "$\\hat y_r = \\frac{1}{5}\\sum_k \\hat y_{r,k}$, Eq. (12)\n"
            "fold disagreement $d_r$, window spread $s_{r,k}$\n"
            "$\\rightarrow$ uncertainty $u_r$, Eq. (14)\n"
            "brain-age gap $\\hat y_r - y_r$; 20 % least\nconfident recordings set aside",
            ha="center", va="top", fontsize=10.5, linespacing=1.45)
    shape(ax, 17.1, 10.9, "one age and one $u_r$ per recording")
    # training note and legend
    ax.text(50.0, 5.6, "Training (17 epochs, one run): window phase (E2L-like, window mean) → attention only → joint;"
            " Huber loss on the z-scored age, Eq. (11).", ha="center", fontsize=10.5, color=MUTED, style="italic")
    hand = [(LB["block"], "-", "LaBraM components (as in Figure 4)"), (OURS, "-", "this work"), (("#ffffff", "#2f5597"), (0, (5, 3)), "pre-trained weights, fine-tuned")]
    for k, ((fc, ec), ls, lbl) in enumerate(hand):
        xx = 10 + k * 31
        ax.add_patch(Rectangle((xx, 0.6), 2.6, 1.8, fc=fc, ec=ec, ls=ls, lw=1.6))
        ax.text(xx + 3.4, 1.5, lbl, va="center", fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
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
    pooled = (d.a * d.window_pred).groupby(d.fold).sum().mean()             # Eq. (12) of the fold models' Eq. (10)
    w = d.groupby("slot").agg(p=("window_pred", "mean"), a=("a", "mean"), minute=("minute", "first")).reset_index()
    n, K = len(w), len(files)
    w["wrel"] = w.a * n                                                      # Eq. (20): 1 = uniform weight
    w["dev"] = (w.p - pooled).abs()                                          # Eq. (21)
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


def attention_heatmaps(path, wdir, cohort, cases, trim, model_glob="P4_L1_f*_test.parquet"):
    """Attention over whole recordings: per case, (top) relative attention weight of every
    window for each fold model and their mean, (middle) the window predictions with the true
    age and the pooled prediction, (bottom) the window's median channel SD after CAR."""
    import matplotlib.pyplot as plt
    from labram.eval.age_plots import assert_max_columns, panel_labels
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12})
    files = sorted(glob.glob(os.path.join(wdir, model_glob)))
    d = pd.concat([pd.read_parquet(f).assign(fold=k) for k, f in enumerate(files)])
    meta = {r["stem"]: r for r in json.load(open(os.path.join(cohort, "manifest.json")))["recordings"]}
    eeg19 = [i for i in range(23) if i not in (16, 17, 21, 22)]
    fig = plt.figure(figsize=(10.5, 4.0 * len(cases)))
    outer = fig.add_gridspec(len(cases), 1, hspace=0.42)
    tops, im = [], None
    for c, (stem, label) in enumerate(cases):
        r = d[d.recording == stem].copy()
        r["a"] = r.alpha / r.groupby("fold").alpha.transform("sum")
        n = r[r.fold == 0].shape[0]
        r["w"] = r.a * n                                                     # Eq. (20): 1 = uniform weight
        pooled = (r.a * r.window_pred).groupby(r.fold).sum().mean()          # Eq. (12)
        W = r.pivot(index="fold", columns="slot", values="w")
        P = r.pivot(index="fold", columns="slot", values="window_pred").mean(axis=0)
        mat = np.vstack([W.to_numpy(), W.mean(axis=0).to_numpy()])
        minutes = (W.columns.to_numpy() + trim) / 6.0
        age = float(r.age.iloc[0])
        x = np.load(os.path.join(cohort, "recordings", f"{stem}.npy"), mmap_mode="r")
        mask = np.asarray(meta[stem]["channel_mask"], bool)
        keep = [i for i in eeg19 if mask[i]]
        sd = []
        for slot in W.columns:
            k = int(slot) + trim
            seg = np.asarray(x[k * 2000:(k + 1) * 2000], dtype=np.float32)[:, keep]
            seg = seg - seg.mean(axis=1, keepdims=True)
            sd.append(float(np.median(seg.std(axis=0))))
        inner = outer[c].subgridspec(2, 1, height_ratios=[1.2, 1.4], hspace=0.08)
        ax0 = fig.add_subplot(inner[0])
        ax1 = fig.add_subplot(inner[1], sharex=ax0)
        step = 1 / 6.0
        im = ax0.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=2.5, interpolation="nearest",
                        extent=(minutes[0] - step / 2, minutes[-1] + step / 2, len(mat) - 0.5, -0.5))
        ax0.set_yticks(range(len(mat)), [f"fold {k}" for k in range(len(W))] + ["mean"], fontsize=8.5)
        ax0.axhline(len(W) - 0.5, color="white", lw=1.5)
        ax0.set_title(f"{label}: true age {age:.1f}, predicted {pooled:.1f}, error {pooled - age:+.1f} y")
        ax1.scatter(minutes, P, c=mat[-1], cmap="viridis", vmin=0, vmax=2.5, s=26, edgecolor="white", lw=0.3, zorder=3)
        ax1.axhline(age, color=INK, ls=":", lw=1.6, label="true age")
        ax1.axhline(pooled, color="#eb6834", lw=2, label="attention-pooled prediction")
        ax1.axhline(P.mean(), color=MUTED, ls="--", lw=1.4, label="plain mean of the windows")
        ax1.set_ylabel("window prediction (y)")
        ax1.legend(frameon=False, fontsize=9, ncol=3, loc="upper left")
        art = np.asarray(sd) > 100                                           # artifact windows: channel SD > 100 uV
        if art.any():
            ax0.scatter(minutes[art], np.full(art.sum(), 0.93), marker="v", s=40, color="#e3242b", zorder=5,
                        edgecolor="white", linewidth=0.6, transform=ax0.get_xaxis_transform())
        ax1.set_xlabel("time in the recording (min)" if c == len(cases) - 1 else "")
        plt.setp(ax0.get_xticklabels(), visible=False)
        for ax in (ax1,):
            ax.grid(color=GRID, lw=0.8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        tops.append(ax0)
    cb = fig.colorbar(im, ax=fig.axes, fraction=0.02, pad=0.01)
    cb.set_label("relative attention weight $w_i$ (1 = uniform)")
    panel_labels(tops)
    assert_max_columns(fig)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(path)


def attention_on_eeg(path, wdir, cohort, cases, trim, model_glob="P4_L1_f*_test.parquet",
                     full_channels=("FP1", "C3", "O1", "FP2", "C4", "O2"),
                     zoom_channels=("FP1", "F3", "C3", "O1", "FP2", "F4", "C4", "O2"), zoom_windows=9):
    """The EEG itself with every 10 s window's background coloured by its relative attention
    weight (mean over the fold models, 1 = uniform): per case, the whole scored recording and a
    zoom of ``zoom_windows`` windows centred on the lowest-weighted one, with each window's
    predicted age and weight written above it. Display only: common average reference over the
    present 10-20 channels and a 0.5 Hz zero-phase high-pass so slow drifts do not hide the traces."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from scipy.signal import butter, sosfiltfilt
    from labram.data.tuh_metadata import TUH_CHANNELS
    from labram.eval.age_plots import assert_max_columns, panel_labels
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11.5})
    files = sorted(glob.glob(os.path.join(wdir, model_glob)))
    d = pd.concat([pd.read_parquet(f).assign(fold=k) for k, f in enumerate(files)])
    meta = {r["stem"]: r for r in json.load(open(os.path.join(cohort, "manifest.json")))["recordings"]}
    names = list(TUH_CHANNELS)
    cmap, norm = plt.get_cmap("RdYlGn"), TwoSlopeNorm(vmin=0.3, vcenter=1.0, vmax=2.0)
    sos = butter(2, 0.5, btype="highpass", fs=200, output="sos")
    fig = plt.figure(figsize=(11, 5.4 * len(cases)))
    outer = fig.add_gridspec(len(cases), 1, hspace=0.3)
    axes = []
    for c, (stem, label) in enumerate(cases):
        r = d[d.recording == stem].copy()
        r["a"] = r.alpha / r.groupby("fold").alpha.transform("sum")
        n = r[r.fold == 0].shape[0]
        w = (r.a * n).groupby(r.slot).mean()                                  # Eq. (20)
        p = r.groupby("slot").window_pred.mean()
        pooled = (r.a * r.window_pred).groupby(r.fold).sum().mean()          # Eq. (12)
        age = float(r.age.iloc[0])
        mask = np.asarray(meta[stem]["channel_mask"], bool)
        x = np.load(os.path.join(cohort, "recordings", f"{stem}.npy"), mmap_mode="r")
        present = [i for i in range(23) if mask[i] and i not in (16, 17, 21, 22)]
        slots = w.index.to_numpy()
        seg = np.asarray(x[(slots[0] + trim) * 2000:(slots[-1] + trim + 1) * 2000], dtype=np.float64)[:, present]
        seg = seg - seg.mean(axis=1, keepdims=True)                          # common average reference
        seg = sosfiltfilt(sos, seg, axis=0)                                  # display-only high-pass
        t0 = (slots[0] + trim) * 10.0                                         # seconds from the file start
        lo = int(w.idxmin())
        zs = max(slots[0], min(lo - zoom_windows // 2, slots[-1] - zoom_windows + 1))
        inner = outer[c].subgridspec(2, 1, height_ratios=[0.8, 1.2], hspace=0.55)
        for row, (s0, s1, chans, spacing, step) in enumerate(
                ((slots[0], slots[-1] + 1, full_channels, 90.0, 4), (zs, zs + zoom_windows, zoom_channels, 70.0, 1))):
            ax = fig.add_subplot(inner[row])
            scale = (lambda v: v / 60) if row == 0 else (lambda v: v)
            for sl in range(s0, s1):
                ax.axvspan(scale(t0 + (sl - slots[0]) * 10), scale(t0 + (sl - slots[0] + 1) * 10),
                           color=cmap(norm(w[sl])), alpha=0.6, lw=0, zorder=0)
            show = [ch for ch in chans if ch in names and names.index(ch) in present]
            a, b = (s0 - slots[0]) * 2000, (s1 - slots[0]) * 2000
            tt = scale(t0 + np.arange(a, b) / 200.0)
            clip = 0.55 * spacing
            for k, ch in enumerate(show):
                v = seg[a:b:step, present.index(names.index(ch))]
                ax.plot(tt[::step], np.clip(v, -clip, clip) - k * spacing, color="#1f1f1e",
                        lw=0.3 if row == 0 else 0.55, rasterized=True, zorder=2)
            ax.set_yticks([-k * spacing for k in range(len(show))], [ch.capitalize() for ch in show], fontsize=8.5)
            top = 0.7 * spacing if row == 0 else 1.5 * spacing
            ax.set_ylim(-(len(show) - 0.35) * spacing, top)
            ax.set_xlim(tt[0], tt[-1])
            if row == 0:
                ax.set_xlabel("time in the recording (min)")
                ax.axvspan(scale(t0 + (zs - slots[0]) * 10), scale(t0 + (zs - slots[0] + zoom_windows) * 10),
                           fill=False, ec="#1f1f1e", lw=1.6, zorder=3)
                ax.set_title(f"{label}: true age {age:.1f}, predicted {pooled:.1f} (error {pooled - age:+.1f} y)")
            else:
                ax.set_xlabel("time in the recording (s)")
                for sl in range(s0, s1):
                    ax.text(t0 + (sl - slots[0]) * 10 + 5, 0.95 * spacing, f"{p[sl]:.0f} y · w {w[sl]:.2f}",
                            ha="center", va="center", fontsize=8, color="#1f1f1e", zorder=4)
                ax.set_title("zoom on the box: window prediction and attention weight above each 10 s window")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            axes.append(ax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label("relative attention weight $w_i$ (1 = uniform)")
    panel_labels(axes)
    assert_max_columns(fig)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(path)

PHASES = {  # recipe -> [(first epoch, last epoch, phase)], from the runs' mil.window_epochs and schedules
    "P1": [(0, 9, "window"), (10, 11, "attention only"), (12, 15, "joint")],
    "P3": [(0, 9, "window"), (10, 10, "attention only"), (11, 16, "joint")],
    "P4": [(0, 7, "window"), (8, 8, "attention only"), (9, 16, "joint")],
}
PHASE_COLORS = {"window": "#e3edf9", "attention only": "#fbecc8", "joint": "#e9def3"}


RECIPE_COLORS = {"P1": "#6250d6", "P3": "#008080", "P4": "#d9480f"}   # distinct from the split colours
RECIPE_PHASE_STARTS = {r: [e0 for e0, _, ph in v if ph != "window"] for r, v in PHASES.items()}


def welch_p(m1, s1, n1, m2, s2, n2):
    """Two-sided Welch t-test from summary statistics (mean, sample SD, n)."""
    from scipy import stats
    v1, v2 = s1 ** 2 / n1, s2 ** 2 / n2
    t = (m1 - m2) / np.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    return 2 * stats.t.sf(abs(t), df)


def recipe_panel(ax, d, split, group, title, sig_pairs=(("P4", "P3"), ("P4", "P1"))):
    """The three recipes on one axis (fold mean +- SD), attention-phase starts as dotted lines
    in the recipe colour, and per-epoch Welch tests of each pair marked under the curves."""
    for recipe, color in RECIPE_COLORS.items():
        g = d[(d.recipe == recipe) & (d.split == split) & (d.group == group)].sort_values("epoch")
        ax.errorbar(g.epoch, g["mean"], yerr=g["sd"], color=color, marker="o", ms=3.8, lw=1.6, capsize=2.5,
                    label=recipe, zorder=3)
        for e in RECIPE_PHASE_STARTS[recipe][:1]:
            ax.axvline(e - 0.5, color=color, ls=":", lw=1.3, zorder=1)
    shown = d[(d.split == split) & (d.group == group)]
    lo, hi = (shown["mean"] - shown["sd"]).min(), (shown["mean"] + shown["sd"]).max()
    ax.set_ylim(lo - 0.25 * (hi - lo), hi + 0.04 * (hi - lo))       # a band below the data for the test marks
    for k, (a, b) in enumerate(sig_pairs):
        ga = d[(d.recipe == a) & (d.split == split) & (d.group == group)].set_index("epoch")
        gb = d[(d.recipe == b) & (d.split == split) & (d.group == group)].set_index("epoch")
        ep = ga.index.intersection(gb.index)
        p = np.array([welch_p(ga.loc[e, "mean"], ga.loc[e, "sd"], ga.loc[e, "n"],
                              gb.loc[e, "mean"], gb.loc[e, "sd"], gb.loc[e, "n"]) for e in ep])
        sig = ep[p < 0.05]
        y = 0.035 + 0.055 * k
        ax.scatter(sig, np.full(len(sig), y), transform=ax.get_xaxis_transform(), marker="*", s=40,
                   color=RECIPE_COLORS[b], edgecolor=RECIPE_COLORS[a], linewidth=0.8, zorder=4)
        ax.text(1.0, y, f"{a} vs {b}", transform=ax.transAxes, fontsize=8, va="center", ha="left", color=INK)
    ax.set_title(title)
    ax.set_xlabel("epoch")


def cv_curves(path, csv):
    """Cross-validation curves per epoch of the one-run recipes, one panel per recipe:
    recording MAE of the train-eval sample, each fold model's own validation fold and the
    held-out test split, mean over the five fold models with the sample SD as error bar;
    background = training phase. The legend takes the fourth grid cell."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MultipleLocator
    from labram.eval.age_plots import assert_max_columns, panel_labels
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
    d = pd.read_csv(csv)
    lines = [("train_eval", "train-eval (500 training recordings per dataset)", "#1baf7a", "s"),
             ("val", "validation (own fold)", "#2a78d6", "o"), ("test", "test (held out)", "#eb6834", "^")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    a = list(axes.flat)
    da = d[d.group == "all"]
    for ax, recipe in zip(a[:3], ("P1", "P3", "P4")):
        for e0, e1, ph in PHASES[recipe]:
            ax.axvspan(e0 - 0.5, e1 + 0.5, color=PHASE_COLORS[ph], zorder=0, lw=0)
        for split, label, color, mk in lines:
            g = da[(da.recipe == recipe) & (da.split == split)].sort_values("epoch")
            ax.errorbar(g.epoch, g["mean"], yerr=g["sd"], color=color, marker=mk, ms=4.5, lw=1.6, capsize=3,
                        label=label, zorder=3)
        last = da[da.recipe == recipe].epoch.max()
        gv = da[(da.recipe == recipe) & (da.epoch == last)].set_index("split")["mean"]
        ax.set_title(f"{recipe}: final val {gv['val']:.2f}, train-eval {gv['train_eval']:.2f} "
                     f"(gap {gv['val'] - gv['train_eval']:.2f})")
        ax.set_xlabel("epoch")
        ax.set_ylabel("recording MAE (years)")
        ax.set_ylim(3, 11)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.grid(color=GRID, lw=0.8, zorder=1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    a[3].axis("off")
    handles = [plt.Line2D([], [], color=c, marker=m, lw=1.6, label=lbl) for _, lbl, c, m in lines]
    handles += [Patch(facecolor=PHASE_COLORS[p], label=f"{p} phase") for p in PHASE_COLORS]
    a[3].legend(handles=handles, loc="center", frameon=False, fontsize=11)
    fig.tight_layout()
    panel_labels(a[:3])
    assert_max_columns(fig)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path)


def cv_compare(path, csv):
    """The three one-run recipes compared over epochs on validation, test and LEMON (fold mean
    +- SD), first attention-only epoch dotted, per-epoch Welch tests marked; legend in the
    fourth grid cell."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    from labram.eval.age_plots import assert_max_columns, panel_labels
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
    d = pd.read_csv(csv)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    a = list(axes.flat)
    recipe_panel(a[0], d, "val", "all", "validation MAE (own fold)")
    recipe_panel(a[1], d, "test", "all", "test MAE (held out)")
    recipe_panel(a[2], d, "test", "lemon", "LEMON MAE (zero-shot)")
    for ax in a[:3]:
        ax.set_ylabel("recording MAE (years)")
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.grid(color=GRID, lw=0.8, zorder=1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    a[3].axis("off")
    handles = [plt.Line2D([], [], color=c, marker="o", lw=1.6, label=r) for r, c in RECIPE_COLORS.items()]
    handles += [plt.Line2D([], [], color=RECIPE_COLORS["P3"], ls=":", lw=1.4, label="first attention-only epoch, P1 and P3 (10)"),
                plt.Line2D([], [], color=RECIPE_COLORS["P4"], ls=":", lw=1.4, label="first attention-only epoch, P4 (8)")]
    handles += [plt.Line2D([], [], color=INK, marker="*", lw=0, ms=9, label="Welch p < 0.05 at that epoch")]
    a[3].legend(handles=handles, loc="center", frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.97, 1))
    panel_labels(a[:3])
    assert_max_columns(fig)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path)

def cv_cohorts(path, csv):
    """Held-out test MAE per data cohort over epochs, the three one-run recipes on each axis
    (fold mean +- SD), with per-epoch Welch tests marked."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    from labram.eval.age_plots import assert_max_columns, panel_labels
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
    d = pd.read_csv(csv)
    panels = [("temple_normal", "Temple, normal report"), ("temple_abnormal", "Temple, abnormal report"),
              ("harvard_normal", "Harvard, normal report"), ("harvard_abnormal", "Harvard, abnormal report"),
              ("harvard_unlabelled", "Harvard, no report"), ("lemon", "LEMON, zero-shot")]
    fig, axes = plt.subplots(3, 2, figsize=(13, 13.5))
    for ax, (group, title) in zip(axes.flat, panels):
        recipe_panel(ax, d, "test", group, title)
        ax.set_ylabel("recording MAE (years)")
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.grid(color=GRID, lw=0.8, zorder=1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=10, loc="upper right")
    fig.tight_layout(rect=(0, 0, 0.97, 1))
    panel_labels(list(axes.flat))
    assert_max_columns(fig)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path)

def preprocessing_figure(path, edf, start_s=140.0, channels=("FP1", "F3", "C3", "P3", "O1", "FP2", "C4", "O2")):
    """The conversion pipeline step by step on one recording of an open cohort (Dortmund,
    CC0): the reader (Dortmund: DC offset removed, FCz reference kept), the 0.1-75 Hz
    band-pass, line removal by spectrum interpolation at 50 and 60 Hz, resampling to
    200 Hz, and the model input (common average reference, x / 100, 1 s patches)."""
    import mne
    import matplotlib.pyplot as plt
    from scipy.signal import welch
    from dataset_maker.make_external_npy import open_dortmund
    from dataset_maker.make_TUH_npy import H_FREQ, L_FREQ, SAMPLE_RATE
    from labram.data.line_removal import spectrum_interpolate
    from labram.data.tuh_metadata import TUH_CHANNELS
    from labram.eval.age_plots import assert_max_columns, panel_labels
    mne.set_log_level("error")
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 13.5})
    raw, present, _ = open_dortmund(edf)
    order = [c for c in TUH_CHANNELS if c in present]
    raw.pick([present[c] for c in order]).load_data()
    fs = raw.info["sfreq"]
    x0 = raw.get_data(units="uV")                                         # as read (stage 0)
    bp = raw.copy().filter(l_freq=L_FREQ, h_freq=H_FREQ)                  # Eq. (1)
    x1 = bp.get_data(units="uV")
    notch = bp.copy()
    for f0 in (50.0, 60.0):
        notch.notch_filter(f0)
    xn = notch.get_data(units="uV")
    li = bp.copy()
    for f0 in (50.0, 60.0):
        li.apply_function(spectrum_interpolate, channel_wise=True, fs=fs, f0=f0)   # Eq. (2)
    x2 = li.get_data(units="uV")
    rs = li.copy().resample(SAMPLE_RATE)
    x3 = rs.get_data(units="uV")                                          # 200 Hz, µV
    show = [c for c in channels if c in order]
    idx = [order.index(c) for c in show]
    occ = order.index("O1")
    spacing = 75.0
    clip = 0.95 * spacing                                                 # display only

    def traces(ax, x, rate, title):
        a, b = int(start_s * rate), int((start_s + 10) * rate)
        t = np.arange(b - a) / rate
        for k, i in enumerate(idx):
            v = x[i, a:b] - x[i, a:b].mean()
            v = np.clip(v, -clip, clip)
            ax.plot(t, v - k * spacing, color=INK, lw=0.5)
        ax.set_yticks([-k * spacing for k in range(len(show))], [c.capitalize() for c in show], fontsize=11)
        ax.set_xlabel("time (s)")
        ax.set_xlim(0, 10)
        ax.set_title(title)

    fig = plt.figure(figsize=(13, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 0.9, 1.05])
    a = fig.add_subplot(gs[0, 0])
    traces(a, x0, fs, f"as read: {fs:.0f} Hz, FCz reference, µV")
    b = fig.add_subplot(gs[0, 1])
    traces(b, x3, SAMPLE_RATE, "0.1–75 Hz, line removal, 200 Hz")
    c = fig.add_subplot(gs[1, 0])
    for x, rate, lbl, col in ((x0, fs, "as read", "#6b6a64"), (x1, fs, "band-pass 0.1–75 Hz, Eq. (1)", "#2a78d6"),
                              (x2, fs, "+ line removal, Eq. (2)", "#eb6834"), (x3, SAMPLE_RATE, "+ resampled to 200 Hz", "#1baf7a")):
        f, pxx = welch(x[occ], fs=rate, nperseg=int(4 * rate))
        keep = f <= min(rate / 2, 150)
        c.semilogy(f[keep], pxx[keep], color=col, lw=1.4 if lbl != "as read" else 1.0, label=lbl)
    c.axvline(75, color=MUTED, ls=":", lw=1)
    c.set_ylim(1e-5, None)
    c.set_xlabel("frequency (Hz)")
    c.set_ylabel("O1 power (µV²/Hz)")
    c.set_title("spectrum after each step")
    c.legend(frameon=False, fontsize=11, loc="upper right")
    d = fig.add_subplot(gs[1, 1])
    for x, lbl, col in ((x1, "band-pass only (line present)", "#2a78d6"), (xn, "notch filter at 50 Hz", "#6250d6"),
                        (x2, "spectrum interpolation, Eq. (2)", "#eb6834")):
        f, pxx = welch(x[occ], fs=fs, nperseg=int(10 * fs))
        keep = (f >= 40) & (f <= 58)
        d.semilogy(f[keep], pxx[keep], color=col, lw=1.4, label=lbl)
    d.axvspan(48.5, 51.5, color="#fbecc8", zorder=0, lw=0)
    d.set_xlabel("frequency (Hz)")
    d.set_ylabel("O1 power (µV²/Hz)")
    d.set_title("around 50 Hz: notch against interpolation")
    d.legend(frameon=False, fontsize=11, loc="lower left")
    e = fig.add_subplot(gs[2, :])
    a0, b0 = int(start_s * SAMPLE_RATE), int((start_s + 10) * SAMPLE_RATE)
    win = x3[:, a0:b0]
    car = win - win.mean(axis=0, keepdims=True)                           # Eq. (5): common average reference
    t = np.arange(b0 - a0) / SAMPLE_RATE
    for sec in range(1, 10):
        e.axvline(sec, color="#2a78d6", lw=0.8, ls="--", zorder=1)
    for k, i in enumerate(idx):
        e.plot(t, np.clip(car[i], -clip, clip) - k * spacing, color=INK, lw=0.6, zorder=2)
    e.set_yticks([-k * spacing for k in range(len(show))], [c.capitalize() for c in show], fontsize=11)
    e.set_xlim(0, 10)
    e.set_xlabel("time in the window (s)")
    e.set_title(f"model input: one 10 s window, common average reference over the {len(order)} channels,\n"
                "ten 1 s patches (dashed) of 200 samples; the model reads values / 100")
    for ax in (a, b, c, d, e):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in (c, d):
        ax.grid(color=GRID, lw=0.8)
    fig.tight_layout()
    panel_labels([a, b, c, d, e])
    assert_max_columns(fig)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", default="docs/tech_report")
    ap.add_argument("--windows", required=True, help="saved windows of the cohort (eval_age_by_cohort --save-windows)")
    ap.add_argument("--cohort", required=True, help="cohort folder with manifest.json and recordings/*.npy")
    ap.add_argument("--recording", required=True)
    ap.add_argument("--trim-windows", type=int, default=0, help="windows trimmed at the start when the cohort was scored")
    ap.add_argument("--preproc-edf", default=None, help="Dortmund EDF for the preprocessing figure (fig_preprocessing.png)")
    ap.add_argument("--cv-curves", default=None, help="docs/data/mil_phased_cv/cv_curves.csv -> fig_cv_curves.png")
    ap.add_argument("--eeg-cases", nargs="*", default=[], metavar="STEM=LABEL",
                    help="recordings drawn as EEG with the attention weight as background (fig_attention_on_eeg.png)")
    ap.add_argument("--heatmap-cases", nargs="*", default=[], metavar="STEM=LABEL",
                    help="recordings for the whole-recording attention heatmaps (fig_attention_heatmaps.png)")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    os.makedirs(args.out_dir, exist_ok=True)
    pipeline(os.path.join(args.out_dir, "fig_pipeline.png"))
    attention_example(os.path.join(args.out_dir, "fig_attention_example.png"), args.windows, args.cohort,
                      args.recording, args.trim_windows)
    if args.preproc_edf:
        preprocessing_figure(os.path.join(args.out_dir, "fig_preprocessing.png"), args.preproc_edf)
    if args.cv_curves:
        cv_curves(os.path.join(args.out_dir, "fig_cv_curves.png"), args.cv_curves)
        cv_compare(os.path.join(args.out_dir, "fig_cv_compare.png"), args.cv_curves)
        cv_cohorts(os.path.join(args.out_dir, "fig_cv_cohorts.png"), args.cv_curves)
    if args.eeg_cases:
        attention_on_eeg(os.path.join(args.out_dir, "fig_attention_on_eeg.png"), args.windows, args.cohort,
                         [tuple(c.split("=", 1)) for c in args.eeg_cases], args.trim_windows)
    if args.heatmap_cases:
        cases = [c.split("=", 1) for c in args.heatmap_cases]
        attention_heatmaps(os.path.join(args.out_dir, "fig_attention_heatmaps.png"), args.windows, args.cohort,
                           [(stem, label) for stem, label in cases], args.trim_windows)


if __name__ == "__main__":
    main()
