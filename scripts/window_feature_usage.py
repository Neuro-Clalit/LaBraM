#!/usr/bin/env python
"""How a model trained with ``mil.window_features=handcrafted`` uses its handcrafted window features.

Figures and tables of ``docs/window_features.md``:

* ``pipeline.png``: where the features enter the model (Figure 1.1);
* ``projection_blocks.png`` + ``projection_blocks.csv``: share of the trained projection's weight
  energy per feature block, and energy per feature relative to its value at initialisation (Figure 8.1);
* ``usage.csv``: on test recordings with all 19 channels, the spread of the class-token and feature
  parts of the window prediction and of the attention pre-activation, and the recording MAE of the
  model with the features set to their training mean (Table 8.2).

    python scripts/window_feature_usage.py --run /data/runs/heedb/mil_features/P4F_L1/cv/fold_0 \\
        --cohort /data/runs/heedb/mil_features/cohort_view_compact --out docs/figures/window_features
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from labram.layers.window_features import BANDS, FILTERBANK, N_CHANNELS  # noqa: E402

CORE_GROUPS = [("absolute band power", 0, 5), ("relative band power", 5, 10), ("power ratios", 10, 12),
               ("edge freq. + entropy", 12, 14), ("peak alpha freq.", 14, 15), ("1/f exponent + offset", 15, 17),
               ("log power, 1 Hz bins", 17, 61), ("time domain", 61, 68)]


def feature_blocks():
    """Name of the block of every one of the 4028 feature columns (layout of ``HandcraftedWindowFeatures``)."""
    names = []
    per = 68
    core = [None] * per
    for g, a, b in CORE_GROUPS:
        for k in range(a, b):
            core[k] = g
    names += [core[k] for _c in range(N_CHANNELS) for k in range(per)]        # channel-major [C, 68]
    names += [f"covariance, {b}" for b in FILTERBANK for _ in range(171)]     # band-major [6, 171]
    names += [f"{kind}, {b}" for b in BANDS for kind in ("imag. coherence", "coherence") for _ in range(171)]
    return np.array(names)


def draw_pipeline(path):
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#34495e", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.3, color="#34495e"))

    box(0.1, 2.4, 2.0, 1.2, "10 s window\n19 channels\n200 Hz (µV)", "#eef3f6")
    box(2.9, 4.0, 2.6, 1.2, "LaBraM encoder\n(trained)", "#d6eaf8")
    box(2.9, 0.8, 2.6, 1.2, "handcrafted features\n4,028, fixed (Sec. 2-6)", "#fdebd0")
    box(6.1, 0.8, 2.0, 1.2, "standardise (7.2)\nproject 4,028 -> 64\n(7.4), trained", "#fdebd0")
    box(6.1, 4.0, 2.0, 1.2, "class token c\n200", "#d6eaf8")
    box(8.8, 2.4, 1.6, 1.2, "h = [c ; e]\n264 (8.1)", "#e8f8f5")
    box(11.0, 4.0, 2.8, 1.2, "attention score\ns = w' tanh(V h) (8.3)", "#e8f8f5")
    box(11.0, 2.4, 2.8, 1.2, "pool over the bag\nz = sum alpha h (8.5)", "#e8f8f5")
    box(11.0, 0.8, 2.8, 1.2, "linear head\nage = w_h' z + b (8.6)", "#e8f8f5")
    arrow(2.1, 3.2, 2.9, 4.6)
    arrow(2.1, 2.8, 2.9, 1.4)
    arrow(5.5, 4.6, 6.1, 4.6)
    arrow(5.5, 1.4, 6.1, 1.4)
    arrow(8.1, 4.6, 8.8, 3.4)
    arrow(8.1, 1.4, 8.8, 2.6)
    ax.text(7.1, 2.1, "e (64)", ha="center", fontsize=10)
    arrow(10.4, 3.2, 11.0, 4.6)
    arrow(10.4, 3.0, 11.0, 3.0)
    arrow(12.4, 4.0, 12.4, 3.6)
    arrow(12.4, 2.4, 12.4, 2.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def projection_blocks(state, out_dir):
    w = state["window_feat.proj.weight"].float().numpy()                     # [64, 4028]
    energy = (w ** 2).sum(0)                                                 # Eq. (10.1)
    blocks = feature_blocks()
    init = w.shape[0] / w.shape[1]                                           # E||W_j||^2 at initialisation
    rows = []
    for name in pd.unique(blocks):
        m = blocks == name
        rows.append({"block": name, "features": int(m.sum()), "energy_share": float(energy[m].sum() / energy.sum()),
                     "column_share": float(m.mean()), "energy_per_feature_vs_init": float(energy[m].mean() / init)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "projection_blocks.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5), sharey=True)
    y = np.arange(len(df))[::-1]
    axes[0].barh(y, 100 * df.energy_share, color="#2e86c1", label="share of weight energy")
    axes[0].barh(y, 100 * df.column_share, color="none", edgecolor="#e67e22", lw=1.2, label="share of columns")
    axes[0].set_yticks(y, df.block)
    axes[0].set_xlabel("% of the projection")
    axes[0].set_title("Share per block, Eq. (10.2)")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[1].barh(y, df.energy_per_feature_vs_init, color="#27ae60")
    axes[1].axvline(1.0, color="#34495e", lw=1, ls="--")
    axes[1].set_xlabel("mean weight energy per feature / value at initialisation")
    axes[1].set_title("Energy per feature, Eq. (10.3)")
    fig.tight_layout()
    for ax, label in zip(axes, "ab"):
        ax.set_title(f"({label}) {ax.get_title()}")
    fig.savefig(os.path.join(out_dir, "projection_blocks.png"), dpi=150)
    plt.close(fig)
    return df


@torch.no_grad()
def usage(run, cohort, out_dir, n_rec, device):
    from labram.configs.run_configs import FinetuneRunConfig
    from labram.data.eeg_constants import get_channel_indices
    from labram.runs import run_finetune
    cfg = FinetuneRunConfig.load_config(os.path.join(run, "run_config.yaml"), raise_exception=False)
    cfg.model.nb_classes = 1                     # set from the data bundle at run time (scalar age head)
    model = run_finetune.get_model(cfg)
    model.load_state_dict(torch.load(os.path.join(run, "checkpoint-best.pth"), map_location="cpu",
                                     weights_only=False)["model"])
    model.to(device).eval()
    man = json.load(open(os.path.join(cohort, "manifest.json")))
    names = man["channels"]
    keep = [i for i, c in enumerate(names) if c not in cfg.data.drop_channels]
    ch = [names[i] for i in keep]
    ci = get_channel_indices(ch)
    recs = man["recordings"]
    recs = recs if isinstance(recs, list) else list(recs.values())
    by_stem = {r["stem"]: r for r in recs}
    own = json.load(open(os.path.join(run, "eval", "own.json")))
    own = own[next(iter(own))]
    mean_y, sd_y = own["pred_target_stats"]
    preds = own["splits"]["test"]["predictions"]
    meta = json.load(open(os.path.join(cohort, "age_metadata.json")))["recordings"]
    full = [s for s in sorted(preds) if all(by_stem[s]["channel_mask"][i] for i in keep)]
    pick = np.random.default_rng(0).choice(full, size=min(n_rec, len(full)), replace=False)
    win = int(man["window_samples"])
    skip, use = int(cfg.data.trim_start_sec // 10), int(cfg.data.eval_minutes * 6)
    n_c = model.head.in_features - model.window_feat.out_dim
    wh, bh = model.head.weight[0].float(), float(model.head.bias[0])
    pool = model.mil.pool
    rows, parts = [], {"cls": [], "feat": [], "v_cls": [], "v_feat": []}
    for stem in pick:
        arr = np.load(os.path.join(cohort, "recordings", f"{stem}.npy"), mmap_mode="r")
        n = arr.shape[0] // win
        lo, hi = min(skip, n), min(skip + use, n)
        if hi - lo < 1:
            continue
        x = torch.from_numpy(np.asarray(arr[lo * win:hi * win], np.float32)).T[keep]      # [19, W*win] µV
        x4 = (x.reshape(len(keep), hi - lo, 10, win // 10).permute(1, 0, 2, 3) / 100.0).to(device)
        with torch.autocast(device_type=x4.device.type, enabled=x4.device.type == "cuda"):  # as in evaluation
            c = model.forward_features(x4, channel_indices=ci).float()                   # [W, 200]
        e = model.window_feat(x4, ci).float()                                            # [W, 64]
        e0 = model.window_feat.proj.bias.float().expand_as(e)                            # z = 0: training mean
        out = {}
        for tag, ee in (("full", e), ("neutral", e0)):
            h = torch.cat([c, ee], 1)
            s = pool.w[0] @ torch.tanh(pool.V(h)).T                                      # Eq. (8.3)
            a = torch.softmax(s, 0)                                                       # Eq. (8.4)
            out[tag] = mean_y + sd_y * float(a @ (h @ wh) + bh)                          # Eqs. (8.5)-(8.8)
        parts["cls"].append((sd_y * (c @ wh[:n_c])).cpu().numpy())
        parts["feat"].append((sd_y * (e @ wh[n_c:])).cpu().numpy())
        parts["v_cls"].append((c @ pool.V.weight[:, :n_c].T.float()).cpu().numpy())
        parts["v_feat"].append((e @ pool.V.weight[:, n_c:].T.float()).cpu().numpy())
        rows.append({"stem": stem, "age": float(meta[stem]["age"]), "pred_eval": float(preds[stem]),
                     "pred_full": out["full"], "pred_neutral": out["neutral"]})
    df = pd.DataFrame(rows)
    cls, feat = np.concatenate(parts["cls"]), np.concatenate(parts["feat"])
    vc, vf = np.concatenate(parts["v_cls"]), np.concatenate(parts["v_feat"])
    err = lambda p: float(np.abs(df[p] - df.age).mean())                                  # noqa: E731
    summary = {
        "recordings": len(df), "windows": int(len(cls)),
        "max_abs_diff_vs_evaluation_years": float(np.abs(df.pred_full - df.pred_eval).max()),
        "mean_abs_diff_vs_evaluation_years": float(np.abs(df.pred_full - df.pred_eval).mean()),
        "sd_window_pred_cls_part_years": float(cls.std()), "sd_window_pred_feature_part_years": float(feat.std()),
        "corr_cls_feature_parts": float(np.corrcoef(cls, feat)[0, 1]),
        "rms_attention_preact_cls_part": float(np.sqrt((vc ** 2).mean())),
        "rms_attention_preact_feature_part": float(np.sqrt((vf ** 2).mean())),
        "mae_full_years": err("pred_full"), "mae_features_neutral_years": err("pred_neutral"),
        "mean_abs_change_neutral_years": float(np.abs(df.pred_neutral - df.pred_full).mean()),
    }
    pd.Series(summary).to_csv(os.path.join(out_dir, "usage.csv"), header=["value"])
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--recordings", type=int, default=600)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    draw_pipeline(os.path.join(args.out, "pipeline.png"))
    state = torch.load(os.path.join(args.run, "checkpoint-best.pth"), map_location="cpu", weights_only=False)["model"]
    pd.set_option("display.width", 200)
    print(projection_blocks(state, args.out).round(4).to_string(index=False))
    print(json.dumps(usage(args.run, args.cohort, args.out, args.recordings, args.device), indent=1))


if __name__ == "__main__":
    main()
