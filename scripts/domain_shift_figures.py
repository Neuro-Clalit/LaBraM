#!/usr/bin/env python
"""Figures of the CHBP domain-shift analysis (docs/chbp_domain_shift.md).

Inputs are the outputs of ``scripts/external_runs/chbp_domain_shift.sh``:
``features.parquet``, ``ref_stems.txt`` and ``scores/*.json`` under
``--dir``, plus the per-recording predictions of the external evaluation.

* ``chbp_shift_psd.png`` -- (a) median PSD per set; (b) CHBP / reference PSD
  ratio with the amplitude response of CHBP's recording filters;
* ``chbp_shift_amplitude.png`` -- (a) channel std against age; (b) W7's mean
  error per amplitude bin;
* ``chbp_fix_methods.png`` -- (a) MAE and (b) r of E2L under every CHBP fix
  (test-time normalizations and the device-robust model; table ``--fixes``);
* ``chbp_shift_tests.png`` -- (a) mean error and (b) MAE of every transform test
  (W7; the table ``--tests`` has every model and CHBP's two caps).
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import butter, sosfreqz  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labram.eval.age_plots import assert_max_columns, panel_labels  # noqa: E402

# reference categorical slots 1-3 (dataviz skill palette), fixed per entity
COLORS = {"temple": "#2a78d6", "harvard": "#eb6834", "chbp": "#1baf7a"}
LABELS = {"temple": "Temple normal", "harvard": "Harvard normal", "chbp": "CHBP"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
SRC = "/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1"
CHBP = "/data/datasets/EEG-public/CHBP/cohorts/chbp_v1"


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def finish(fig, axes, path):
    fig.tight_layout()
    if len(axes) > 1:
        panel_labels(axes)
    assert_max_columns(fig)
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print("wrote", path)


def set_psd(stems, root):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from make_transformed_cohort import _psd_one
    res = [_psd_one(os.path.join(root, "recordings", s + ".npy")) for s in stems]
    return res[0][0], np.median(np.stack([p for _, p in res]), axis=0)


def fig_psd(d, out):
    feats = pd.read_parquet(os.path.join(d, "features.parquet"))
    ref = [s.strip() for s in open(os.path.join(d, "ref_stems.txt")) if s.strip()]
    sets = {name: [s for s in ref if feats.loc[s, "set"] == name] for name in ("temple", "harvard")}
    psd = {name: set_psd(st, SRC) for name, st in sets.items()}
    psd["chbp"] = set_psd(list(json.load(open(os.path.join(CHBP, "age_metadata.json")))["recordings"]), CHBP)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    ax = axes[0]
    for name in ("temple", "harvard", "chbp"):
        f, p = psd[name]
        sel = (f >= 0.5) & (f <= 80)
        ax.semilogy(f[sel], p[sel], color=COLORS[name], linewidth=2, label=f"{LABELS[name]} (n = "
                    f"{len(sets.get(name, [])) or 250})")
    ax.set_xlabel("frequency (Hz)", color=INK)
    ax.set_ylabel("median PSD (µV²/Hz)", color=INK)
    ax.set_title("Median spectrum, common average reference")
    ax.legend(frameon=False)
    style(ax)
    ax = axes[1]
    f = psd["chbp"][0]
    sel = (f >= 0.5) & (f <= 80)
    for name in ("temple", "harvard"):
        ax.semilogy(f[sel], psd["chbp"][1][sel] / psd[name][1][sel], color=COLORS[name], linewidth=2,
                    label=f"CHBP / {LABELS[name]}")
    sos = np.vstack([butter(1, 0.5, "highpass", fs=200, output="sos"),
                     butter(2, 50, "lowpass", fs=200, output="sos")])
    w, h = sosfreqz(sos, worN=4096, fs=200)
    ax.semilogy(w[(w >= 0.5) & (w <= 80)], np.abs(h[(w >= 0.5) & (w <= 80)]) ** 2, color=INK,
                linestyle="--", linewidth=1.5, label="CHBP recording filters, |H(f)|² (spec)")
    ax.axhline(1, color=MUTED, linewidth=1)
    ax.set_xlabel("frequency (Hz)", color=INK)
    ax.set_ylabel("power ratio", color=INK)
    ax.set_title("CHBP relative to the training data")
    ax.legend(frameon=False)
    style(ax)
    finish(fig, list(axes), os.path.join(out, "chbp_shift_psd.png"))


def fig_amplitude(d, out, pred_val, pred_chbp, model="W7"):
    feats = pd.read_parquet(os.path.join(d, "features.parquet"))
    pv = pd.Series(json.load(open(pred_val))[model]["splits"]["val"]["predictions"])
    pc = pd.Series(json.load(open(pred_chbp))[model]["splits"]["test"]["predictions"])
    df = feats.join(pd.concat([pv, pc]).rename("pred"), how="inner")
    df["err"] = df.pred - df.age
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    ax = axes[0]
    for name in ("temple", "harvard", "chbp"):
        s = df[df.set == name]
        ax.scatter(s.age, s.std_uv, s=10 if name != "chbp" else 16, color=COLORS[name],
                   alpha=0.45 if name != "chbp" else 0.85, edgecolors="none",
                   label=f"{LABELS[name]} (n = {len(s)})")
    ax.set_yscale("log")
    ax.set_xlabel("age (years)", color=INK)
    ax.set_ylabel("channel std (µV, robust)", color=INK)
    ax.set_title("Amplitude against age")
    ax.legend(frameon=False, markerscale=2)
    style(ax)
    ax = axes[1]
    bins = [0, 5, 6, 7, 8, 10, 12, 15, 20, 1e9]
    names = ["<5", "5-6", "6-7", "7-8", "8-10", "10-12", "12-15", "15-20", ">20"]
    for name, s in (("temple", df[df.set == "temple"]), ("harvard", df[df.set == "harvard"]),
                    ("chbp", df[df.set == "chbp"])):
        g = s.groupby(pd.cut(s.std_uv, bins), observed=False).err.agg(["mean", "size"])
        keep = g["size"] >= 8
        x = np.arange(len(names))[keep.to_numpy()]
        ax.plot(x, g["mean"][keep], color=COLORS[name], linewidth=2, marker="o", markersize=7,
                label=LABELS[name])
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(range(len(names)), names)
    ax.set_xlabel("channel std bin (µV); bins with ≥ 8 recordings", color=INK)
    ax.set_ylabel(f"{model} mean error, predicted − true (years)", color=INK)
    ax.set_title("Over-prediction against amplitude")
    ax.legend(frameon=False)
    style(ax)
    finish(fig, list(axes), os.path.join(out, "chbp_shift_amplitude.png"))


TESTS = [  # (score file, set, label)
    ("ref_raw", "Temple+Harvard", "as recorded"),
    ("ref_x0.62", "Temple+Harvard", "× 0.62 (CHBP amplitude)"),
    ("ref_hp0.5", "Temple+Harvard", "high-pass 0.5 Hz"),
    ("ref_lp50", "Temple+Harvard", "low-pass 50 Hz"),
    ("ref_band", "Temple+Harvard", "band 0.5-50 Hz"),
    ("ref_x0.62_band", "Temple+Harvard", "× 0.62 + band 0.5-50 Hz"),
    ("ref_peramp", "Temple+Harvard", "per-recording amplitude"),
    ("ref_gsf", "Temple+Harvard", "per-recording GSF"),
    ("chbp_raw", "CHBP", "as recorded"),
    ("chbp_x1.615", "CHBP", "× 1.615 (training amplitude)"),
    ("chbp_peramp", "CHBP", "per-recording amplitude"),
    ("chbp_gsf", "CHBP", "per-recording GSF"),
    ("chbp_psdmatch", "CHBP", "spectrum matched"),
]


def tests_table(d, out_csv, chbp_raw, encoding_csv, models=("E2L", "L1", "W4", "W7")):
    """Bias / MAE / r of every transform test, per model and for CHBP's two caps."""
    ages = {s: r["age"] for s, r in json.load(open(os.path.join(SRC, "age_metadata.json")))["recordings"].items()}
    ages.update({s: r["age"] for s, r in json.load(open(os.path.join(CHBP, "age_metadata.json")))["recordings"].items()})
    enc = pd.read_csv(encoding_csv).set_index("stem")["enc"]
    rows = []
    for name, group, label in TESTS:
        path = chbp_raw if name == "chbp_raw" else os.path.join(d, "scores", name + ".json")
        if not os.path.exists(path):
            continue
        runs = json.load(open(path))
        for m in models:
            p = pd.Series(runs[m]["splits"]["test"]["predictions"])
            y = np.array([ages[s] for s in p.index])
            subsets = [("all", np.ones(len(p), bool))]
            if group == "CHBP":
                cap = enc.reindex(p.index).to_numpy()
                subsets += [("64-ch cap", cap == "12-bit"), ("120-ch cap", cap == "16-bit")]
            for sub, mask in subsets:
                e = p.to_numpy()[mask] - y[mask]
                rows.append({"test": name, "set": group, "label": label, "subset": sub, "model": m,
                             "n": int(mask.sum()), "bias": e.mean(), "mae": np.abs(e).mean(),
                             "r": np.corrcoef(p.to_numpy()[mask], y[mask])[0, 1]})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


S = "/data/runs/heedb/domain_shift/scores"
F = "/data/runs/chbp_fix/scores"
FC = "/data/runs/chbp_fix/scores/e2l_band45_amp_cont_20261112_001004_075"   # 4 + 6 epochs
EXT = "/data/runs/heedb/external_eval"
FIXES = [  # (label, kind, [score files], model key)  -- CHBP, first 5 minutes; no CHBP labels used
    ("as recorded", "test-time", [f"{EXT}/chbp_5min.json"], None),
    ("global gain x 1.615", "test-time", [f"{S}/chbp_x1.615.json"], None),
    ("per-recording amplitude", "test-time", [f"{S}/chbp_peramp.json"], None),
    ("per-recording GSF", "test-time", [f"{S}/chbp_gsf.json"], None),
    ("spectrum matching, whole cohort", "test-time", [f"{S}/chbp_psdmatchU.json"], None),
    ("spectrum matching per cap", "test-time", [f"{S}/chbp64_psdmatchU.json", f"{S}/chbp120_psdmatchU.json"], None),
    ("spectrum matching per cap and channel", "test-time",
     [f"{S}/chbp64_psdmatchchU.json", f"{S}/chbp120_psdmatchchU.json"], None),
    ("spectrum matching 0.5-45 Hz + refill (age-matched ref.)", "test-time", [f"{S}/chbp_psdmatch45_fill.json"], None),
    ("device-robust E2L (4 epochs), as recorded", "training", [f"{F}/chbp_raw.json"], "E2L-robust"),
    ("device-robust E2L (4 epochs), per-cap matching", "training",
     [f"{F}/chbp64_psdmatchU.json", f"{F}/chbp120_psdmatchU.json"], "E2L-robust"),
    ("device-robust E2L (+6 epochs), as recorded", "training", [f"{FC}/chbp_raw.json"], "E2L-robust"),
    ("device-robust E2L (+6 epochs), per-cap matching", "training",
     [f"{FC}/chbp64_psdmatchU.json", f"{FC}/chbp120_psdmatchU.json"], "E2L-robust"),
]


def fixes_table(out_csv, encoding_csv, models=("E2L", "L1", "W4", "W7")):
    """MAE / bias / r of every CHBP fix, per model, overall and per cap."""
    ages = {s: r["age"] for s, r in json.load(open(os.path.join(CHBP, "age_metadata.json")))["recordings"].items()}
    enc = pd.read_csv(encoding_csv).set_index("stem")["enc"]
    rows = []
    for label, kind, paths, only in FIXES:
        if not all(os.path.exists(p) for p in paths):
            continue
        for m in ([only] if only else models):
            pred = {}
            for path in paths:
                pred.update(json.load(open(path))[m]["splits"]["test"]["predictions"])
            p = pd.Series(pred)
            y = np.array([ages[s] for s in p.index])
            cap = enc.reindex(p.index).to_numpy()
            row = {"method": label, "kind": kind, "model": "E2L" if m == "E2L-robust" else m, "n": len(p)}
            for sub, mask in (("", np.ones(len(p), bool)), ("_64ch", cap == "12-bit"), ("_120ch", cap == "16-bit")):
                e = p.to_numpy()[mask] - y[mask]
                row.update({f"mae{sub}": np.abs(e).mean(), f"bias{sub}": e.mean(),
                            f"r{sub}": np.corrcoef(p.to_numpy()[mask], y[mask])[0, 1]})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def fig_fixes(table, out):
    t = pd.read_csv(table)
    t = t[t.model == "E2L"].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.5 * len(t) + 1.8))
    y = np.arange(len(t))[::-1]
    color = [COLORS["temple"] if k == "training" else COLORS["chbp"] for k in t["kind"]]
    for ax, col, xlabel, title, ref in ((axes[0], "mae", "MAE (years)", "MAE", 7.27),
                                        (axes[1], "r", "Pearson r", "Correlation with age", None)):
        ax.barh(y, t[col], color=color, height=0.6)
        for yi, v in zip(y, t[col]):
            ax.text(v + 0.02 * t[col].max(), yi, f"{v:.2f}", va="center", ha="left", color=INK, fontsize=9)
        if ref is not None:
            ax.axvline(ref, color=MUTED, linestyle="--", linewidth=1)
            ax.text(ref, len(t) - 0.4, " predict the mean", color=MUTED, fontsize=9, va="bottom")
        ax.set_xlim(0, t[col].max() * 1.18)
        ax.set_yticks(y, t["method"])
        ax.set_xlabel(xlabel, color=INK)
        ax.set_title(title)
        style(ax)
    finish(fig, list(axes), os.path.join(out, "chbp_fix_methods.png"))


def fig_tests(table, out):
    t = pd.read_csv(table)
    t = t[(t.model == "W7") & (t.subset == "all")].reset_index(drop=True)
    t["label"] = t["set"] + ": " + t["label"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.42 * len(t) + 1.8))
    y = np.arange(len(t))[::-1]
    for ax, col, xlabel, title in ((axes[0], "bias", "mean error, predicted − true (years)", "Bias"),
                                   (axes[1], "mae", "MAE (years)", "MAE")):
        color = [COLORS["chbp"] if s.startswith("CHBP") else COLORS["temple"] for s in t["set"]]
        ax.barh(y, t[col], color=color, height=0.6)
        for yi, v in zip(y, t[col]):
            ax.text(v + (0.1 if v >= 0 else -0.1), yi, f"{v:.2f}", va="center",
                    ha="left" if v >= 0 else "right", color=INK, fontsize=9)
        ax.axvline(0, color=MUTED, linewidth=1)
        lo, hi = min(0.0, t[col].min()), max(0.0, t[col].max())
        pad = 0.18 * (hi - lo)                       # room for the value labels inside the axes
        ax.set_xlim(lo - (pad if lo < 0 else 0), hi + pad)
        ax.set_yticks(y, t["label"])
        ax.set_xlabel(xlabel, color=INK)
        ax.set_title(title)
        style(ax)
    finish(fig, list(axes), os.path.join(out, "chbp_shift_tests.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default="/data/runs/heedb/domain_shift")
    ap.add_argument("--out", default="docs/figures/chbp")
    ap.add_argument("--pred-val", default="/data/runs/heedb/external_eval/own_all_val_W7.json")
    ap.add_argument("--pred-chbp", default="/data/runs/heedb/external_eval/chbp_all.json")
    ap.add_argument("--pred-chbp-5min", default="/data/runs/heedb/external_eval/chbp_5min.json",
                    help="CHBP as recorded, scored like the tests (first 5 min)")
    ap.add_argument("--tests", default=None, help="write the transform-test table to this CSV and plot it")
    ap.add_argument("--encoding", default="/data/runs/heedb/domain_shift/chbp_edf_encoding.csv")
    ap.add_argument("--fixes", default=None, help="write the CHBP-fix table to this CSV and plot it")
    ap.add_argument("--only", nargs="*", default=["psd", "amplitude", "tests", "fixes"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if "psd" in args.only:
        fig_psd(args.dir, args.out)
    if "amplitude" in args.only:
        fig_amplitude(args.dir, args.out, args.pred_val, args.pred_chbp)
    if "fixes" in args.only and args.fixes:
        fixes_table(args.fixes, args.encoding)
        fig_fixes(args.fixes, args.out)
    if "tests" in args.only and args.tests:
        tests_table(args.dir, args.tests, args.pred_chbp_5min, args.encoding)
        fig_tests(args.tests, args.out)


if __name__ == "__main__":
    main()
