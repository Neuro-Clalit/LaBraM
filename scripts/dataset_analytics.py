#!/usr/bin/env python
"""Dataset analytics for the technical report (Section 3.5 of ``docs/tech_report/``):
demographics and label-free signal features of every report cohort, one table and
one figure, aggregates only (TDBRAIN and HEEDB are under data use agreements).

Signal features are the per-recording features of ``scripts/domain_shift_analysis.py``
(what the models see: 19 channels, common average reference, µV at 200 Hz; per 10 s
window, then the median over windows), read from its ``features.parquet`` outputs.
Temple and Harvard are represented by its reference sets: validation recordings with a
normal EEG report, aged 18-68. Every statistic is a median with the interquartile range
over recordings (Dortmund: over runs).

    python scripts/dataset_analytics.py --out docs/data/external_eval/tech_report/dataset_analytics.csv \\
        --figure docs/tech_report/fig_dataset_analytics.png
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

E = "/data/datasets/EEG-public"
RUNS = "/data/runs/heedb"
TRAIN = "/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1"
# name, label, cohort dir (None: the training cohort's test split of that corpus), features parquet, set name
COHORTS = [
    ("temple", "Temple", None, f"{RUNS}/domain_shift/features.parquet", "temple"),
    ("harvard", "Harvard", None, f"{RUNS}/domain_shift/features.parquet", "harvard"),
    ("tdbrain", "TDBRAIN", f"{E}/TDBRAIN/cohorts/tdbrain_v1", f"{RUNS}/domain_shift_tdbrain/features.parquet", "tdbrain"),
    ("lemon", "LEMON", f"{E}/LEMON/cohorts/lemon_v1", f"{RUNS}/domain_shift_lemon/features.parquet", "lemon"),
    ("dortmund", "Dortmund", f"{E}/DORTMUND/cohorts/dortmund_v1", f"{RUNS}/domain_shift_dortmund/features.parquet", "dortmund"),
    ("srm", "SRM", f"{E}/SRM/cohorts/srm_v1", f"{RUNS}/domain_shift_open3/features.parquet", "srm"),
    ("chbp", "CHBP", f"{E}/CHBP/cohorts/chbp_v1", f"{RUNS}/domain_shift/features.parquet", "chbp"),
]
FEATURES = [  # column, label, scale
    ("std_uv", "channel SD (µV)", 1.0),
    ("artifact_window_share", "windows with a channel SD > 100 µV (%)", 100.0),
    ("rel_delta", "relative delta 1–4 Hz (%)", 100.0),
    ("rel_theta", "relative theta 4–8 Hz (%)", 100.0),
    ("rel_alpha", "relative alpha 8–13 Hz (%)", 100.0),
    ("rel_beta", "relative beta 13–30 Hz (%)", 100.0),
    ("abs_45_70", "power 45–70 Hz (µV²/Hz)", 1.0),
    ("alpha_peak_hz", "occipital alpha peak (Hz)", 1.0),
    ("aperiodic_exponent", "aperiodic exponent", 1.0),
]
TRIMMED = {"lemon", "chbp"}                 # external cohorts scored after the 60 s trims
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]


def scored_units(name, cdir):
    """Manifest rows of the scored adults (Temple / Harvard: the test split of the training cohort)."""
    from external_cohort_table import load_meta
    if cdir is None:
        meta = load_meta(TRAIN, name, 18.0)
        split = json.load(open(os.path.join(TRAIN, "age_split.json")))["files"]["test"]
        test = {f.split("/")[-1].rsplit("_", 1)[0] for f in split}
        return meta[meta.index.isin(test)]
    meta = load_meta(cdir, None, 18.0)
    if name in TRIMMED:                      # scored after 60 s trims: recordings with windows left
        meta = meta[meta["n_windows"] > 12]
    return meta


def describe(name, label, cdir, feats):
    m = scored_units(name, cdir)
    units = m.groupby(["subject", "session"]).first() if "task" in m else m
    row = {"cohort": label, "units": len(units), "unit": "sessions" if "task" in m else "recordings",
           "subjects": m["subject"].nunique(), "age_mean": units["age"].mean(), "age_sd": units["age"].std(),
           "age_min": units["age"].min(), "age_max": units["age"].max(),
           "men_share": (units["sex"].astype(str).str.upper().str[0] == "M").mean() if "sex" in units else np.nan,
           "healthy_share": units["healthy"].mean(), "minutes_median": m["duration_sec"].median() / 60
           if "duration_sec" in m else np.nan,
           "sfreq_native": m["sfreq_native"].median() if "sfreq_native" in m else np.nan,
           "feature_n": len(feats)}
    for col, _, scale in FEATURES:
        q = feats[col].quantile([0.25, 0.5, 0.75]).to_numpy() * scale
        row[f"{col}_q1"], row[f"{col}_median"], row[f"{col}_q3"] = q
    return row, units["age"].to_numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--figure", required=True)
    args = ap.parse_args()
    rows, ages, feats_all = [], {}, {}
    for name, label, cdir, fpath, setname in COHORTS:
        f = pd.read_parquet(fpath)
        f = f[(f["set"] == setname) & (f["age"] >= 18)]
        row, age = describe(name, label, cdir, f)
        rows.append(row)
        ages[label], feats_all[label] = age, f
        print(f"{label}: {row['units']} {row['unit']}, features on {len(f)}", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(args.out)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from labram.eval.age_plots import assert_max_columns, panel_labels
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13})
    labels = [r["cohort"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    panels = [("age", "age of the scored adults (years)", False),
              ("std_uv", "channel SD (µV)", True),
              ("rel_alpha", "relative alpha power 8–13 Hz (%)", False),
              ("abs_45_70", "power 45–70 Hz (µV²/Hz)", True)]
    for ax, (col, title, logy) in zip(axes.flat, panels):
        data = [ages[c] if col == "age" else feats_all[c][col].to_numpy() * (100 if col == "rel_alpha" else 1)
                for c in labels]
        bp = ax.boxplot(data, widths=0.6, showfliers=False, patch_artist=True, whis=(5, 95))
        for patch, color in zip(bp["boxes"], COLORS):
            patch.set(facecolor=color, alpha=0.75, edgecolor="#1f1f1e")
        for med in bp["medians"]:
            med.set(color="#1f1f1e", lw=1.6)
        ax.set_xticks(range(1, len(labels) + 1), labels, rotation=0, fontsize=10.5)
        ax.set_title(title)
        if logy:
            ax.set_yscale("log")
        if col == "std_uv":
            from matplotlib.ticker import FixedLocator, NullLocator
            ax.yaxis.set_major_locator(FixedLocator([3, 5, 10, 20, 30]))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.set_yticklabels(["3", "5", "10", "20", "30"])
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    panel_labels(list(axes.flat))
    assert_max_columns(fig)
    fig.savefig(args.figure, dpi=150)
    print(args.figure)


if __name__ == "__main__":
    main()
