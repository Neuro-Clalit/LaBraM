#!/usr/bin/env python
"""Zero-shot brain-age evaluation on external cohorts: metrics, figures and preprocessing QC.

Reads ``scripts/eval_age_by_cohort.py`` outputs (one JSON per cohort, every run's
per-recording test predictions) and the cohorts' manifests, and writes

* ``metrics.csv`` -- per cohort x model x group: n, MAE, mean signed error
  (bias), Pearson r, R^2; the predict-the-mean baseline (cohort mean age); and a
  within-cohort linear recalibration ``age ~ a * pred + b`` cross-fitted over
  5 subject-grouped folds (uses the cohort's ages, so it is *not* zero-shot);
* ``qc.csv`` (external cohorts, per recording) and ``qc_summary.csv`` (per
  source, quartiles; with a 100-recording TUH / HEEDB reference sample of
  normal reports, aggregated only): median channel std, relative band powers and the
  occipital alpha-peak frequency of the first 5 min after the 60 s trim, after
  a common average reference (as the models see it);
* figures ``external_pred_vs_age.png`` and ``external_qc_psd.png``.

    python scripts/external_eval_report.py \\
        --eval lemon=/data/runs/heedb/external_eval/lemon_5min.json \\
        --eval mipdb=/data/runs/heedb/external_eval/mipdb_5min.json \\
        --cohort lemon=/data/datasets/EEG-public/LEMON/cohorts/lemon_v1 \\
        --cohort mipdb=/data/datasets/EEG-public/MIPDB/cohorts/mipdb_v1 \\
        --out-dir docs/data/external_eval --fig-dir docs/figures
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd

REFERENCE = "/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1"
EEG19 = [i for i in range(23) if i not in (16, 17, 21, 22)]      # drop A1, A2, T1, T2
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
SMALL_CELL = 10


def groups(cohort, meta):
    """(name, mask) subgroups of a cohort's recordings (``meta`` indexed by stem)."""
    age = meta["age"]
    out = [("all", np.ones(len(meta), bool))]
    if cohort == "lemon":
        out += [("young (20-40)", (age < 40).to_numpy()), ("old (55-80)", (age >= 55).to_numpy())]
    if cohort == "mipdb":
        out += [("6-12", (age < 13).to_numpy()), ("13-17", ((age >= 13) & (age < 18)).to_numpy()),
                ("adults 18-44", (age >= 18).to_numpy()),
                ("controls (DX 0)", (meta["dx_status"] == 0).to_numpy())]
    if cohort == "chbp":
        out += [("18-29", (age < 30).to_numpy()), ("30-39", ((age >= 30) & (age < 40)).to_numpy()),
                ("40-68", (age >= 40).to_numpy())]
    return out


def crossfit_recalibration(pred, age, subjects, folds=5, seed=0):
    """Out-of-fold ``a * pred + b`` with (a, b) fitted on the other folds (OLS of age on pred)."""
    rng = np.random.default_rng(seed)
    subs = np.array(sorted(set(subjects)))
    fold_of = dict(zip(subs, rng.permutation(len(subs)) % folds))
    f = np.array([fold_of[s] for s in subjects])
    out = np.empty_like(pred)
    for k in range(folds):
        a, b = np.polyfit(pred[f != k], age[f != k], 1)
        out[f == k] = a * pred[f == k] + b
    return out


def metrics(pred, age):
    err = pred - age
    ss_tot = ((age - age.mean()) ** 2).sum()
    return {"mae": np.abs(err).mean(), "bias": err.mean(),
            "r": np.corrcoef(pred, age)[0, 1] if pred.std() > 0 and age.std() > 0 else np.nan,
            "r2": 1 - (err ** 2).sum() / ss_tot if ss_tot > 0 else np.nan}


def cohort_meta(cohort_dir):
    m = json.load(open(os.path.join(cohort_dir, "manifest.json")))
    return pd.DataFrame(m["recordings"]).set_index("stem")


def eval_tables(evals, cohorts):
    rows, preds = [], []
    for name, path in evals.items():
        res = json.load(open(path))
        meta = cohort_meta(cohorts[name])
        for model, r in res.items():
            p = pd.Series(r["splits"]["test"]["predictions"])
            d = meta.join(p.rename("pred"), how="inner")
            age, pred = d["age"].to_numpy(float), d["pred"].to_numpy(float)
            recal = crossfit_recalibration(pred, age, d["subject"].to_numpy())
            preds.append(d.assign(cohort=name, model=model, pred_recal=recal)[
                ["cohort", "model", "age", "pred", "pred_recal"]].reset_index())
            for g, mask in groups(name, d):
                if mask.sum() < SMALL_CELL:
                    continue
                m0 = metrics(pred[mask], age[mask])
                rows.append({"cohort": name, "model": model, "group": g, "n": int(mask.sum()),
                             "age_mean": age[mask].mean(), **m0,
                             "mae_mean_baseline": np.abs(age[mask] - age[mask].mean()).mean(),
                             "mae_recal_cv": np.abs(recal[mask] - age[mask]).mean()})
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


# ------------------------------------------------------------------ signal QC
def recording_qc(path, mask, minutes=5.0, trim_windows=6):
    from scipy.signal import welch
    x = np.load(path, mmap_mode="r")
    n = x.shape[0] // 2000
    sel = range(min(trim_windows, max(n - 1, 0)), min(n, trim_windows + int(minutes * 6)))
    cols = [i for i in EEG19 if mask[i]]
    w = np.stack([np.asarray(x[k * 2000:(k + 1) * 2000, cols], dtype=np.float32) for k in sel])
    w = w - w.mean(axis=2, keepdims=True)
    f, p = welch(w, fs=200, nperseg=400, axis=1)
    psd = np.median(p, axis=(0, 2))
    tot = psd[(f >= 1) & (f < 45)].sum()
    occ = [cols.index(i) for i in (8, 9) if i in cols]
    fo, po = welch(w[..., occ], fs=200, nperseg=800, axis=1)
    po = np.median(po, axis=(0, 2))
    a = (fo >= 6) & (fo <= 14)
    out = {"std_uv": float(np.median(w.std(axis=1))), "alpha_peak_hz": float(fo[a][po[a].argmax()]),
           "frac_abs_gt_300uv": float((np.abs(w) > 300).mean())}
    out.update({b: float(psd[(f >= lo) & (f < hi)].sum() / tot) for b, (lo, hi) in BANDS.items()})
    return out, f, psd


def qc_table(cohorts, n_ref=100, seed=0):
    rows, spectra = [], {}
    for name, cdir in cohorts.items():
        meta = cohort_meta(cdir)
        acc = []
        for stem, r in meta.iterrows():
            q, f, psd = recording_qc(os.path.join(cdir, "recordings", f"{stem}.npy"), r["channel_mask"])
            rows.append({"source": name, "stem": stem, **q})
            acc.append(psd)
        spectra[name] = (f, np.median(acc, axis=0))
    m = json.load(open(os.path.join(REFERENCE, "manifest.json")))["recordings"]
    lab = json.load(open(os.path.join(REFERENCE, "age_metadata.json")))["recordings"]
    rng = random.Random(seed)
    for corpus, name in (("heedb", "HEEDB normal (ref)"), ("tuh", "TUH normal (ref)")):
        pool = [r for r in m if (r.get("corpus") == "heedb") == (corpus == "heedb")
                and lab.get(r["stem"], {}).get("label") == "normal" and r["n_windows"] >= 36]
        acc = []
        for r in rng.sample(pool, min(n_ref, len(pool))):
            q, f, psd = recording_qc(os.path.join(REFERENCE, "recordings", f"{r['stem']}.npy"), r["channel_mask"])
            rows.append({"source": name, "stem": r["stem"], **q})
            acc.append(psd)
        spectra[name] = (f, np.median(acc, axis=0))
    return pd.DataFrame(rows), spectra


def plot_preds(preds, models, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from labram.eval.age_plots import assert_max_columns, panel_labels
    cohorts = list(dict.fromkeys(preds["cohort"]))
    fig, axes = plt.subplots(len(cohorts), 2, figsize=(14, 4.5 * len(cohorts)), squeeze=False)
    for i, c in enumerate(cohorts):
        for j, m in enumerate(models):
            ax = axes[i][j]
            d = preds[(preds.cohort == c) & (preds.model == m)]
            ax.scatter(d.age, d.pred, s=12, alpha=0.6, label="zero-shot")
            ax.scatter(d.age, d.pred_recal, s=10, alpha=0.5, marker="x", label="recalibrated (5-fold CV)")
            lo, hi = 0, max(90, d.age.max() + 5)
            ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.8)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_xlabel("true age (y)")
            ax.set_ylabel("predicted age (y)")
            ax.set_title(f"{c}: {m}")
            ax.legend(fontsize=8)
    fig.tight_layout()
    panel_labels(list(axes.flat))
    assert_max_columns(fig)
    path = os.path.join(fig_dir, "external_pred_vs_age.png")
    fig.savefig(path, dpi=120)
    print(path)


def plot_psd(spectra, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    for name, (f, p) in spectra.items():
        keep = (f >= 1) & (f <= 45)
        ax.semilogy(f[keep], p[keep], label=name, lw=1.6 if "ref" in name else 2.2,
                    ls="--" if "ref" in name else "-")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("median PSD (µV²/Hz), common average reference")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(fig_dir, "external_qc_psd.png")
    fig.savefig(path, dpi=120)
    print(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval", action="append", required=True, metavar="COHORT=JSON")
    ap.add_argument("--cohort", action="append", required=True, metavar="COHORT=DIR")
    ap.add_argument("--plot-models", nargs=2, default=["E2L", "W4"])
    ap.add_argument("--no-qc", action="store_true")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fig-dir", default=None)
    args = ap.parse_args()
    evals = dict(e.split("=", 1) for e in args.eval)
    cohorts = dict(c.split("=", 1) for c in args.cohort)
    os.makedirs(args.out_dir, exist_ok=True)
    table, preds = eval_tables(evals, cohorts)
    table.to_csv(os.path.join(args.out_dir, "metrics.csv"), index=False)
    pd.set_option("display.width", 250)
    print(table.round(2).to_string(index=False))
    if args.fig_dir:
        plot_preds(preds, args.plot_models, args.fig_dir)
    if not args.no_qc:
        qc, spectra = qc_table(cohorts)
        # per recording only for the external cohorts: the TUH / HEEDB reference
        # rows are aggregated (HEEDB DUA: no per-session data in the repo)
        qc[~qc.source.str.contains("(ref)", regex=False)].to_csv(
            os.path.join(args.out_dir, "qc.csv"), index=False)
        cols = [c for c in qc.columns if c not in ("source", "stem")]
        summary = qc.groupby("source")[cols].quantile([0.25, 0.5, 0.75]).unstack().round(3)
        summary.insert(0, "n", qc.groupby("source").size())
        summary.to_csv(os.path.join(args.out_dir, "qc_summary.csv"))
        print(summary.to_string())
        if args.fig_dir:
            plot_psd(spectra, args.fig_dir)


if __name__ == "__main__":
    main()
