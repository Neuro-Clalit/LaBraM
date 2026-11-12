#!/usr/bin/env python
"""Handcrafted-feature brain-age models across domains (Temple / Harvard -> CHBP).

``extract`` runs the handcrafted-feature extractors of ``labram/features``
(``tuab.extract_recording_cohort``: time, Welch spectral + 1/f, 1 Hz log-PSD
bins, DWT, wavelet packets, multiscale permutation entropy, complex-Morlet CWT,
STFT statistics, band-power dynamics, cepstral coefficients, filter-bank
covariances; ``tuab.extract_recording_scattering``: wavelet scattering J = 7 /
9) on recordings of an npy cohort. The preparation is the toolkit's reference
one: 19 channels, common average reference, 1-45 Hz, skip the first minute,
median over 30 clean 10 s windows.

``model`` trains ridge regressions of age on the training set (Temple + Harvard
normal reports; never CHBP) for every feature family and for all families
together, and scores every other set (in-domain validation, CHBP and its
label-free normalizations). ``--gain-invariant`` additionally removes, per
recording, the mean of every log-power family (the feature-domain global scale
factor), which makes those families insensitive to a global gain.

    python scripts/hc_domain_features.py sample --src COHORT --n 1200 --out train_stems.txt
    python scripts/hc_domain_features.py extract --cohort COHORT --stems train_stems.txt --out train.npz
    python scripts/hc_domain_features.py model --train train.npz --test val=val.npz --test chbp=chbp.npz
    python scripts/hc_domain_features.py domain --ref val.npz --test chbp=chbp.npz --out domain_auc.csv

``domain`` measures how separable two sets are per feature family: the
5-fold cross-validated ROC AUC of an L2 logistic regression (on standardized
features, PCA to 50 components) between the reference set (in-domain, age
matched) and a test set. 0.5 = the family looks the same in both domains.
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAMILIES = ["time", "spectral", "psd_bins", "dwt_db4_L6", "wpd_db4_L5", "mspe", "cwt_cmor1.5-1.0",
            "stft", "stft_dyn", "cepstral", "scat_J7_Q8", "scat_J9_Q8"]
# columns that scale with a global gain (log power / log amplitude); everything else is gain-free
LOGPOWER = re.compile(r"^(log_var|log_line_length|log_abs_|aperiodic_offset|log_psd_|log_E_|cwt_mean_|"
                      r"stft_mean_|cep_mean_c0|S1_)")


# ------------------------------------------------------------------ extraction
def _extract(path):
    from labram.features import tuab
    out = tuab.extract_recording_cohort(path)
    out.update(tuab.extract_recording_scattering(path))
    return out


def extract(cohort, stems, out_path, jobs):
    paths = [os.path.join(cohort, "recordings", s + ".npy") for s in stems]
    feats = {f: [] for f in FAMILIES}
    fbcov, ok = [], []
    with ProcessPoolExecutor(jobs) as pool:
        for i, (stem, res) in enumerate(zip(stems, pool.map(_extract, paths, chunksize=2)), 1):
            for f in FAMILIES:
                feats[f].append(res[f"rec/{f}"])
            fbcov.append(res["fbcov"])
            ok.append(stem)
            if i % 100 == 0:
                print(f"  {i}/{len(stems)}", flush=True)
    meta = json.load(open(os.path.join(cohort, "age_metadata.json")))["recordings"]
    np.savez_compressed(out_path, stems=np.array(ok), ages=np.array([meta[s]["age"] for s in ok], float),
                        fbcov=np.stack(fbcov), **{f"f_{k}": np.stack(v) for k, v in feats.items()})
    print(f"{len(ok)} recordings -> {out_path}")


def sample_train(src, n, seed=0, min_age=18):
    """N normal-report recordings per corpus from the cohort's TRAIN split (random, seeded)."""
    man = {r["stem"]: r for r in json.load(open(os.path.join(src, "manifest.json")))["recordings"]}
    meta = json.load(open(os.path.join(src, "age_metadata.json")))["recordings"]
    train = {f.split("/")[-1].rsplit("_", 1)[0]
             for f in json.load(open(os.path.join(src, "age_split.json")))["files"]["train"]}
    rng = np.random.default_rng(seed)
    out = []
    for heedb in (False, True):
        cand = sorted(s for s in train if meta.get(s, {}).get("label") == "normal"
                      and meta[s].get("age") is not None and meta[s]["age"] >= min_age
                      and (man[s].get("corpus") == "heedb") == heedb)
        out += list(rng.choice(cand, size=min(n, len(cand)), replace=False))
    return out


# ------------------------------------------------------------------ modelling
def names():
    from labram.features import tuab
    n = {**tuab.feature_names(), **tuab.spectrogram_names(), **tuab.scattering_names()}
    return {f: list(n[f]) for f in FAMILIES}


def design(npz, family_names, families, gain_invariant=False):
    """[n, 19 * F] per family (channel-major), optionally gain-centred per recording."""
    blocks = []
    for fam in families:
        x = np.asarray(npz[f"f_{fam}"], dtype=np.float64)          # [n, 19, F]
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if gain_invariant:
            cols = [i for i, c in enumerate(family_names[fam]) if LOGPOWER.match(c)]
            if cols:
                x = x.copy()
                x[:, :, cols] -= x[:, :, cols].mean(axis=(1, 2), keepdims=True)
        blocks.append(x.reshape(len(x), -1))
    return np.concatenate(blocks, axis=1)


def fit_predict(xtr, ytr, xte_list):
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 25)))
    model.fit(xtr, ytr)
    return [model.predict(x) for x in xte_list], float(model[-1].alpha_)


def scores(y, p):
    e = p - y
    return {"n": len(y), "mae": float(np.abs(e).mean()), "bias": float(e.mean()),
            "r": float(np.corrcoef(y, p)[0, 1])}


def model(train_path, tests, out_csv, gain_variants):
    fam_names = names()
    tr = np.load(train_path)
    te = {k: np.load(v) for k, v in tests}
    groups = [[f] for f in FAMILIES] + [["spectral", "psd_bins", "dwt_db4_L6", "wpd_db4_L5", "cwt_cmor1.5-1.0",
                                          "stft", "stft_dyn", "cepstral", "time", "mspe"],
                                         FAMILIES]
    labels = FAMILIES + ["all but scattering", "all"]
    rows = []
    for gi in gain_variants:
        for label, fams in zip(labels, groups):
            xtr = design(tr, fam_names, fams, gi)
            preds, alpha = fit_predict(xtr, tr["ages"], [design(z, fam_names, fams, gi) for z in te.values()])
            for (name, z), p in zip(te.items(), preds):
                rows.append({"features": label, "gain_invariant": gi, "test": name, "alpha": alpha,
                             **scores(np.asarray(z["ages"], float), p)})
            print(f"{label:22s} gain-inv={gi}: " + "  ".join(
                f"{r['test']} {r['mae']:.2f}/{r['bias']:+.2f}/{r['r']:.2f}" for r in rows[-len(te):]), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def domain_auc(ref_path, tests, out_csv, gain_variants):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    fam_names = names()
    ref = np.load(ref_path)
    rows = []
    for gi in gain_variants:
        for label, fams in [(f, [f]) for f in FAMILIES] + [("all", FAMILIES)]:
            xr = design(ref, fam_names, fams, gi)
            for name, path in tests:
                xt = design(np.load(path), fam_names, fams, gi)
                x = np.concatenate([xr, xt])
                y = np.r_[np.zeros(len(xr)), np.ones(len(xt))]
                clf = make_pipeline(StandardScaler(), PCA(n_components=min(50, x.shape[1]), random_state=0),
                                    LogisticRegression(C=1.0, max_iter=2000))
                p = cross_val_predict(clf, x, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                      method="predict_proba")[:, 1]
                rows.append({"features": label, "gain_invariant": gi, "test": name,
                             "auc": float(roc_auc_score(y, p))})
            print(f"{label:18s} gain-inv={gi}: " + "  ".join(
                f"{r['test']} {r['auc']:.3f}" for r in rows[-len(tests):]), flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["sample", "extract", "model", "domain"])
    ap.add_argument("--src", help="sample: cohort")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--cohort", help="extract: npy cohort")
    ap.add_argument("--stems", default=None, help="extract: stem list (default: every recording)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--train", help="model: training npz")
    ap.add_argument("--ref", help="domain: in-domain reference npz")
    ap.add_argument("--test", action="append", default=[], metavar="NAME=NPZ")
    ap.add_argument("--gain", nargs="+", default=["no", "yes"], help="model: gain-invariant variants to run")
    args = ap.parse_args()
    if args.cmd == "sample":
        stems = sample_train(args.src, args.n)
        with open(args.out, "w") as fh:
            fh.write("\n".join(stems) + "\n")
        print(f"{len(stems)} stems -> {args.out}")
    elif args.cmd == "extract":
        if args.stems:
            stems = [s.strip() for s in open(args.stems) if s.strip()]
        else:
            stems = sorted(json.load(open(os.path.join(args.cohort, "age_metadata.json")))["recordings"])
        extract(args.cohort, stems, args.out, args.jobs)
    elif args.cmd == "domain":
        domain_auc(args.ref, [t.split("=", 1) for t in args.test], args.out, [g == "yes" for g in args.gain])
    else:
        model(args.train, [t.split("=", 1) for t in args.test], args.out,
              [g == "yes" for g in args.gain])


if __name__ == "__main__":
    main()
