#!/usr/bin/env python
"""Brain-age error split by pathology: the 18 multi-task targets of
``multitask_targets_v2.npz`` (status_abnormal + 17 diagnosis groups, as the
E2P pathology classifier predicts them), on every held-out recording (val +
test) of TUH and HEEDB.

For each model, dataset (TUH, HEEDB, both) and target, the positives are
compared with the negatives (known label, value 0):

* ``n_pos`` / ``n_neg``, mean age of each;
* raw ``mae`` / ``bias`` (mean predicted - true age) of each;
* ``mae_neg_std``: the negatives' MAE re-weighted to the positives' age-decade
  mix, ``sum_d w_d MAE_neg,d`` with ``w_d`` the share of positives in decade d.
  Error grows with age and diagnoses cluster in older patients, so the raw
  negative MAE is not a fair baseline;
* ``d_mae = mae_pos - mae_neg_std`` with a 95 % bootstrap interval (positives
  resampled, negatives resampled within decade);
* ``bagc``: the brain-age gap after the quadratic age correction of Gemein et
  al. 2024, fitted per dataset on the validation recordings with a normal
  report; ``d_bagc`` = positives - negatives, Welch p-value.

Rows with fewer than 10 positives are dropped (aggregates only, DUA).

    python scripts/age_by_pathology.py --scores /data/runs/heedb/scores \\
        --models E0 E2 E2L --out docs/data/age_by_pathology.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from labram.eval.age_analysis import apply_bias_correction, fit_bias_correction  # noqa: E402

JOINT = "/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1"
CORPORA = {"tuab": ("TUH", "/data/datasets/EEG-public/TUH_EEG/cohorts/all_both_v3_anymiss"),
           "heedb": ("HEEDB", "/data/datasets/HEEDB/v4.1/cohorts/heedb_routine_v1")}
EDGES = [0, 30, 40, 50, 60, 70, 80, 200]
MIN_POS = 10


def predictions(scores, model):
    """One row per held-out recording: dataset, split, stem, age, pred, status."""
    rows = []
    for key, (dataset, root) in CORPORA.items():
        path = os.path.join(scores, f"{model}_{key}_valtest.json")
        if not os.path.exists(path):
            return None
        meta = json.load(open(os.path.join(root, "age_metadata.json")))["recordings"]
        for split, r in json.load(open(path))[model]["splits"].items():
            for stem, p in r["predictions"].items():
                rows.append({"dataset": dataset, "split": split, "stem": stem, "pred": float(p),
                             "age": float(meta[stem]["age"]), "label": meta[stem].get("label")})
    return pd.DataFrame(rows)


def add_corrected_gap(df):
    """Quadratic gap correction per dataset, fitted on its val recordings with a normal report."""
    df = df.copy()
    df["bagc"] = np.nan
    for ds, g in df.groupby("dataset"):
        ref = g[(g.split == "val") & (g.label == "normal")]
        params = fit_bias_correction(ref.pred, ref.age)
        corr = apply_bias_correction(g.pred, params, "quadratic_gap", age=g.age)
        df.loc[g.index, "bagc"] = corr - g.age
    return df


def _std_mae(err_neg, dec_neg, w):
    out = 0.0
    for d, wd in w.items():
        e = err_neg[dec_neg == d]
        if not len(e):
            return np.nan
        out += wd * e.mean()
    return out


def compare(g, col, n_boot=500, seed=0):
    """Statistics of positives (col == 1) vs negatives (col == 0) in ``g``."""
    from scipy.stats import ttest_ind
    pos, neg = g[g[col] == 1], g[g[col] == 0]
    if len(pos) < MIN_POS or len(neg) < MIN_POS:
        return None
    ep, en = (pos.pred - pos.age).abs().to_numpy(), (neg.pred - neg.age).abs().to_numpy()
    dp, dn = pos.decade.to_numpy(), neg.decade.to_numpy()
    w = pd.Series(dp).value_counts(normalize=True).to_dict()
    neg_std = _std_mae(en, dn, w)
    rng = np.random.default_rng(seed)
    boots = []
    by_dec = {d: en[dn == d] for d in w}
    for _ in range(n_boot):
        bp = ep[rng.integers(0, len(ep), len(ep))].mean()
        bn = sum(wd * by_dec[d][rng.integers(0, len(by_dec[d]), len(by_dec[d]))].mean()
                 for d, wd in w.items() if len(by_dec[d]))
        boots.append(bp - bn)
    return {"n_pos": len(pos), "n_neg": len(neg),
            "age_pos": pos.age.mean(), "age_neg": neg.age.mean(),
            "mae_pos": ep.mean(), "mae_neg": en.mean(), "mae_neg_std": neg_std,
            "d_mae": ep.mean() - neg_std, "d_mae_lo": np.percentile(boots, 2.5),
            "d_mae_hi": np.percentile(boots, 97.5),
            "bias_pos": (pos.pred - pos.age).mean(), "bias_neg": (neg.pred - neg.age).mean(),
            "bagc_pos": pos.bagc.mean(), "bagc_neg": neg.bagc.mean(),
            "d_bagc": pos.bagc.mean() - neg.bagc.mean(),
            "p_bagc": ttest_ind(pos.bagc, neg.bagc, equal_var=False).pvalue}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scores", required=True)
    ap.add_argument("--models", nargs="+", default=["E0", "E2", "E2L"])
    ap.add_argument("--targets", default=os.path.join(JOINT, "multitask_targets_v2.npz"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    z = np.load(args.targets, allow_pickle=True)
    names = [str(n).replace("pathology/", "") for n in z["names"]]
    tgt = pd.DataFrame(z["targets"], columns=names, index=pd.Index(z["stems"].astype(str), name="stem"))
    rows = []
    for model in args.models:
        df = predictions(args.scores, model)
        if df is None:
            print(f"{model}: no scores, skipped")
            continue
        df = add_corrected_gap(df).join(tgt, on="stem")
        df["decade"] = pd.cut(df.age, EDGES, right=False, labels=False)
        known = df[names].notna()
        # Healthy reference: normal report and no diagnosis flagged (among known labels).
        df["healthy"] = np.where(df.status_abnormal.isna(), np.nan,
                                 ((df.status_abnormal == 0) & (df[names[1:]].fillna(0).sum(1) == 0)).astype(float))
        for ds_name, sub in (("TUH", df[df.dataset == "TUH"]), ("HEEDB", df[df.dataset == "HEEDB"]), ("both", df)):
            for col in ["healthy"] + names:
                r = compare(sub, col)
                if r is not None:
                    rows.append({"model": model, "dataset": ds_name, "target": col, **r})
        print(f"{model}: {len(df)} held-out recordings, labels known per target: "
              f"{int(known.sum().min())}-{int(known.sum().max())}")
    out = pd.DataFrame(rows)
    out = out.round({c: 3 for c in out.columns if c not in ("model", "dataset", "target", "p_bagc")})
    out["p_bagc"] = out.p_bagc.map(lambda v: f"{v:.2e}")
    out.to_csv(args.out, index=False)
    print(out[out.dataset == "both"][["model", "target", "n_pos", "age_pos", "mae_pos", "mae_neg_std", "d_mae",
                                       "d_mae_lo", "d_mae_hi", "d_bagc", "p_bagc"]].to_string(index=False))


if __name__ == "__main__":
    main()
