#!/usr/bin/env python
"""Attention-based window trimming and recording rejection for bag (MIL) brain-age models.

Reads the per-window parquets written by ``scripts/eval_age_by_cohort.py
--save-windows`` (``recording, slot, minute, alpha, window_pred, age``; one per
run and split) and compares, per run and cohort group, recording predictions

* ``attention`` -- the model's own pooling, sum_i alpha_i p_i (all scored windows);
* ``mean`` -- the plain mean of the window predictions;
* ``trim20_rec`` -- each recording's 20 % lowest-attention windows dropped,
  the rest re-weighted by their renormalised attention;
* ``trim20_glob`` -- windows whose relative weight r_i = alpha_i n (1 = uniform)
  is below the 20th percentile of r over the whole split dropped (a recording
  keeps at least its top window); label-free;
* ``trim20_dev`` -- each recording's 20 % windows whose prediction lies farthest
  from the recording's median window prediction dropped, the rest re-weighted
  (the label-free trim for window-level models, whose uniform alpha makes the
  attention trims a no-op: for them it is a 20 % trimmed mean);

and recording **rejection** at 20 %: the 20 % least confident recordings are
set aside and the MAE of the kept 80 % is reported, for attention-based
confidences

* ``spread`` -- the attention-weighted std of the window predictions,
  sqrt(sum_i alpha_i (p_i - p)^2) (high = unsure);
* ``neff`` -- the effective number of attended windows 1 / sum_i alpha_i^2
  as a share of n (low = the attention found few usable windows);
* ``entropy`` -- the normalised attention entropy H(alpha) / log n (low = peaked);

with the oracle (reject the 20 % largest errors) and random rejection (the
all-recording MAE) as bounds. Rows are written to ``--out``.

    python scripts/attention_trimming.py --windows /data/runs/heedb/external_eval/windows \\
        --out docs/data/external_eval/attention_trimming.csv
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

FRAC = 0.2


def pooled(w, keep=None):
    """Attention-pooled prediction per recording over the kept windows (renormalised)."""
    d = w if keep is None else w[keep]
    num = (d["alpha"] * d["window_pred"]).groupby(d["recording"]).sum()
    den = d["alpha"].groupby(d["recording"]).sum()
    return num / den


def recording_frame(w):
    g = w.groupby("recording")
    n = g.size()
    a = w["alpha"] / g["alpha"].transform("sum")
    p = (a * w["window_pred"]).groupby(w["recording"]).sum()
    dev2 = a * (w["window_pred"] - w["recording"].map(p)) ** 2
    ent = -(a * np.log(a.clip(lower=1e-12))).groupby(w["recording"]).sum()
    return pd.DataFrame({
        "n": n, "age": g["age"].first(), "attention": p, "mean": g["window_pred"].mean(),
        "spread": np.sqrt(dev2.groupby(w["recording"]).sum()),
        "neff": 1.0 / (a ** 2).groupby(w["recording"]).sum() / n,
        "entropy": ent / np.log(n.clip(lower=2)),
    })


def trims(w):
    w = w.copy()
    n = w.groupby("recording")["alpha"].transform("size")
    w["r"] = w["alpha"] * n
    rank = w.groupby("recording")["alpha"].rank(method="first", ascending=True)
    drop_rec = rank <= np.floor(FRAC * n)
    top = w.groupby("recording")["alpha"].transform("max") == w["alpha"]
    thr = w["r"].quantile(FRAC)
    dev = (w["window_pred"] - w.groupby("recording")["window_pred"].transform("median")).abs()
    dev_rank = dev.groupby(w["recording"]).rank(method="first", ascending=False)
    drop_dev = dev_rank <= np.floor(FRAC * n)
    return {"trim20_rec": pooled(w, ~drop_rec), "trim20_dev": pooled(w, ~drop_dev),
            "trim20_glob": pooled(w, (w["r"] >= thr) | top), "glob_threshold_r": thr,
            "share_dropped_glob": float(((w["r"] < thr) & ~top).mean())}


def mae_kept(err, score, low_is_unsure=True):
    """MAE of the recordings kept after setting aside the FRAC least confident."""
    k = int(round(len(err) * (1 - FRAC)))
    order = np.argsort(score if low_is_unsure else -score, kind="stable")   # least confident first
    keep = order[len(err) - k:]
    return float(np.abs(err[keep]).mean())


def groups_for(cohort, rec, meta):
    out = [("all", np.ones(len(rec), bool))]
    if cohort.startswith("lemon"):
        out += [("young", (rec["age"] < 40).to_numpy()), ("old", (rec["age"] >= 55).to_numpy())]
    elif meta is not None:
        m = meta.reindex(rec.index)
        for src in ("TUAB", "HEEDB"):
            out.append((src, (m["source"] == src).to_numpy()))
        for src in ("TUAB", "HEEDB"):
            for c in ("normal", "abnormal", "unlabelled"):
                mk = ((m["source"] == src) & (m["cohort"] == c)).to_numpy()
                out.append((f"{src} {c}", mk))
    return out


def analyse(path, cohort, meta):
    run = os.path.basename(path).rsplit("_", 1)[0]
    w = pd.read_parquet(path)
    rec = recording_frame(w)
    t = trims(w)
    rec["trim20_rec"] = t["trim20_rec"]
    rec["trim20_glob"] = t["trim20_glob"]
    rec["trim20_dev"] = t["trim20_dev"]
    rows = []
    for gname, mask in groups_for(cohort, rec, meta):
        if mask.sum() < 10:
            continue
        r = rec[mask]
        age = r["age"].to_numpy()
        row = {"cohort": cohort, "run": run, "group": gname, "n": int(mask.sum()),
               "windows_per_rec": float(r["n"].mean())}
        for m in ("attention", "mean", "trim20_rec", "trim20_glob", "trim20_dev"):
            row[f"mae_{m}"] = float(np.abs(r[m] - age).mean())
        err = (r["attention"] - age).to_numpy()
        row["mae_keep80_spread"] = mae_kept(err, -r["spread"].to_numpy())
        row["mae_keep80_neff"] = mae_kept(err, r["neff"].to_numpy())
        row["mae_keep80_entropy"] = mae_kept(err, r["entropy"].to_numpy())
        row["mae_keep80_oracle"] = mae_kept(err, -np.abs(err))
        terr = (r["trim20_glob"] - age).to_numpy()
        row["mae_trim20glob_keep80_spread"] = mae_kept(terr, -r["spread"].to_numpy())
        derr = (r["trim20_dev"] - age).to_numpy()
        row["mae_trim20dev_keep80_spread"] = mae_kept(derr, -r["spread"].to_numpy())
        for s in ("spread", "neff", "entropy"):
            row[f"rho_{s}_abs_err"] = float(pd.Series(r[s].to_numpy()).corr(
                pd.Series(np.abs(err)), method="spearman"))
        row["glob_threshold_r"] = t["glob_threshold_r"]
        row["share_dropped_glob"] = t["share_dropped_glob"]
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--windows", required=True, help="root with <cohort>/<run>_test.parquet")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    meta = None
    try:
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        import mil_cohort_summary as mcs
        meta = mcs.recording_table()
    except Exception as exc:                       # LEMON-only use
        print(f"no cohort table ({exc}); own-test groups limited to 'all'")
    rows = []
    for d in sorted(glob.glob(os.path.join(args.windows, "*"))):
        cohort = os.path.basename(d)
        for p in sorted(glob.glob(os.path.join(d, "*_test.parquet"))):
            rows += analyse(p, cohort, meta)
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    cols = ["cohort", "run", "group", "n", "windows_per_rec", "mae_attention", "mae_mean", "mae_trim20_rec",
            "mae_trim20_glob", "mae_trim20_dev", "mae_keep80_spread", "mae_keep80_neff", "mae_keep80_entropy", "mae_keep80_oracle",
            "mae_trim20glob_keep80_spread", "rho_spread_abs_err", "rho_neff_abs_err"]
    print(out[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
