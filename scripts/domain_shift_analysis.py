#!/usr/bin/env python
"""Signal features of an external cohort vs the training corpora (domain-shift analysis).

For every recording the features are computed on exactly what the brain-age
models see: the first ``--minutes`` after the ``--trim`` seconds, the 19 10-20
channels (A1/A2/T1/T2 dropped), common average reference, µV at 200 Hz (the
npy cohort format). Features per recording:

Statistics are taken per 10 s window and then the median over windows (robust
to the settling / movement artifacts of clinical recordings); the PSD is a
Welch estimate with median averaging.

* amplitude: median channel std, 99th percentile of |x| (dynamic range used),
  share of windows with a channel std above 100 µV (artifact windows), share of
  samples above 100 µV, median kurtosis;
* spectrum (Welch, 2 s Hann segments; channel-median PSD): absolute power of
  0.1-1 (10 s segments), 1-4, 4-8, 8-13, 13-30, 30-45 and 45-70 Hz, relative
  power of the 1-45 Hz bands, 95 % spectral edge (0.5-70 Hz);
* aperiodic: OLS line of log10 PSD on log10 f over 2-40 Hz without 6-14 Hz
  (``exponent`` = -slope, ``offset``);
* alpha: occipital (O1/O2) peak frequency in 6-14 Hz above the aperiodic line
  and its height (log10 ratio); posterior / frontal alpha power ratio;
* Hjorth mobility and complexity (channel medians); mean |correlation|
  between channels after the average reference.

Reference sets come from the training cohort (``--ref-cohort``), recordings of
its validation split with a normal report, aged ``--age-range``, grouped by
``corpus`` (heedb -> harvard, else temple). Writes ``features.parquet`` and a
summary with medians weighted to the external cohort's age distribution
(2-year bins) and the AUC of external vs each reference.

    python scripts/domain_shift_analysis.py \\
        --cohort chbp=/data/datasets/EEG-public/CHBP/cohorts/chbp_v1 \\
        --ref-cohort /data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1 \\
        --out-dir /data/runs/heedb/domain_shift --jobs 12
"""
import argparse
import concurrent.futures as cf
import json
import os

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis

FS = 200
WIN = 10 * FS
EEG19 = [i for i in range(23) if i not in (16, 17, 21, 22)]
CH = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
      'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2']
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
POSTERIOR = [CH.index(c) for c in ("O1", "O2", "P3", "P4", "PZ")]
FRONTAL = [CH.index(c) for c in ("FP1", "FP2", "F3", "F4", "FZ")]


def load_segment(path, trim_s, minutes):
    x = np.load(path, mmap_mode="r")
    start = int(trim_s * FS)
    stop = start + int(minutes * 60 * FS)
    if x.shape[0] < start + 60 * FS:            # short: take what there is after the trim
        start = 0
    return np.asarray(x[start:stop], dtype=np.float64)


def features(path, trim_s=60.0, minutes=5.0):
    """Robust per-recording features: statistics of each 10 s window, then the
    median over windows (clinical recordings carry large settling / movement
    artifacts that would otherwise dominate); Welch PSD with median averaging."""
    x = load_segment(path, trim_s, minutes)                 # [T, 23]
    cols = [i for i in EEG19 if np.abs(x[:, i]).max() > 0]
    xs = x[:, cols]
    xs = xs - xs.mean(axis=1, keepdims=True)                 # common average reference
    n_win = xs.shape[0] // WIN
    w = xs[:n_win * WIN].reshape(n_win, WIN, len(cols))
    w = w - w.mean(axis=1, keepdims=True)
    out = {"n_channels": len(cols), "seconds": xs.shape[0] / FS}
    std = w.std(axis=1)                                      # [win, ch]
    out["std_uv"] = float(np.median(np.median(std, axis=1)))
    out["p99_abs_uv"] = float(np.median(np.percentile(np.abs(w), 99, axis=(1, 2))))
    out["artifact_window_share"] = float((std.max(axis=1) > 100).mean())
    out["frac_gt_100uv"] = float((np.abs(xs) > 100).mean())
    out["kurtosis"] = float(np.median(kurtosis(w, axis=1)))
    f, p = welch(xs.T, fs=FS, nperseg=2 * FS, noverlap=FS, average="median")  # [ch, f]
    psd = np.median(p, axis=0)
    total = psd[(f >= 1) & (f < 45)].sum()
    for name, (lo, hi) in BANDS.items():
        band = psd[(f >= lo) & (f < hi)]
        out[f"abs_{name}"] = float(band.mean())
        out[f"rel_{name}"] = float(band.sum() / total)
    out["abs_45_70"] = float(psd[(f >= 45) & (f < 70)].mean())
    f10, p10 = welch(xs.T, fs=FS, nperseg=10 * FS, average="median")
    out["abs_0p1_1"] = float(np.median(p10, axis=0)[(f10 >= 0.1) & (f10 < 1)].mean())
    sel = (f >= 0.5) & (f <= 70)
    cum = np.cumsum(psd[sel]) / psd[sel].sum()
    out["edge95_hz"] = float(f[sel][np.searchsorted(cum, 0.95)])
    fit = (f >= 2) & (f <= 40) & ~((f >= 6) & (f <= 14))
    slope, offset = np.polyfit(np.log10(f[fit]), np.log10(psd[fit]), 1)
    out["aperiodic_exponent"], out["aperiodic_offset"] = float(-slope), float(offset)
    occ = [cols.index(c) for c in (CH.index("O1"), CH.index("O2")) if c in cols]
    po = p[occ].mean(axis=0) if occ else psd
    a = (f >= 6) & (f <= 14)
    resid = np.log10(po[a]) - (offset + slope * np.log10(f[a]))
    out["alpha_peak_hz"] = float(f[a][resid.argmax()])
    out["alpha_peak_height"] = float(resid.max())
    al = (f >= 8) & (f < 13)
    post = [cols.index(c) for c in POSTERIOR if c in cols]
    front = [cols.index(c) for c in FRONTAL if c in cols]
    out["alpha_post_front_ratio"] = float(p[post][:, al].mean() / p[front][:, al].mean())
    d1, d2 = np.diff(w, axis=1), np.diff(w, n=2, axis=1)
    mob = np.sqrt(d1.var(axis=1) / w.var(axis=1))
    comp = np.sqrt(d2.var(axis=1) / d1.var(axis=1)) / mob
    out["hjorth_mobility"] = float(np.median(mob))
    out["hjorth_complexity"] = float(np.median(comp))
    iu = np.triu_indices(len(cols), 1)
    out["mean_abs_corr"] = float(np.median([np.abs(np.corrcoef(wi.T)[iu]).mean() for wi in w]))
    return out


def edf_encoding(raw_dir, channel="O1"):
    """Per recording: EDF header calibration of ``channel`` (number of signals, physical
    and digital range, resolution) and the encoding (12-bit / 16-bit), from
    ``<raw_dir>/sub-*/eeg/*.edf`` (stem ``sub-<code>_chbp-rs``)."""
    import glob
    rows = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "sub-*", "eeg", "*_eeg.edf"))):
        with open(path, "rb") as fh:
            ns = int(fh.read(256)[252:256])
            h = fh.read(ns * 256)

        def field(offset, width):
            return [h[offset + i * width: offset + (i + 1) * width].decode("latin-1").strip() for i in range(ns)]
        labels = field(0, 16)
        off = ns * (16 + 80 + 8)
        pmin, pmax = field(off, 8), field(off + ns * 8, 8)
        dmin, dmax = field(off + ns * 16, 8), field(off + ns * 24, 8)
        i = next(k for k, lab in enumerate(labels) if lab.replace(" ", "").upper().startswith(channel))
        rows.append({"stem": os.path.basename(path).split("_")[0] + "_chbp-rs", "n_sig": ns,
                     "pmin": float(pmin[i]), "pmax": float(pmax[i]), "dmin": int(dmin[i]), "dmax": int(dmax[i]),
                     "lsb": (float(pmax[i]) - float(pmin[i])) / (int(dmax[i]) - int(dmin[i])),
                     "enc": "16-bit" if int(dmax[i]) > 4096 else "12-bit"})
    return pd.DataFrame(rows).set_index("stem")


def _one(args):
    stem, path, trim_s, minutes = args
    try:
        return stem, features(path, trim_s, minutes)
    except Exception as exc:                       # record, never abort
        return stem, {"error": f"{type(exc).__name__}: {exc}"}


def ref_recordings(cohort, age_range):
    man = {r["stem"]: r for r in json.load(open(os.path.join(cohort, "manifest.json")))["recordings"]}
    meta = json.load(open(os.path.join(cohort, "age_metadata.json")))["recordings"]
    split = json.load(open(os.path.join(cohort, "age_split.json")))["files"]["val"]
    val = {f.split("/")[-1].rsplit("_", 1)[0] for f in split}
    rows = []
    for s in sorted(val):
        m = meta.get(s, {})
        if m.get("label") != "normal" or m.get("age") is None or not age_range[0] <= m["age"] <= age_range[1]:
            continue
        rows.append({"stem": s, "set": "harvard" if man[s].get("corpus") == "heedb" else "temple",
                     "age": float(m["age"]), "path": os.path.join(cohort, "recordings", s + ".npy")})
    return rows


def age_weights(ages, target_ages, lo=18, width=2):
    bins = np.arange(lo, max(target_ages.max(), ages.max()) + width + 1, width)
    h = np.histogram(target_ages, bins)[0] / len(target_ages)
    b = np.clip(np.digitize(ages, bins) - 1, 0, len(h) - 1)
    cnt = np.bincount(b, minlength=len(h))
    return h[b] / np.maximum(cnt[b], 1)


def weighted_quantile(x, w, q):
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return float(np.interp(q, cw, x[o]))


def auc(a, b, wa=None, wb=None):
    """P(a > b) (+ half ties), optionally weighted."""
    wa = np.ones(len(a)) if wa is None else wa
    wb = np.ones(len(b)) if wb is None else wb
    order = np.argsort(b)
    bs, cw = b[order], np.cumsum(wb[order])
    lo = np.searchsorted(bs, a, side="left")
    hi = np.searchsorted(bs, a, side="right")
    below = np.where(lo > 0, cw[lo - 1], 0.0)
    tie = np.where(hi > 0, cw[hi - 1], 0.0) - below
    return float((wa * (below + 0.5 * tie)).sum() / (wa.sum() * wb.sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cohort", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--ref-cohort", required=True)
    ap.add_argument("--age-range", nargs=2, type=float, default=[18, 68])
    ap.add_argument("--trim", type=float, default=60.0)
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--edf-root", default=None,
                    help="CHBP BIDS root: also write chbp_edf_encoding.csv (EDF calibration per recording)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.edf_root:
        enc = edf_encoding(args.edf_root)
        enc.to_csv(os.path.join(args.out_dir, "chbp_edf_encoding.csv"))
        print("EDF encodings:", enc["enc"].value_counts().to_dict())
    recs = ref_recordings(args.ref_cohort, args.age_range)
    externals = []
    for item in args.cohort:
        name, d = item.split("=", 1)
        meta = json.load(open(os.path.join(d, "age_metadata.json")))["recordings"]
        for s, m in meta.items():
            recs.append({"stem": s, "set": name, "age": float(m["age"]),
                         "path": os.path.join(d, "recordings", s + ".npy")})
        externals.append(name)
    print(f"{len(recs)} recordings: {pd.Series([r['set'] for r in recs]).value_counts().to_dict()}", flush=True)
    feats = {}
    with cf.ProcessPoolExecutor(args.jobs) as pool:
        for i, (stem, f) in enumerate(pool.map(_one, [(r["stem"], r["path"], args.trim, args.minutes)
                                                       for r in recs], chunksize=8), 1):
            feats[stem] = f
            if i % 500 == 0:
                print(f"  {i}/{len(recs)}", flush=True)
    df = pd.DataFrame(recs).set_index("stem").join(pd.DataFrame.from_dict(feats, orient="index"))
    if "error" in df:
        print("errors:", df["error"].notna().sum())
        df = df[df["error"].isna()].drop(columns="error")
    df.drop(columns="path").to_parquet(os.path.join(args.out_dir, "features.parquet"))

    df = df.drop(columns="path")
    feat_cols = [c for c in df.columns if c not in ("set", "age", "n_channels", "seconds")]
    rows = []
    for ext in externals:
        e = df[df.set == ext]
        for ref in ("temple", "harvard"):
            r = df[df.set == ref]
            w = age_weights(r.age.to_numpy(), e.age.to_numpy())
            for c in feat_cols:
                rows.append({"external": ext, "reference": ref, "feature": c,
                             "ext_median": e[c].median(), "ref_median_age_matched": weighted_quantile(
                                 r[c].to_numpy(), w, 0.5),
                             "ratio": e[c].median() / weighted_quantile(r[c].to_numpy(), w, 0.5),
                             "auc_ext_gt_ref": auc(e[c].to_numpy(), r[c].to_numpy(), None, w),
                             "rho_age_ext": e[[c, "age"]].corr("spearman").iloc[0, 1],
                             "rho_age_ref": r[[c, "age"]].corr("spearman").iloc[0, 1]})
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out_dir, "feature_shift_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
