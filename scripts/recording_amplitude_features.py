#!/usr/bin/env python
"""Per-recording amplitude statistics and the cohort-relative amplitude feature.

``stats`` computes, for every recording of the given npy cohorts, on the segment
the age models see (first 5 min after a 60 s trim, 19 channels, common average
reference): the robust amplitude ``amp_uv`` (median over 10 s windows of the
channel-median std, µV) and the share of 45-70 Hz power in 1-70 Hz
(``hf_ratio``, Welch, median averaging), plus its acquisition group, which is
label-free metadata:

* Temple (TUH): ``tuh/<montage>/<native sampling rate>``;
* Harvard (HEEDB): ``heedb/<site>/<native sampling rate>``;
* CHBP: ``chbp/<12-bit | 16-bit EDF>`` (64- / 120-channel cap, ``--chbp-encoding``);
* any other cohort: one group, its corpus.

Groups with fewer than ``--min-group`` recordings fall back to their corpus.

``features`` writes a session-feature file (``labram/data/session_features.py``
format, group ``rel_amplitude`` [N, 1, 2]) into each cohort: r = the log10 of a
recording's amplitude relative to the median of its acquisition group, and r²
(amplitude is U-shaped in age: highest in young adults and again in the oldest,
and the head that receives these features is linear), each standardized with
the training split of ``--fit-cohort``. With per-window
amplitude normalization (``labram_plus.amplitude_norm=window``) the waveform
loses absolute gain; this feature (``model.session_features=head``) gives back
how loud a recording is *relative to its peers on the same equipment*, which
carries age, without the gain of the equipment itself, which does not.

    python scripts/recording_amplitude_features.py stats --cohort COHORT ... --out stats.parquet
    python scripts/recording_amplitude_features.py features --stats stats.parquet \\
        --fit-cohort TRAIN_COHORT --cohort COHORT ... --name rel_amplitude_v1.npz
"""
import argparse
import concurrent.futures as cf
import json
import os

import numpy as np
import pandas as pd
from scipy.signal import welch

FS = 200
WIN = 10 * FS
EEG19 = [i for i in range(23) if i not in (16, 17, 21, 22)]


def recording_stats(path, trim_s=60.0, minutes=5.0):
    x = np.load(path, mmap_mode="r")
    start = int(trim_s * FS) if x.shape[0] > int(trim_s * FS) + 60 * FS else 0
    seg = np.asarray(x[start:start + int(minutes * 60 * FS)], dtype=np.float64)
    cols = [i for i in EEG19 if np.abs(seg[:, i]).max() > 0]
    seg = seg[:, cols] - seg[:, cols].mean(axis=1, keepdims=True)
    n = seg.shape[0] // WIN
    if n == 0 or not cols:
        return np.nan, np.nan
    w = seg[:n * WIN].reshape(n, WIN, len(cols))
    amp = float(np.median(np.median(w.std(axis=1), axis=1)))
    f, p = welch(seg.T, fs=FS, nperseg=2 * FS, noverlap=FS, average="median")
    psd = np.median(p, axis=0)
    hf = float(psd[(f >= 45) & (f < 70)].sum() / max(psd[(f >= 1) & (f < 70)].sum(), 1e-12))
    return amp, hf


def _one(args):
    stem, path = args
    try:
        return stem, *recording_stats(path)
    except Exception:                       # unreadable recording: leave it out
        return stem, np.nan, np.nan


def cohort_rows(cohort, chbp_encoding=None):
    man = json.load(open(os.path.join(cohort, "manifest.json")))["recordings"]
    enc = (pd.read_csv(chbp_encoding).set_index("stem")["enc"].to_dict()
           if chbp_encoding and os.path.exists(chbp_encoding) else {})
    rows = []
    for r in man:
        corpus = r.get("corpus") or os.path.basename(cohort.rstrip("/"))
        if corpus == "tuh":
            group = f"tuh/{r.get('montage')}/{r.get('sfreq_native')}"
        elif corpus == "heedb":
            group = f"heedb/{r.get('site')}/{r.get('sfreq_native')}"
        elif corpus == "chbp":
            group = f"chbp/{enc.get(r['stem'], 'unknown')}"
        else:
            group = corpus
        rows.append({"stem": r["stem"], "cohort": cohort, "corpus": corpus, "group": group,
                     "split": r.get("source_split"),
                     "path": os.path.join(cohort, "recordings", r["stem"] + ".npy")})
    return rows


def stats(cohorts, out, jobs, chbp_encoding, min_group):
    rows = [row for c in cohorts for row in cohort_rows(c, chbp_encoding)]
    print(f"{len(rows)} recordings", flush=True)
    res = {}
    with cf.ProcessPoolExecutor(jobs) as pool:
        for i, (stem, amp, hf) in enumerate(pool.map(_one, [(r["stem"], r["path"]) for r in rows],
                                                     chunksize=64), 1):
            res[stem] = (amp, hf)
            if i % 10000 == 0:
                print(f"  {i}/{len(rows)}", flush=True)
    df = pd.DataFrame(rows).drop(columns="path")
    df["amp_uv"] = [res[s][0] for s in df.stem]
    df["hf_ratio"] = [res[s][1] for s in df.stem]
    size = df.groupby("group").stem.transform("size")
    df.loc[size < min_group, "group"] = df.loc[size < min_group, "corpus"]
    df.to_parquet(out)
    print(df.groupby("group").agg(n=("stem", "size"), amp=("amp_uv", "median"),
                                  hf=("hf_ratio", "median")).round(4).to_string())


def features(stats_path, fit_cohort, cohorts, name):
    df = pd.read_parquet(stats_path)
    df["log_amp"] = np.log10(df["amp_uv"].clip(lower=1e-3))
    med = df.groupby("group")["log_amp"].transform("median")
    df["rel"] = (df["log_amp"] - med).fillna(0.0)
    df["rel2"] = df["rel"] ** 2                 # amplitude is U-shaped in age: the head is linear
    fit = df[(df.cohort == fit_cohort) & (df.split == "train")]
    mu = fit[["rel", "rel2"]].mean()
    sd = fit[["rel", "rel2"]].std()
    for cohort in cohorts:
        d = df[df.cohort == cohort]
        z = np.clip(((d[["rel", "rel2"]] - mu) / sd).to_numpy(np.float32), -5, 5)
        path = os.path.join(cohort, name)
        np.savez(path, stems=d["stem"].to_numpy().astype(str), groups=np.array(["rel_amplitude"]),
                 **{"tokens/rel_amplitude": z.reshape(-1, 1, 2)},
                 info=json.dumps({"feature": "[r, r^2], r = log10(amp / median amp of acquisition group), "
                                             "each standardized with the training split",
                                  "fit_cohort": fit_cohort, "mean": mu.tolist(), "std": sd.tolist()}))
        print(f"{cohort}: {len(d)} recordings -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["stats", "features"])
    ap.add_argument("--cohort", action="append", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--fit-cohort", default=None)
    ap.add_argument("--name", default="rel_amplitude_v1.npz")
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--min-group", type=int, default=30)
    ap.add_argument("--chbp-encoding", default="/data/runs/heedb/domain_shift/chbp_edf_encoding.csv")
    args = ap.parse_args()
    if args.cmd == "stats":
        stats(args.cohort, args.out, args.jobs, args.chbp_encoding, args.min_group)
    else:
        features(args.stats, args.fit_cohort, args.cohort, args.name)


if __name__ == "__main__":
    main()
