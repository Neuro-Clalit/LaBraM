#!/usr/bin/env python
"""Signal-transformed copies of an npy cohort, for domain-shift tests of frozen models.

Every transform is label-free (it never looks at ages). Applied to a whole
recording ``[T, 23]`` (µV, 200 Hz, missing channels stay zero):

* ``scale:K`` -- multiply by K (a cohort-level gain);
* ``bandlimit:LO,HI`` -- zero-phase Butterworth high-pass LO Hz (order 2) and
  low-pass HI Hz (order 4), imitating a recording chain like CHBP's 0.5-50 Hz;
* ``peramp:TARGET`` -- per recording, scale so its robust amplitude (median
  over 10 s windows of the channel-median std, after the average reference)
  equals TARGET µV;
* ``gsf:TARGET`` -- per recording, remove the global scale factor of the
  qEEG norms (Hernández et al. 1994): scale so the mean log10 PSD over 1-19 Hz
  and the 19 channels equals TARGET;
* ``psdmatch:REF.npz`` -- cohort-level spectral equalization: multiply every
  channel's spectrum by ``sqrt(ref_psd / src_psd)`` (medians over recordings and
  channels, from ``psd`` below), clipped to [1/10, 10], zero-phase (FFT).

Transforms chain with ``+`` (``scale:1.7+bandlimit:0.5,50``). ``--stems`` picks
recordings (one per line); ``--sample N`` draws N recordings per reference
corpus (temple / harvard) from the validation split's normal reports, weighted
to the ages of ``--match-ages`` (2-year bins, seed 0). The output is a
test-only cohort (manifest, age_metadata, age_split).

    python scripts/make_transformed_cohort.py psd --src COHORT [--stems F] --out psd_src.npz
    python scripts/make_transformed_cohort.py make --src COHORT --out OUT --transform peramp:11
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_maker.make_TUAB_npy import npy_bytes  # noqa: E402

FS = 200
WIN = 10 * FS
EEG19 = [i for i in range(23) if i not in (16, 17, 21, 22)]
FLOAT16_MAX = 65504.0


def present(x):
    return [i for i in EEG19 if np.abs(x[:, i]).max() > 0]


def robust_std(x, cols):
    xs = x[:, cols] - x[:, cols].mean(axis=1, keepdims=True)
    n = xs.shape[0] // WIN
    w = xs[:n * WIN].reshape(n, WIN, len(cols))
    return float(np.median(np.median(w.std(axis=1), axis=1)))


def car_psd(x, cols, trim_s=60.0, minutes=5.0):
    start = int(trim_s * FS) if x.shape[0] > int(trim_s * FS) + 60 * FS else 0
    seg = x[start:start + int(minutes * 60 * FS), cols]
    seg = seg - seg.mean(axis=1, keepdims=True)
    f, p = welch(seg.T, fs=FS, nperseg=2 * FS, noverlap=FS, average="median")
    return f, p                                              # [ch, f]


def apply(x, transform, ref=None):
    x = x.astype(np.float64)
    cols = present(x)
    for step in transform.split("+"):
        name, _, arg = step.partition(":")
        if name == "scale":
            x[:, cols] *= float(arg)
        elif name == "bandlimit":
            lo, hi = (float(v) for v in arg.split(","))
            sos = np.vstack([butter(2, lo, "highpass", fs=FS, output="sos"),
                             butter(4, hi, "lowpass", fs=FS, output="sos")])
            x[:, cols] = sosfiltfilt(sos, x[:, cols], axis=0)
        elif name == "peramp":
            x[:, cols] *= float(arg) / robust_std(x, cols)
        elif name == "gsf":
            f, p = car_psd(x, cols)
            band = (f >= 1) & (f <= 19)
            gsf = np.log10(p[:, band]).mean()
            x[:, cols] *= 10 ** ((float(arg) - gsf) / 2)
        elif name == "psdmatch":
            fr, gain = ref
            spec = np.fft.rfft(x[:, cols], axis=0)
            fx = np.fft.rfftfreq(x.shape[0], 1 / FS)
            spec *= np.interp(fx, fr, gain)[:, None]
            x[:, cols] = np.fft.irfft(spec, n=x.shape[0], axis=0)
        else:
            raise ValueError(f"unknown transform {step!r}")
    return x


def cohort_rows(src, stems=None):
    man = json.load(open(os.path.join(src, "manifest.json")))
    rows = man["recordings"]
    if stems is not None:
        keep = set(stems)
        rows = [r for r in rows if r["stem"] in keep]
    return man, rows


def sample_reference(src, match_ages, n, seed=0):
    """N val normal-report recordings per corpus, age-matched (2-year bins) to ``match_ages``."""
    man = {r["stem"]: r for r in json.load(open(os.path.join(src, "manifest.json")))["recordings"]}
    meta = json.load(open(os.path.join(src, "age_metadata.json")))["recordings"]
    val = {f.split("/")[-1].rsplit("_", 1)[0]
           for f in json.load(open(os.path.join(src, "age_split.json")))["files"]["val"]}
    rng = np.random.default_rng(seed)
    bins = np.arange(18, 72, 2)
    h = np.histogram(match_ages, bins)[0] / len(match_ages)
    out = []
    for corpus in ("temple", "harvard"):
        cand = [s for s in sorted(val) if meta.get(s, {}).get("label") == "normal"
                and meta[s].get("age") is not None and bins[0] <= meta[s]["age"] < bins[-1]
                and (man[s].get("corpus") == "heedb") == (corpus == "harvard")]
        ages = np.array([meta[s]["age"] for s in cand], float)
        b = np.digitize(ages, bins) - 1
        cnt = np.bincount(b, minlength=len(h))
        w = h[b] / np.maximum(cnt[b], 1)
        out += list(rng.choice(cand, size=min(n, len(cand)), replace=False, p=w / w.sum()))
    return out


def _psd_one(path):
    x = np.load(path, mmap_mode="r").astype(np.float64)
    f, p = car_psd(x, present(x))
    return f, np.median(p, axis=0)


def _make_one(args):
    src, out, row, transform, ref = args
    x = np.load(os.path.join(src, "recordings", row["stem"] + ".npy"))
    y = np.clip(apply(x, transform, ref), -FLOAT16_MAX, FLOAT16_MAX).astype(np.float16)
    data = npy_bytes(np.ascontiguousarray(y))
    with open(os.path.join(out, "recordings", row["stem"] + ".npy"), "wb") as fh:
        fh.write(data)
    return row["stem"], hashlib.sha256(data).hexdigest(), len(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["psd", "make", "sample"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stems", default=None, help="file with one stem per line")
    ap.add_argument("--transform", default="")
    ap.add_argument("--ref-psd", default=None, help="psdmatch: npz of the reference PSD")
    ap.add_argument("--src-psd", default=None, help="psdmatch: npz of the source PSD")
    ap.add_argument("--match-ages", default=None, help="sample: age_metadata.json to match")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    stems = [s.strip() for s in open(args.stems) if s.strip()] if args.stems else None

    if args.cmd == "sample":
        ages = np.array([r["age"] for r in json.load(open(args.match_ages))["recordings"].values()], float)
        chosen = sample_reference(args.src, ages, args.n)
        with open(args.out, "w") as fh:
            fh.write("\n".join(chosen) + "\n")
        print(f"{len(chosen)} stems -> {args.out}")
        return
    man, rows = cohort_rows(args.src, stems)
    paths = [os.path.join(args.src, "recordings", r["stem"] + ".npy") for r in rows]
    if args.cmd == "psd":
        with cf.ProcessPoolExecutor(args.jobs) as pool:
            res = list(pool.map(_psd_one, paths, chunksize=4))
        f = res[0][0]
        np.savez(args.out, f=f, psd=np.median(np.stack([p for _, p in res]), axis=0), n=len(res))
        print(f"median PSD of {len(res)} recordings -> {args.out}")
        return

    ref = None
    if "psdmatch" in args.transform:
        r, s = np.load(args.ref_psd), np.load(args.src_psd)
        ref = (r["f"], np.clip(np.sqrt(r["psd"] / s["psd"]), 0.1, 10.0))
    os.makedirs(os.path.join(args.out, "recordings"), exist_ok=True)
    with cf.ProcessPoolExecutor(args.jobs) as pool:
        done = {stem: (h, n) for stem, h, n in pool.map(
            _make_one, [(args.src, args.out, r, args.transform, ref) for r in rows], chunksize=2)}
    for r in rows:
        r["sha256"], r["bytes"] = done[r["stem"]]
        r["source_split"] = "test"
    man.update(recordings=rows, n_recordings=len(rows),
               n_windows=int(sum(r["n_windows"] for r in rows)),
               provenance=man.get("provenance", "") + f"; scripts/make_transformed_cohort.py {args.transform}")
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(man, fh)
    meta = json.load(open(os.path.join(args.src, "age_metadata.json")))
    keep = {r["stem"] for r in rows}
    meta["recordings"] = {k: v for k, v in meta["recordings"].items() if k in keep}
    with open(os.path.join(args.out, "age_metadata.json"), "w") as fh:
        json.dump(meta, fh)
    files = [f"test/{r['stem']}_{k}.pkl" for r in rows for k in range(r["n_windows"])]
    with open(os.path.join(args.out, "age_split.json"), "w") as fh:
        json.dump({"version": 1, "seed": None, "counts": {"train": 0, "val": 0, "test": len(rows)},
                   "files": {"train": [], "val": [], "test": files}}, fh)
    print(f"{len(rows)} recordings ({args.transform}) -> {args.out}")


if __name__ == "__main__":
    main()
