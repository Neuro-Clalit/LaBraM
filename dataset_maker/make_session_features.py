#!/usr/bin/env python
"""Build session-level classical EEG feature tokens for a cohort (model.session_features).

Reads per-recording handcrafted features from the caches written by
``labram.features.tuab`` (branch ``feat/handcrafted-eeg-features``): per channel
``rec/<family>`` [N, 19, F] (median over the clean 10 s windows of minutes 1-6,
CAR, 1-45 Hz) and the filter-bank covariances ``fbcov`` [N, 6, 19, 19]. Writes,
for the fit cohort and every ``--apply-to`` dataset, ``<dir>/<name>`` with two
token groups:

* ``channels`` [N, 19, F]: the families concatenated per channel (F = 283 for
  the default families), one token per scalp channel;
* ``riemann`` [N, 6, 171]: per band, the covariance projected onto the 18-dim
  complement of the all-ones vector (CAR leaves rank 18), shrunk by 1e-3,
  mapped to the tangent space at the Riemannian mean of the *training*
  recordings (Barachant 2012; Sabbagh 2020), upper triangle with off-diagonal
  entries x sqrt(2).

Every column is standardized with the fit cohort's training-split mean / std
(missing values -> 0 after standardizing, then clipped to +-clip), so nothing
from val / test enters the transform. The first cache that has a recording
wins (TUAB and the TUEG healthy cohort share 992 recordings).

    python -m dataset_maker.make_session_features \\
        --fit-cohort /data/.../cohorts/normal_both_v1 \\
        --cache .../tuab/v3.0.1/edf/features_handcrafted/tuab_features_v1.npz \\
        --cache .../tuab/v3.0.1/edf/features_handcrafted/tuab_spectrogram_v1.npz \\
        --cache .../tueg/v2.0.2/features_handcrafted/tueg_healthy_cohort_v1.npz \\
        --apply-to /data/.../cohorts/normal_tueg_v1
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.linalg import null_space

FAMILIES = ("time", "spectral", "psd_bins", "dwt_db4_L6", "wpd_db4_L5", "cwt_cmor1.5-1.0",
            "mspe", "stft", "stft_dyn", "cepstral")
CHANNELS_1020 = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8",
                 "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"]
SHRINK = 1e-3
DEFAULT_NAME = "session_features_v1.npz"


# ------------------------------------------------- Riemannian geometry (NumPy)

def _eig_fn(c, fn):
    w, v = np.linalg.eigh(c)
    return (v * fn(w)[..., None, :]) @ np.swapaxes(v, -1, -2)


def mean_riemann(covs, tol=1e-8, max_iter=50):
    """Affine-invariant (Karcher) mean of SPD matrices [N, n, n]."""
    ref = covs.mean(axis=0)
    for _ in range(max_iter):
        s, si = _eig_fn(ref, np.sqrt), _eig_fn(ref, lambda w: 1.0 / np.sqrt(w))
        t = _eig_fn(si @ covs @ si, np.log).mean(axis=0)
        ref = s @ _eig_fn(t, np.exp) @ s
        if np.linalg.norm(t) < tol:
            break
    return ref


def tangent_space(covs, ref):
    si = _eig_fn(ref, lambda w: 1.0 / np.sqrt(w))
    logm = _eig_fn(si @ covs @ si, np.log)
    iu = np.triu_indices(ref.shape[-1])
    return logm[:, iu[0], iu[1]] * np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))


def project_covs(fbcov):
    """[N, B, C, C] -> [N, B, C-1, C-1]: drop the CAR null direction, shrink."""
    c = fbcov.shape[-1]
    u = null_space(np.ones((1, c)))
    cov = u.T @ fbcov.astype(np.float64) @ u
    tr = np.trace(cov, axis1=-2, axis2=-1)[..., None, None] / cov.shape[-1]
    return (1 - SHRINK) * cov + SHRINK * tr * np.eye(cov.shape[-1])


# ------------------------------------------------------------------ gathering

class Caches:
    def __init__(self, paths, families):
        self.families = families
        self.sources = []
        for p in paths:
            z = np.load(p, allow_pickle=False)
            stems = [str(s) for s in z["meta/stem"]]
            keys = set(z.keys())
            self.sources.append((p, z, {s: i for i, s in enumerate(stems)}, keys))

    def gather(self, stems):
        """``channels`` [N, 19, F] and ``fbcov`` [N, 6, 19, 19] for ``stems``."""
        fam = {f: [None] * len(stems) for f in self.families}
        cov = [None] * len(stems)
        for p, z, rows, keys in self.sources:
            for f in [f for f in self.families if f"rec/{f}" in keys] + (["fbcov"] if "fbcov" in keys else []):
                arr = z[f"rec/{f}" if f != "fbcov" else "fbcov"]
                target = cov if f == "fbcov" else fam[f]
                for i, s in enumerate(stems):
                    if target[i] is None and s in rows:
                        target[i] = arr[rows[s]]
        missing = {f: [s for s, v in zip(stems, fam[f]) if v is None] for f in self.families}
        missing["fbcov"] = [s for s, v in zip(stems, cov) if v is None]
        bad = {k: v for k, v in missing.items() if v}
        if bad:
            k = next(iter(bad))
            raise ValueError(f"{len(bad[k])} recordings have no '{k}' in any cache "
                             f"(e.g. {bad[k][:3]})")
        self.dims = {f: int(fam[f][0].shape[-1]) for f in self.families}
        channels = np.concatenate([np.stack(fam[f]) for f in self.families], axis=-1)
        return channels.astype(np.float64), np.stack(cov)


def cohort_stems(root):
    manifest = json.load(open(os.path.join(root, "manifest.json")))
    return [r["stem"] for r in manifest["recordings"]], [r.get("source_split") for r in manifest["recordings"]]


# ------------------------------------------------------------------ transform

def fit(channels, fbcov):
    covs = project_covs(fbcov)
    refs = np.stack([mean_riemann(covs[:, b]) for b in range(covs.shape[1])])
    riem = _riemann(fbcov, refs)
    stats = {}
    for name, x in (("channels", channels), ("riemann", riem)):
        x = np.where(np.isfinite(x), x, np.nan)
        mean, std = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
        stats[name] = (np.nan_to_num(mean), np.where(np.isfinite(std) & (std > 1e-8), std, 1.0))
    return {"refs": refs, "stats": stats}


def _riemann(fbcov, refs):
    covs = project_covs(fbcov)
    return np.stack([tangent_space(covs[:, b], refs[b]) for b in range(covs.shape[1])], axis=1)


def transform(t, channels, fbcov, clip):
    out = {}
    for name, x in (("channels", channels), ("riemann", _riemann(fbcov, t["refs"]))):
        mean, std = t["stats"][name]
        z = (np.where(np.isfinite(x), x, np.nan) - mean) / std
        out[name] = np.clip(np.nan_to_num(z, nan=0.0), -clip, clip).astype(np.float32)
    return out


def write(path, stems, tokens, meta):
    np.savez(path, stems=np.array(stems), groups=np.array(["channels", "riemann"]),
             **{f"tokens/{k}": v for k, v in tokens.items()},
             meta=np.array(json.dumps(meta)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fit-cohort", required=True, help="cohort dir; its train split fits the transform")
    p.add_argument("--cache", action="append", required=True, help="feature cache .npz (first wins)")
    p.add_argument("--apply-to", action="append", default=[], help="more dataset dirs to write")
    p.add_argument("--families", nargs="+", default=list(FAMILIES))
    p.add_argument("--name", default=DEFAULT_NAME)
    p.add_argument("--clip", type=float, default=6.0)
    args = p.parse_args(argv)

    caches = Caches(args.cache, args.families)
    stems, splits = cohort_stems(args.fit_cohort)
    channels, fbcov = caches.gather(stems)
    train = np.array([s == "train" for s in splits])
    t = fit(channels[train], fbcov[train])
    meta = {"families": args.families, "channels": CHANNELS_1020, "fit_cohort": args.fit_cohort,
            "fit_split": "train", "n_fit": int(train.sum()), "caches": args.cache,
            "clip": args.clip, "riemann_shrink": SHRINK,
            "per_channel_dims": caches.dims}
    for root in [args.fit_cohort] + args.apply_to:
        if root == args.fit_cohort:
            s, (c, f) = stems, (channels, fbcov)
        else:
            s, _ = cohort_stems(root)
            c, f = caches.gather(s)
        tokens = transform(t, c, f, args.clip)
        out = os.path.join(root, args.name)
        write(out, s, tokens, meta)
        print(f"{out}: {len(s)} recordings, " + ", ".join(
            f"{k} {list(v.shape[1:])}" for k, v in tokens.items()))


if __name__ == "__main__":
    sys.exit(main())
