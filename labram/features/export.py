# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Training-ready export of the handcrafted-feature caches.
# ---------------------------------------------------------
"""Export the recording-level feature caches as flat, training-ready arrays.

The ``tuab.py`` caches are research caches: nested ``rec/<family>`` arrays of
shape ``[n, 19, F]`` plus ablation variants and per-window quality arrays. This
module writes one directory per dataset that any training loop can read
directly (see :mod:`labram.features.dataset`)::

    flat_v1/
      features.npy      float32 [n_recordings, n_features]  (memory-mappable)
      fbcov.npy         float32 [n_recordings, 6, 19, 19]   filter-bank covariances
      meta.parquet      one row per recording: stem, subject, age, sex, label, splits
      columns.parquet   one row per feature column: name, family, group, feature, channel
      info.json         shapes, families, split definitions, sha256 of every file

Columns are ordered family by family, channel-major inside a family
(``<family>:<feature>@<channel>``), the same order as
:func:`labram.features.modeling.build_feature_set`. Values are the raw
recording-level medians (no scaling, no imputation; the caches contain no NaN).

Usage::

    python -m labram.features.export --dataset tuab --npy_root <tuab>/processed_npy
    python -m labram.features.export --dataset tueg --npz <tueg>/features_handcrafted/tueg_healthy_cohort_v1.npz \\
        --npy_root <tuab>/processed_npy
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os

import numpy as np
import pandas as pd

from labram.features import tuab

# The 12 canonical per-channel families of the notebook (§8), in a fixed order;
# the 13th group, filter-bank Riemann, is ``fbcov.npy``.
CANONICAL_FAMILIES = ("time", "spectral", "psd_bins", "dwt_db4_L6", "wpd_db4_L5", "cwt_cmor1.5-1.0",
                      "scat_J7_Q8", "stft", "multitaper", "stft_dyn", "cepstral", "mspe")
FAMILY_GROUP = {"time": "time", "spectral": "spectral", "psd_bins": "psd_bins", "dwt_db4_L6": "dwt",
                "wpd_db4_L5": "wpd", "cwt_cmor1.5-1.0": "cwt", "scat_J7_Q8": "scattering", "stft": "stft",
                "multitaper": "multitaper", "stft_dyn": "dynamics", "cepstral": "cepstral", "mspe": "mspe"}
FORMAT = "flat_v1"


def flat_matrix(D: dict, names: dict, families, channels=tuab.CHANNELS_1020):
    """``[n, F]`` float32 matrix and its column table (raw values, canonical order)."""
    n = D["meta/stem"].shape[0]
    blocks, rows = [], []
    for fam in families:
        a = np.asarray(D[f"rec/{fam}"], dtype=np.float32)
        assert a.shape[1:] == (len(channels), len(names[fam])), (fam, a.shape)
        blocks.append(a.reshape(n, -1))
        rows += [{"name": f"{fam}:{f}@{c}", "family": fam, "group": FAMILY_GROUP.get(fam, fam),
                  "feature": f, "channel": c} for c in channels for f in names[fam]]
    cols = pd.DataFrame(rows)
    cols.insert(0, "column", np.arange(len(cols)))
    return np.ascontiguousarray(np.concatenate(blocks, axis=1)), cols


def tuab_split_columns(meta: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """The notebook's splits as columns: ``split`` (dev / test) and the CV folds.

    ``fold_abn``: subject-grouped, label-stratified 5-fold over the development
    pool (all recordings); ``fold_age``: subject-grouped 5-fold over the normal
    development recordings with a known age. ``-1`` = not in that pool.
    """
    from labram.features import modeling as M

    sp = M.make_splits(meta, seed=seed)
    out = pd.DataFrame(index=meta.index)
    out["split"] = np.where(meta.source_split.isin(["train", "val"]), "dev",
                            np.where(meta.source_split.eq("test"), "test", "other"))
    for task in ("abn", "age"):
        fold = np.full(len(meta), -1, dtype=np.int8)
        for k, (_, va) in enumerate(sp.folds[task]):
            fold[sp.dev[task][va]] = k
        out[f"fold_{task}"] = fold
    return out


def tueg_cohort_split(meta_t: pd.DataFrame, meta_tuab: pd.DataFrame, frac: float = 0.2,
                      seed: int = 0) -> np.ndarray:
    """§13 split of the TUEG healthy subset: ``train`` / ``test`` / ``excluded``.

    ``test`` = every recording of a random ``frac`` of the subjects that appear
    only in TUEG (seed 0, as in the notebook). ``excluded`` = recordings of
    subjects in the official TUAB eval set, which are never trained on.
    """
    dup = meta_t.stem.isin(set(meta_tuab.stem)).to_numpy()
    tueg_only = np.array(sorted(set(meta_t.subject[~dup]) - set(meta_tuab.subject)))
    rng = np.random.default_rng(seed)
    test_subj = set(rng.choice(tueg_only, int(round(frac * len(tueg_only))), replace=False))
    tuab_test_subj = set(meta_tuab.subject[meta_tuab.source_split.eq("test") & meta_tuab.label.eq("normal")
                                           & np.isfinite(meta_tuab.age.astype(float))])
    split = np.where(meta_t.subject.isin(test_subj), "test", "train").astype(object)
    split[meta_t.subject.isin(tuab_test_subj).to_numpy()] = "excluded"
    return split


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def write_flat(out_dir: str, X: np.ndarray, fbcov: np.ndarray, meta: pd.DataFrame, cols: pd.DataFrame,
               info: dict) -> dict:
    """Write one ``flat_v1`` directory; returns the ``info.json`` content."""
    os.makedirs(out_dir, exist_ok=True)
    assert len(X) == len(meta) == len(fbcov) and X.shape[1] == len(cols)
    np.save(os.path.join(out_dir, "features.npy"), X.astype(np.float32))
    np.save(os.path.join(out_dir, "fbcov.npy"), np.asarray(fbcov, dtype=np.float32))
    meta.reset_index(drop=True).to_parquet(os.path.join(out_dir, "meta.parquet"), index=False)
    cols.to_parquet(os.path.join(out_dir, "columns.parquet"), index=False)
    fams = cols.drop_duplicates("family")
    info = {"format": FORMAT,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "n_recordings": int(len(X)), "n_features": int(X.shape[1]),
            "families": {f: int((cols.family == f).sum()) for f in fams.family},
            "channels": list(tuab.CHANNELS_1020),
            "fbcov": {"shape": list(fbcov.shape[1:]), "bands_hz": {k: list(v) for k, v in tuab.fx.FILTERBANK.items()}},
            **info}
    info["files"] = {f: {"bytes": os.path.getsize(os.path.join(out_dir, f)), "sha256": _sha256(os.path.join(out_dir, f))}
                     for f in ("features.npy", "fbcov.npy", "meta.parquet", "columns.parquet")}
    with open(os.path.join(out_dir, "info.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    return info


def export_tuab(npy_root: str, out_dir: str = None) -> dict:
    from labram.features import modeling as M

    D, names, meta = M.load_feature_table(npy_root)
    fams = [f for f in CANONICAL_FAMILIES if f"rec/{f}" in D]
    X, cols = flat_matrix(D, names, fams)
    m = meta[["stem", "subject", "age", "sex", "label", "abnormal", "source_split"]].copy()
    m["n_clean_windows"] = np.asarray(D["n_clean"]).astype(np.int16)
    m = pd.concat([m, tuab_split_columns(meta)], axis=1)
    out_dir = out_dir or os.path.join(os.path.dirname(tuab.default_cache_path(npy_root)), FORMAT)
    return write_flat(out_dir, X, D["fbcov"], m, cols, {
        "dataset": "TUAB v3.0.1 (all 2,990 recordings)", "source": [tuab.default_cache_path(npy_root, k)
                                                               for k in ("main", "scattering", "spectrogram")],
        "splits": {"split": "dev = official train (+val), test = official eval",
                   "fold_abn": "pathology CV fold (subject-grouped, label-stratified, seed 0); -1 = test",
                   "fold_age": "age CV fold over normal dev recordings with an age (subject-grouped, seed 0); -1 = not in pool"}})


def export_tueg(npz: str, tuab_npy_root: str, out_dir: str = None) -> dict:
    from labram.features import modeling as M

    z = np.load(npz)
    D = {k: z[k] for k in z.files}
    names = tuab.feature_names()
    names.update(tuab.spectrogram_names())
    fams = [f for f in CANONICAL_FAMILIES if f"rec/{f}" in D]
    X, cols = flat_matrix(D, names, fams)
    meta_t = pd.DataFrame({k[len("meta/"):]: D[k] for k in D if k.startswith("meta/") and D[k].ndim == 1})
    _, _, meta_a = M.load_feature_table(tuab_npy_root, kinds=("main",))
    a = meta_a.set_index("stem")
    m = meta_t[["stem", "subject", "age", "sex"]].copy()
    m["in_tuab"] = m.stem.isin(a.index)
    m["tuab_source_split"] = m.stem.map(a.source_split).fillna("")
    m["tuab_label"] = m.stem.map(a.label).fillna("")
    m["n_clean_windows"] = np.asarray(D["n_clean"]).astype(np.int16)
    m["cohort_split"] = tueg_cohort_split(meta_t, meta_a)
    out_dir = out_dir or os.path.join(os.path.dirname(npz), FORMAT)
    return write_flat(out_dir, X, D["fbcov"], m, cols, {
        "dataset": "TUEG v2.0.2 healthy subset v1 (8,592 recordings, adults 18-89)", "source": [npz],
        "splits": {"cohort_split": "notebook §13: test = all recordings of a random 20 % of TUEG-only subjects "
                                   "(seed 0); excluded = subjects of the TUAB eval set; train = the rest",
                   "in_tuab": "recording is also a TUAB recording (its TUAB features are in the TUAB export; "
                              "the notebook keeps the TUAB copy when pooling)"}})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", choices=("tuab", "tueg"), required=True)
    ap.add_argument("--npy_root", required=True, help="TUAB processed_npy root (its feature caches are used)")
    ap.add_argument("--npz", help="TUEG cohort cache (labram.features.tuab --kind cohort)")
    ap.add_argument("--out", default=None, help="output directory (default: <cache dir>/flat_v1)")
    a = ap.parse_args(argv)
    info = export_tuab(a.npy_root, a.out) if a.dataset == "tuab" else export_tueg(a.npz, a.npy_root, a.out)
    print(json.dumps({k: info[k] for k in ("dataset", "n_recordings", "n_features", "families")}, indent=1))


if __name__ == "__main__":
    main()
