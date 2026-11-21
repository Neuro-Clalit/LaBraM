#!/usr/bin/env python
"""Do handcrafted features add to a trained brain-age model? A stacking test on one CV fold.

The frozen fold model (e.g. P4 fold 0) keeps its predictions; a ridge regression on the
recordings of its *validation fold* (never trained on) learns to predict age from the
model's prediction plus handcrafted recording-level features, and is scored once on the
held-out test split, paired against the fold model alone on the same recordings.

Features are the lean cohort families of ``labram/features`` (``tuab.extract_recording_cohort``:
time, Welch spectral + 1/f, 1 Hz log-PSD bins, DWT, wavelet packets, complex-Morlet CWT,
multiscale permutation entropy, STFT statistics, band-power dynamics, cepstral coefficients;
19 channels each) and the six filter-bank covariances, from the windows the model scores:
19 channels, common average reference, 1-45 Hz, first minute skipped, median over the clean
10 s windows of the next five minutes.

    # 1) recording lists of the fold (train / val / test), checked against the fold's saved predictions
    python scripts/feature_stack.py stems --fold-dir RUN/cv/fold_0 --cohort COHORT --out-dir OUT
    # 2) features, resumable (one part file per chunk); val and test first, train last
    python scripts/feature_stack.py extract --cohort COHORT --stems OUT/stems_val.txt --out-dir OUT/features/val
    # 3) one memory-mappable matrix per split
    python scripts/feature_stack.py merge --parts OUT/features/val --out OUT/features_val
    # 4) the fold model's predictions on its validation fold and the test split, same windows
    python scripts/eval_cv_folds.py --data-path COHORT --run P4_L1=RUN/cv/fold_0 --splits val test --preds-out OUT/preds.json
    # 5) ridge on the validation fold (strength by patient-grouped CV inside it), scored once on test
    python scripts/feature_stack.py fit --cohort COHORT --preds OUT/preds.json --val OUT/features_val \
        --test OUT/features_test --train-stems OUT/stems_train.txt --out OUT/results
    # 6) with the training recordings' features: an age model on the features alone (fitted on train),
    #    whose predictions the validation-fold ridge then combines with the fold model
    python scripts/feature_stack.py feature-model --cohort COHORT --train OUT/features_train \\
        --val OUT/features_val --test OUT/features_test --out OUT/results
    python scripts/feature_stack.py fit ... --feature-model OUT/results/feature_model_predictions.parquet

Per-recording outputs stay outside git (Harvard and TDBRAIN are under data use agreements);
only aggregates are reported.
"""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAMILIES = ["time", "spectral", "psd_bins", "dwt_db4_L6", "wpd_db4_L5", "cwt_cmor1.5-1.0", "mspe",
            "stft", "stft_dyn", "cepstral"]


# ------------------------------------------------------------------ stems
def fold_stems(fold_dir, cohort):
    """(train, val, test) recording stems of a ``test_set=held_out`` CV fold: val = the fold's
    groups, train = the rest of the pool, test = the cohort's own test split. Groups are the
    leading ``_`` token of the stem (``cross_validation.group_id_for``, ``split_by=subject``)."""
    split = json.load(open(os.path.join(fold_dir, "cv_split.json")))
    if split.get("split_by") != "subject" or split.get("test_set") != "held_out":
        raise SystemExit("expects a subject-grouped CV split with the held-out test set")
    k = int(os.path.basename(os.path.normpath(fold_dir)).rsplit("_", 1)[1])
    co = pd.read_parquet(os.path.join(cohort, "cohort.parquet"))
    grp = co.stem.str.split("_").str[0]
    val_groups = set(split["folds"][k]["groups"])
    pool = co.split.isin(["train", "val"])
    if not grp[pool].isin(set().union(*(f["groups"] for f in split["folds"]))).all():
        raise SystemExit("a pool recording is in no fold")
    return (sorted(co.stem[pool & ~grp.isin(val_groups)]), sorted(co.stem[pool & grp.isin(val_groups)]),
            sorted(co.stem[co.split == "test"]))


def cmd_stems(args):
    train, val, test = fold_stems(args.fold_dir, args.cohort)
    os.makedirs(args.out_dir, exist_ok=True)
    for name, stems in (("train", train), ("val", val), ("test", test)):
        saved = glob.glob(os.path.join(args.fold_dir, "analysis", f"{name}_bag_weights_epoch*.parquet"))
        if saved:                                   # the fold's own scored recordings
            got = set(pd.read_parquet(sorted(saved)[-1], columns=["recording"]).recording)
            if got != set(stems):
                raise SystemExit(f"{name}: {len(set(stems) ^ got)} recording(s) differ from the fold's saved predictions")
        with open(os.path.join(args.out_dir, f"stems_{name}.txt"), "w") as fh:
            fh.write("\n".join(stems) + "\n")
        print(f"{name}: {len(stems)} recordings" + (" (matches the fold's saved predictions)" if saved else ""))


# ------------------------------------------------------------------ extraction
STATE_FAMILIES = [f"state/{s}/{part}" for s in ("high", "low") for part in ("periodic", "aperiodic", "logpsd")] + [
    "state/all/periodic", "state/all/aperiodic"]
QUALITY = ["max_abs", "max_ptp", "min_std", "median_std"]


def _extract_main(path):
    """``(features by family, per-recording extras)`` of one recording: the cohort families."""
    from labram.features import tuab
    out = tuab.extract_recording_cohort(path)
    q = out["quality"]
    return ({f: out[f"rec/{f}"] for f in FAMILIES},
            {"fbcov": out["fbcov"], "n_clean": np.int16(out["n_clean"]),
             "quality": np.median(q[6:] if len(q) > 6 else q, axis=0).astype(np.float32)})


def _extract_state(path):
    """Periodic / aperiodic spectra of the high-alpha and low-alpha windows (``tuab.extract_recording_state_spectra``)."""
    from labram.features import tuab
    out = tuab.extract_recording_state_spectra(path)
    return ({f: out[f] for f in STATE_FAMILIES},
            {"alpha_index": out["alpha_index"], "n_state": np.int16(out["n_state"])})


STUDY_CORE = ["len10", "len30", "len60", "len30_seg4", "state_high", "state_low",
              "agg_mean", "agg_q10", "agg_q90", "agg_sd", "agg_attn"]
STUDY_CONN = ["conn_all", "conn_len10", "conn_len30", "conn_high", "conn_low"]
STUDY_COV = ["cov_all", "cov_high", "cov_low"]
STUDY_FAMILIES = STUDY_CORE + STUDY_CONN + STUDY_COV
_ATTENTION = None


def _attention(stem):
    """The fold model's attention weights of a recording's scored windows (in window order), from the
    window files named in ``$FEATURE_STACK_ATTENTION`` (comma-separated ``eval_cv_folds.py --save-windows`` parquets)."""
    global _ATTENTION
    if _ATTENTION is None:
        _ATTENTION = {}
        for path in filter(None, os.environ.get("FEATURE_STACK_ATTENTION", "").split(",")):
            w = pd.read_parquet(path, columns=["recording", "slot", "alpha"]).sort_values(["recording", "slot"])
            _ATTENTION.update({r: g.to_numpy() for r, g in w.groupby("recording", sort=False)["alpha"]})
    return _ATTENTION.get(stem)


def _extract_study(path):
    """The method study (``tuab.extract_recording_study``): window length, eye state, window summary, connectivity."""
    from labram.features import tuab
    out = tuab.extract_recording_study(path, _attention(os.path.basename(path)[:-4]))
    return {f: out[f] for f in STUDY_FAMILIES}, {"n_clean": np.int16(out["n_clean"])}


STUDY2_FAMILIES = ["conn_calm", "conn_attn", "cov_calm", "cov_rand", "cov_tn", "cov_attn"]


def _extract_study2(path):
    """Controls for the eye-state result (``tuab.extract_recording_study2``): calm, random, power-normalised, attention-weighted."""
    from labram.features import tuab
    out = tuab.extract_recording_study2(path, _attention(os.path.basename(path)[:-4]))
    return {f: out[f] for f in STUDY2_FAMILIES}, {"calm_high_overlap": np.float32(out["calm_high_overlap"])}


# kind -> (worker, families); "main" = handcrafted cohort families + filter-bank covariances,
# "state" = the feature space of Azami et al. 2025 on the recording's own two alpha states,
# "study" = the same cheap features under different analysis choices
KINDS = {"main": (_extract_main, FAMILIES), "state": (_extract_state, STATE_FAMILIES),
         "study": (_extract_study, STUDY_FAMILIES), "study2": (_extract_study2, STUDY2_FAMILIES)}


def _extract(job):
    """One recording, or its error text (it then holds NaN): one bad file must not stop a 100k run."""
    kind, path = job
    try:
        return (*KINDS[kind][0](path), None)
    except Exception as exc:                        # noqa: BLE001
        return None, None, f"{type(exc).__name__}: {exc}"


def cmd_extract(args):
    families = KINDS[args.kind][1]
    stems = [s.strip() for s in open(args.stems) if s.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    chunks = [stems[i:i + args.chunk] for i in range(0, len(stems), args.chunk)]
    todo = [(i, c) for i, c in enumerate(chunks)
            if not os.path.exists(os.path.join(args.out_dir, f"part_{i:05d}.npz"))]
    print(f"{len(stems)} recordings in {len(chunks)} chunks, {len(todo)} to do, {args.jobs} jobs", flush=True)
    t0, done = time.time(), 0
    with ProcessPoolExecutor(args.jobs) as pool:
        for i, chunk in todo:
            jobs = [(args.kind, os.path.join(args.cohort, "recordings", s + ".npy")) for s in chunk]
            res = list(pool.map(_extract, jobs, chunksize=4))
            ok = next((r for r in res if r[0] is not None), None)
            if ok is None:
                raise SystemExit(f"chunk {i}: every recording failed, e.g. {res[0][2]}")

            def stack(which, key):                  # a failed recording holds NaN (0 for a count)
                like = np.asarray(ok[which][key])
                fill = np.zeros_like(like) if like.dtype.kind in "iu" else np.full(like.shape, np.nan, like.dtype)
                return np.stack([np.asarray(r[which][key]) if r[0] is not None else fill for r in res])
            arrays = {f"f_{f}": stack(0, f) for f in families}
            arrays.update({k: stack(1, k) for k in ok[1]})
            errors = [f"{s}\t{r[2]}" for s, r in zip(chunk, res) if r[2]]
            tmp = os.path.join(args.out_dir, f".part_{i:05d}.tmp.npz")
            np.savez(tmp, stems=np.array(chunk), **arrays)
            os.replace(tmp, os.path.join(args.out_dir, f"part_{i:05d}.npz"))     # atomic: a part is complete or absent
            if errors:
                with open(os.path.join(args.out_dir, "errors.txt"), "a") as fh:
                    fh.write("\n".join(errors) + "\n")
            done += len(chunk)
            rate = done / (time.time() - t0)
            left = sum(len(c) for j, c in todo if j > i)
            print(f"  chunk {i + 1}/{len(chunks)}: {rate:.1f} rec/s, {left / rate / 60:.0f} min left"
                  + (f", {len(errors)} failed" if errors else ""), flush=True)
    print(f"done: {args.out_dir}", flush=True)


def feature_names(kind="main"):
    """Column names per family (channel-major: 19 channels x the family's features)."""
    from labram.features import tuab
    if kind in ("study", "study2"):
        n = tuab.study_names()
        return {f: ([f"{ch}:{c}" for ch in tuab.CHANNELS_1020 for c in n[f]] if f in STUDY_CORE else list(n[f]))
                for f in KINDS[kind][1]}
    n = ({**tuab.feature_names(), **tuab.spectrogram_names()} if kind == "main" else tuab.state_spectra_names())
    return {f: [f"{ch}:{c}" for ch in tuab.CHANNELS_1020 for c in n[f]] for f in KINDS[kind][1]}


def cmd_merge(args):
    parts = sorted(glob.glob(os.path.join(args.parts, "part_*.npz")))
    z = [np.load(p) for p in parts]
    kind = ("state" if "f_state/all/periodic" in z[0].files else "study" if "f_len10" in z[0].files
            else "study2" if "f_cov_calm" in z[0].files else "main")
    families, names = KINDS[kind][1], feature_names(kind)
    stems = np.concatenate([p["stems"] for p in z])
    x = np.concatenate([np.concatenate([p[f"f_{f}"].reshape(len(p["stems"]), -1) for f in families], axis=1)
                        for p in z]).astype(np.float32)
    cols = pd.DataFrame({"family": [f for f in families for _ in names[f]],
                         "column": [c for f in families for c in names[f]]})
    if len(cols) != x.shape[1]:
        raise SystemExit(f"{x.shape[1]} feature columns but {len(cols)} names")
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "features.npy"), x)
    meta = {"stem": stems}
    if kind == "main":
        np.save(os.path.join(args.out, "fbcov.npy"), np.concatenate([p["fbcov"] for p in z]).astype(np.float32))
        meta["n_clean"] = np.concatenate([p["n_clean"] for p in z])
        meta.update({f"q_{k}": np.concatenate([p["quality"][:, j] for p in z]) for j, k in enumerate(QUALITY)})
    elif kind == "study":
        meta["n_clean"] = np.concatenate([p["n_clean"] for p in z])
    elif kind == "study2":
        meta["calm_high_overlap"] = np.concatenate([p["calm_high_overlap"] for p in z])
    else:
        meta["n_state"] = np.concatenate([p["n_state"] for p in z])
        meta.update({f"alpha_index_{k}": np.concatenate([p["alpha_index"][:, j] for p in z])
                     for j, k in enumerate(("high", "low", "all"))})
    pd.DataFrame(meta).to_parquet(os.path.join(args.out, "meta.parquet"))
    cols.to_parquet(os.path.join(args.out, "columns.parquet"))
    print(f"{len(stems)} recordings x {x.shape[1]} columns -> {args.out} "
          f"({int(np.isnan(x).all(axis=1).sum())} failed recordings)")


# ------------------------------------------------------------------ stacking
AGE_BINS = [(18, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 200)]
ALPHAS = np.logspace(1, 7, 19)
PRIMARY = "all + riemann + sex"                    # fixed before the test split is scored
# The feature space of Azami et al. 2025 (Dortmund Vital Study): periodic and aperiodic spectra of
# eyes-closed and eyes-open rest; here of the recording's own high-alpha and low-alpha windows.
PAPER_A = "paper A: periodic + aperiodic, high + low alpha"
TIME_NAMES = ["log_var", "log_line_length", "skew", "kurtosis", "hjorth_mobility", "hjorth_complexity", "zero_cross_rate"]
EVERYTHING = "all + riemann + sex + paper A"
# The feature space of Arambula Villarino 2025 (PsyArXiv, Dortmund Vital Study): absolute and relative
# power in five bands, spectral entropy and the three Hjorth parameters, per channel.
PAPER_B = ({f"log_abs_{b}" for b in ("delta", "theta", "alpha", "beta", "gamma")}
           | {f"rel_{b}" for b in ("delta", "theta", "alpha", "beta", "gamma")}
           | {"spectral_entropy", "log_var", "hjorth_mobility", "hjorth_complexity"})


def load_split(feat_dir, preds, cohort):
    """Aligned arrays of one split: stems, features, covariances, model prediction, age, cohort rows."""
    meta = pd.read_parquet(os.path.join(feat_dir, "meta.parquet"))
    keep = meta.stem.isin(preds).to_numpy()
    stems = meta.stem.to_numpy()[keep]
    x = np.load(os.path.join(feat_dir, "features.npy"), mmap_mode="r")[keep]
    cov = np.load(os.path.join(feat_dir, "fbcov.npy"), mmap_mode="r")[keep]
    rows = cohort.loc[stems]
    return {"stems": stems, "x": np.asarray(x, np.float64), "cov": np.asarray(cov, np.float64),
            "pred": np.array([preds[s][0] for s in stems]), "age": np.array([preds[s][1] for s in stems]),
            "rows": rows, "n_dropped": int((~keep).sum()),
            "patient": np.where(rows.corpus == "heedb", rows.patient_id.astype(str), rows.subject.astype(str))}


def _rows(feat_dir, stems):
    """Row selector of a merged feature folder for ``stems``: a slice when they are its rows in order."""
    meta = pd.read_parquet(os.path.join(feat_dir, "meta.parquet"), columns=["stem"])
    if len(meta) == len(stems) and (meta.stem.to_numpy() == np.asarray(stems)).all():
        return slice(None)
    return pd.Series(np.arange(len(meta)), index=meta.stem.to_numpy()).loc[stems].to_numpy()


def _state_block(feat_dir, stems):
    """State-spectra features of ``stems`` (rows aligned to them)."""
    meta = pd.read_parquet(os.path.join(feat_dir, "meta.parquet"))
    row = pd.Series(np.arange(len(meta)), index=meta.stem.to_numpy())
    return np.asarray(np.load(os.path.join(feat_dir, "features.npy"), mmap_mode="r")[row.loc[stems].to_numpy()],
                      np.float64)


def cross_fitted_age(a, b, age, fit, folds, alphas):
    """Age predicted from one feature block alone, never from a model that saw the recording:
    ridge on the fit rows (strength by its own patient-grouped CV MAE), out-of-fold predictions
    for the fit rows, the all-rows model for the other validation rows and the test split.
    Returns ``(val predictions, test predictions, alpha, out-of-fold MAE)``."""
    af, yf = a[fit], age[fit]
    one = lambda n: np.ones((n, 1))                 # noqa: E731
    oof = np.zeros((len(alphas), len(yf)))
    for k in range(folds.max() + 1):
        tr, te = folds != k, folds == k
        oof[:, te] = ridge_path(one(tr.sum()), af[tr], yf[tr], alphas)(one(te.sum()), af[te])
    mae = np.abs(oof - yf[None]).mean(1)
    best = int(mae.argmin())
    final = ridge_path(one(len(yf)), af, yf, alphas[best:best + 1])
    fv = np.empty(len(a))
    fv[fit] = oof[best]
    if (~fit).any():
        fv[~fit] = final(one(int((~fit).sum())), a[~fit])[0]
    return fv, final(one(len(b)), b)[0], float(alphas[best]), float(mae[best])


class Standardize:
    """z-score with the fit rows' mean / SD, clip to +-6, NaN / inf -> 0 (the fit mean)."""

    def fit(self, x):
        x = np.where(np.isfinite(x), x, np.nan)
        self.mean, self.sd = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
        self.sd[~(self.sd > 1e-12)] = 1.0
        return self

    def __call__(self, x):
        z = (np.where(np.isfinite(x), x, np.nan) - self.mean) / self.sd
        return np.clip(np.nan_to_num(z, nan=0.0), -6.0, 6.0)


def sex_block(rows):
    s = rows.sex.astype(str).str.upper().str[0]
    return np.c_[(s == "F").to_numpy(float), (~s.isin(["F", "M"])).to_numpy(float)]


def ridge_path(z_fit, x_fit, r_fit, alphas):
    """Ridge of the residual ``r`` on ``x`` with the columns of ``z`` (intercept, model prediction)
    left unpenalised: project ``z`` out of both, then solve the whole path from one decomposition
    (eigenvectors of the Gram matrix, or the thin SVD when there are more columns than rows).
    Returns ``predict(z, x) -> [n_alphas, n]``."""
    q, _ = np.linalg.qr(z_fit)
    xr, rr = x_fit - q @ (q.T @ x_fit), r_fit - q @ (q.T @ r_fit)
    n, p = xr.shape
    if p == 0:
        beta = np.zeros((len(alphas), 0))
    elif p > n:
        u, sv, vt = np.linalg.svd(xr, full_matrices=False)
        uty = u.T @ rr
        beta = np.stack([vt.T @ (sv / (sv ** 2 + a) * uty) for a in alphas])
    else:
        w, v = np.linalg.eigh(xr.T @ xr)
        vty = v.T @ (xr.T @ rr)
        beta = np.stack([v @ (vty / (np.maximum(w, 0) + a)) for a in alphas])     # [A, p]
    gamma = np.linalg.lstsq(z_fit, r_fit[:, None] - x_fit @ beta.T, rcond=None)[0]  # [k, A]
    return lambda z, x: (x @ beta.T + z @ gamma).T


def grouped_folds(groups, n_folds=5, seed=0):
    u = np.unique(groups)
    fold_of = dict(zip(u, np.random.default_rng(seed).permutation(len(u)) % n_folds))
    return np.array([fold_of[g] for g in groups])


def paired_bootstrap(d, groups, n_boot=2000, seed=0):
    """95 % interval of mean(d) with groups (patients) resampled."""
    idx = pd.factorize(groups)[0]
    s, n = np.bincount(idx, weights=d), np.bincount(idx).astype(float)
    draw = np.random.default_rng(seed).integers(0, len(s), size=(n_boot, len(s)))
    m = s[draw].sum(1) / n[draw].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def slope(pred, age):
    return float(np.polyfit(age, pred, 1)[0])


PAPER_A_FAMILIES = [f"state/{s}/{part}" for s in ("high", "low") for part in ("periodic", "aperiodic")]


def _design_blocks(feat_dir, stems_keep, std, ts, rstd, cohort, chunk=8192, state=None):
    """Yield (rows, standardized [handcrafted | riemann | sex | paper A]) chunks of a merged feature
    folder; ``state`` = (state-spectra folder, its scaler, its column indices) adds the last block."""
    meta = pd.read_parquet(os.path.join(feat_dir, "meta.parquet"))
    keep = np.flatnonzero(meta.stem.isin(stems_keep).to_numpy()) if stems_keep is not None else np.arange(len(meta))
    x = np.load(os.path.join(feat_dir, "features.npy"), mmap_mode="r")
    cov = np.load(os.path.join(feat_dir, "fbcov.npy"), mmap_mode="r")
    for i in range(0, len(keep), chunk):
        idx = keep[i:i + chunk]
        rows = cohort.loc[meta.stem.to_numpy()[idx]]
        r = rstd(ts.transform(np.nan_to_num(np.asarray(cov[idx], np.float64)).reshape(len(idx), -1)))
        blocks = [std(np.asarray(x[idx], np.float64)), r, sex_block(rows)]
        if state is not None:
            blocks.append(state[1](_state_block(state[0], rows.index.to_numpy())[:, state[2]]))
        yield rows, np.concatenate(blocks, axis=1)


def cmd_feature_model(args):
    """Age from the features alone, fitted on the fold's TRAINING recordings: one Gram matrix of
    [handcrafted | riemann | sex], every feature set a sub-block of it; ridge strength per set by the
    validation fold's MAE. Writes each set's predicted age for the validation and test recordings
    (``fit --feature-model`` then combines them with the fold model on the validation fold)."""
    from labram.features.modeling import FBTangentSpace
    cohort = pd.read_parquet(os.path.join(args.cohort, "cohort.parquet")).set_index("stem")
    cols = pd.read_parquet(os.path.join(args.train, "columns.parquet"))
    meta = pd.read_parquet(os.path.join(args.train, "meta.parquet"))
    x = np.load(os.path.join(args.train, "features.npy"), mmap_mode="r")
    cov = np.load(os.path.join(args.train, "fbcov.npy"), mmap_mode="r")
    rng = np.random.default_rng(0)
    ref = np.sort(rng.choice(len(meta), size=min(args.ref_rows, len(meta)), replace=False))
    std = Standardize().fit(np.asarray(x[ref], np.float64))          # scaling and Riemann means from a train sample
    cref = np.nan_to_num(np.asarray(cov[ref], np.float64)).reshape(len(ref), -1)
    ts = FBTangentSpace().fit(cref)
    rstd = Standardize().fit(ts.transform(cref))
    n_h, n_r = x.shape[1], ts.transform(cref[:2]).shape[1]
    p = n_h + n_r + 2
    sets = {f: np.flatnonzero((cols.family == f).to_numpy()) for f in FAMILIES}
    sets["paper B: band power, entropy, Hjorth"] = np.flatnonzero(cols.column.str.split(":").str[1].isin(PAPER_B).to_numpy())
    sets.update({"riemann": np.arange(n_h, n_h + n_r), "sex": np.arange(p - 2, p), "all handcrafted": np.arange(n_h),
                 PRIMARY: np.arange(p)})
    state = {s: None for s in ("train", "val", "test")}
    if args.train_state:                            # paper A: periodic / aperiodic spectra of the two alpha states
        sidx = np.flatnonzero(pd.read_parquet(os.path.join(args.train_state, "columns.parquet")).family.isin(
            PAPER_A_FAMILIES).to_numpy())
        sstd = Standardize().fit(_state_block(args.train_state, meta.stem.to_numpy()[ref])[:, sidx])
        state = {"train": (args.train_state, sstd, sidx), "val": (args.val_state, sstd, sidx),
                 "test": (args.test_state, sstd, sidx)}
        sets.update({PAPER_A: np.arange(p, p + len(sidx)), EVERYTHING: np.arange(p + len(sidx))})
        p += len(sidx)
    gram, xty, xsum, ysum, n = np.zeros((p, p)), np.zeros(p), np.zeros(p), 0.0, 0
    t0 = time.time()
    for rows, d in _design_blocks(args.train, None, std, ts, rstd, cohort, state=state["train"]):
        y = rows.age.to_numpy(float)
        ok = np.isfinite(y)
        d, y = d[ok], y[ok]
        gram += d.T @ d
        xty += d.T @ y
        xsum += d.sum(0)
        ysum += y.sum()
        n += len(y)
        print(f"  train Gram: {n} recordings [{time.time() - t0:.0f}s]", flush=True)
    xm, ym = xsum / n, ysum / n
    gram -= n * np.outer(xm, xm)                    # centred: the intercept is unpenalised
    xty -= n * xm * ym
    ev = {s: [(rows, d) for rows, d in _design_blocks(dd, None, std, ts, rstd, cohort, state=state[s])]
          for s, dd in (("val", args.val), ("test", args.test))}
    stems = {s: np.concatenate([rows.index.to_numpy() for rows, _ in ev[s]]) for s in ev}
    age = {s: np.concatenate([rows.age.to_numpy(float) for rows, _ in ev[s]]) for s in ev}
    out = {s: {"stem": stems[s], "age": age[s]} for s in ev}
    rows_out = []
    for name, idx in sets.items():
        w, v = np.linalg.eigh(gram[np.ix_(idx, idx)])
        vty = v.T @ xty[idx]
        beta = np.stack([v @ (vty / (np.maximum(w, 0) + a)) for a in ALPHAS])           # [A, p_set]
        pred = {s: np.concatenate([(d[:, idx] - xm[idx]) @ beta.T for _, d in ev[s]]).T + ym for s in ev}
        mae = np.abs(pred["val"] - age["val"][None]).mean(1)
        best = int(mae.argmin())
        for s in ev:
            out[s][name] = pred[s][best]
        e = pred["test"][best] - age["test"]
        rows_out.append({"features": name, "columns": len(idx), "alpha": float(ALPHAS[best]), "n_train": n,
                         "val_mae": float(mae[best]), "test_mae": float(np.abs(e).mean()), "test_bias": float(e.mean()),
                         "test_slope": slope(pred["test"][best], age["test"])})
        print(f"{name:22s} p={len(idx):5d} alpha={ALPHAS[best]:9.0f}  val {mae[best]:.3f}  "
              f"test {rows_out[-1]['test_mae']:.3f}  slope {rows_out[-1]['test_slope']:.3f}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(os.path.join(args.out, "feature_model_summary.csv"), index=False)
    pd.concat([pd.DataFrame(out[s]).assign(split=s) for s in ev]).to_parquet(
        os.path.join(args.out, "feature_model_predictions.parquet"))                    # per recording: stays local


def cmd_fit(args):
    cohort = pd.read_parquet(os.path.join(args.cohort, "cohort.parquet")).set_index("stem")
    preds = json.load(open(args.preds))
    preds = preds[args.model][str(args.fold)] if args.model in preds else preds
    val, test = (load_split(d, preds[s], cohort) for d, s in ((args.val, "val"), (args.test, "test")))
    cols = pd.read_parquet(os.path.join(args.val, "columns.parquet"))
    fam_idx = {f: np.flatnonzero((cols.family == f).to_numpy()) for f in FAMILIES}
    # validation patients also seen in the fold's training set (a patient under two site ids): not fitted on
    if args.train_stems:
        tr = cohort.loc[[s.strip() for s in open(args.train_stems) if s.strip()]]
        seen = set(np.where(tr.corpus == "heedb", tr.patient_id.astype(str), tr.subject.astype(str)))
        fit = ~np.isin(val["patient"], list(seen))
    else:
        fit = np.ones(len(val["stems"]), bool)
    print(f"val {len(fit)} recordings ({val['n_dropped']} without a prediction), fitted on {int(fit.sum())} "
          f"({len(np.unique(val['patient'][fit]))} patients); test {len(test['stems'])} "
          f"({len(np.unique(test['patient']))} patients)", flush=True)

    from labram.features.modeling import FBTangentSpace
    # Blocks are kept once, as float32; a feature set is either a (val, test) pair or a function that
    # builds one when its turn comes, so the wide combined sets are never all in memory together.
    f32 = lambda a: np.asarray(a, np.float32)       # noqa: E731

    def cat(*parts):
        return lambda: tuple(np.concatenate([p[k] for p in parts], axis=1) for k in (0, 1))
    std = Standardize().fit(val["x"][fit])
    xv, xt = f32(std(val["x"])), f32(std(test["x"]))
    ts = FBTangentSpace().fit(val["cov"][fit].reshape(int(fit.sum()), -1))
    rv, rt = (ts.transform(np.nan_to_num(d["cov"]).reshape(len(d["cov"]), -1)) for d in (val, test))
    rstd = Standardize().fit(rv[fit])
    rv, rt = f32(rstd(rv)), f32(rstd(rt))
    sv, st = f32(sex_block(val["rows"])), f32(sex_block(test["rows"]))
    for d in (val, test):                           # the raw matrices are no longer needed
        d.pop("x"), d.pop("cov")
    hand, riem, sex = (xv, xt), (rv, rt), (sv, st)
    sets = {"recalibration only": (np.zeros((len(xv), 0)), np.zeros((len(xt), 0)))}
    sets.update({f: (lambda i=i: (xv[:, i], xt[:, i])) for f, i in fam_idx.items()})
    b_idx = np.flatnonzero(cols.column.str.split(":").str[1].isin(PAPER_B).to_numpy())
    sets["paper B: band power, entropy, Hjorth"] = lambda: (xv[:, b_idx], xt[:, b_idx])
    sets.update({"riemann": riem, "sex": sex, "all handcrafted": hand,
                 "all + riemann": cat(hand, riem), PRIMARY: cat(hand, riem, sex)})
    if args.val_state:                              # periodic / aperiodic spectra per alpha state (paper A)
        sval, stest = (_state_block(d, part["stems"]) for d, part in ((args.val_state, val), (args.test_state, test)))
        scols = pd.read_parquet(os.path.join(args.val_state, "columns.parquet")).family.to_numpy()
        sstd = Standardize().fit(sval[fit])
        sval, stest = f32(sstd(sval)), f32(sstd(stest))

        def state(*families):
            i = np.flatnonzero(np.isin(scols, [f"state/{f}" for f in families]))
            return lambda: (sval[:, i], stest[:, i])
        paper_a = state("high/periodic", "high/aperiodic", "low/periodic", "low/aperiodic")
        sets.update({PAPER_A: paper_a,
                     "paper A, high alpha only": state("high/periodic", "high/aperiodic"),
                     "paper A, low alpha only": state("low/periodic", "low/aperiodic"),
                     "paper A, no state split": state("all/periodic", "all/aperiodic"),
                     "full spectrum, high + low alpha": state("high/logpsd", "low/logpsd"),
                     EVERYTHING: lambda: cat(hand, riem, sex, paper_a())()})
    if args.val_study:                              # the method study: one set per analysis choice
        fam = {}                                     # family -> standardized (val, test), float32
        core_names = None
        for dv, dt in zip(args.val_study, args.test_study):
            ycols = pd.read_parquet(os.path.join(dv, "columns.parquet"))
            yfam, yname = ycols.family.to_numpy(), ycols.column.str.split(":").str[-1].to_numpy().astype(str)
            mats = [(np.load(os.path.join(d, "features.npy"), mmap_mode="r"), _rows(d, part["stems"]))
                    for d, part in ((dv, val), (dt, test))]
            for f in dict.fromkeys(yfam):            # each family is a contiguous block of columns
                c = np.flatnonzero(yfam == f)
                a, b = (np.asarray(m[:, c[0]:c[-1] + 1][r], np.float64) for m, r in mats)
                if f.startswith("cov_"):             # band covariances -> Riemannian tangent space
                    t = FBTangentSpace().fit(np.nan_to_num(a[fit]))
                    a, b = t.transform(np.nan_to_num(a)), t.transform(np.nan_to_num(b))
                sd = Standardize().fit(a[fit])
                fam[f] = (sd(a).astype(np.float32), sd(b).astype(np.float32))
                if f == "len10":
                    core_names = yname[c]

        def join(*families):
            return cat(*(fam[f] for f in families))
        sets.update({f"study: {f}": v for f, v in fam.items()})
        if core_names is not None:
            sub_cols = {"psd_bins": np.char.startswith(core_names, "log_psd_"), "time": np.isin(core_names, TIME_NAMES)}
            sub_cols["spectral"] = ~sub_cols["psd_bins"] & ~sub_cols["time"]
            for f in ("len10", "len30", "len60"):
                for sub, m in sub_cols.items():
                    sets[f"study: {f} [{sub}]"] = lambda f=f, m=m: tuple(fam[f][k][:, m] for k in (0, 1))
        for name, parts in (("study: state_high + state_low", ("state_high", "state_low")),
                            ("study: median + q10 + q90 + sd", ("len10", "agg_q10", "agg_q90", "agg_sd")),
                            ("study: conn_high + conn_low", ("conn_high", "conn_low")),
                            ("study: cov_high + cov_low", ("cov_high", "cov_low")),
                            ("study: core + conn_all + cov_all", ("len10", "conn_all", "cov_all"))):
            if all(q in fam for q in parts):
                sets[name] = join(*parts)
    if args.feature_model:                          # ages predicted from the features by the train-fitted model
        fm = pd.read_parquet(args.feature_model).set_index("stem")
        names = [c for c in fm.columns if c not in ("age", "split")]
        fv, ft = ((fm.loc[d["stems"], names].to_numpy(float) - 50.0) / 20.0 for d in (val, test))
        sets.update({f"feature model: {n}": (fv[:, [i]], ft[:, [i]]) for i, n in enumerate(names)
                     if n in (PRIMARY, EVERYTHING, PAPER_A, "paper B: band power, entropy, Hjorth",
                              "all handcrafted", "riemann", "sex")})
        sets["feature model: every set"] = (fv, ft)
    for spec in args.combine or []:                 # NAME=set1+set2: columns of several sets side by side
        name, parts = spec.split("=", 1)
        parts = [q.strip() for q in parts.split("+")]
        sets[name] = lambda parts=parts, pool=sets: cat(*(resolve(pool[q]) for q in parts))()
        args.sets = (args.sets or []) + [name]
    if args.sets or args.set_prefix:
        keep = set(args.sets or []) | {"recalibration only"}
        sets = {k: v for k, v in sets.items() if k in keep or any(k.startswith(p) for p in args.set_prefix or [])}

    def resolve(v):
        return v() if callable(v) else v

    def design(d):                                  # unpenalised: intercept + the model's prediction
        return np.c_[np.ones(len(d["pred"])), (d["pred"] - 50.0) / 20.0]
    zv, zt = design(val), design(test)
    res_v = val["age"] - val["pred"]
    folds = grouped_folds(val["patient"][fit])
    age_rows = []
    if args.two_stage:                              # each block -> one column: the age it predicts on its own
        staged = {"recalibration only": sets["recalibration only"]}
        for name, v in sets.items():
            a, b = resolve(v)
            if a.shape[1] == 0 or name.startswith("feature model"):
                continue
            t0 = time.time()
            fv, ft, alpha, oof_mae = cross_fitted_age(a, b, val["age"], fit, folds, ALPHAS)
            e = ft - test["age"]
            age_rows.append({"features": name, "columns": a.shape[1], "alpha": alpha, "val_oof_mae": oof_mae,
                             "test_mae": float(np.abs(e).mean()), "test_bias": float(e.mean()),
                             "test_slope": slope(ft, test["age"]),
                             "corr_with_base_error": float(np.corrcoef(ft - test["age"], test["pred"] - test["age"])[0, 1])})
            print(f"age from {name:38s} p={a.shape[1]:5d} alpha={alpha:9.0f}  val OOF {oof_mae:.3f}  "
                  f"test {age_rows[-1]['test_mae']:.3f}  slope {age_rows[-1]['test_slope']:.3f}  "
                  f"error corr with P4 {age_rows[-1]['corr_with_base_error']:.2f}  [{time.time() - t0:.0f}s]", flush=True)
            staged[f"age model: {name}"] = ((fv[:, None] - 50.0) / 20.0, (ft[:, None] - 50.0) / 20.0)
        staged["age model: every block"] = tuple(np.concatenate([v[i] for k, v in staged.items() if k != "recalibration only"],
                                                                axis=1) for i in (0, 1))
        # the feature blocks combined without the fold model: how far the features get on their own
        fv, ft, alpha, oof_mae = cross_fitted_age(*staged["age model: every block"], val["age"], fit, folds, ALPHAS)
        e = ft - test["age"]
        age_rows.append({"features": "every block's prediction combined", "columns": staged["age model: every block"][0].shape[1],
                         "alpha": alpha, "val_oof_mae": oof_mae, "test_mae": float(np.abs(e).mean()),
                         "test_bias": float(e.mean()), "test_slope": slope(ft, test["age"]),
                         "corr_with_base_error": float(np.corrcoef(e, test["pred"] - test["age"])[0, 1])})
        print(f"age from every block's prediction combined: val OOF {oof_mae:.3f}  test {age_rows[-1]['test_mae']:.3f}  "
              f"slope {age_rows[-1]['test_slope']:.3f}  error corr with P4 {age_rows[-1]['corr_with_base_error']:.2f}", flush=True)
        sets = staged
    base_v, base_t = np.abs(res_v[fit]), np.abs(test["age"] - test["pred"])
    rows, out_pred = [], {"stem": test["stems"], "age": test["age"], "base": test["pred"]}
    for name, v in sets.items():
        a, b = resolve(v)
        t0 = time.time()
        af, zf, rf = a[fit], zv[fit], res_v[fit]
        alphas = ALPHAS if a.shape[1] else ALPHAS[:1]
        oof = np.zeros((len(alphas), len(rf)))
        for k in range(folds.max() + 1):
            tr, te = folds != k, folds == k
            oof[:, te] = ridge_path(zf[tr], af[tr], rf[tr], alphas)(zf[te], af[te])
        cv_mae = np.abs(rf[None] - oof).mean(1)
        best = int(cv_mae.argmin())
        corr = ridge_path(zf, af, rf, alphas[best:best + 1])(zt, b)[0]
        new = test["pred"] + corr
        err = np.abs(test["age"] - new)
        lo, hi = paired_bootstrap(err - base_t, test["patient"])
        row = {"features": name, "columns": a.shape[1], "alpha": float(alphas[best]),
               "val_mae_base": float(base_v.mean()), "val_cv_mae": float(cv_mae[best]),
               "val_cv_delta": float(cv_mae[best] - base_v.mean()),
               "test_mae_base": float(base_t.mean()), "test_mae": float(err.mean()),
               "test_delta": float((err - base_t).mean()), "delta_lo": lo, "delta_hi": hi,
               "test_slope_base": slope(test["pred"], test["age"]), "test_slope": slope(new, test["age"]),
               "test_r2": float(1 - ((test["age"] - new) ** 2).sum() / ((test["age"] - test["age"].mean()) ** 2).sum())}
        rows.append(row)
        out_pred[name] = new
        print(f"{name:38s} p={a.shape[1]:5d} alpha={row['alpha']:9.0f}  val CV {row['val_cv_mae']:.3f} "
              f"({row['val_cv_delta']:+.3f})  test {row['test_mae']:.3f} ({row['test_delta']:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}])  slope {row['test_slope']:.3f}  [{time.time() - t0:.0f}s]", flush=True)
    os.makedirs(args.out, exist_ok=True)
    if age_rows:
        pd.DataFrame(age_rows).to_csv(os.path.join(args.out, "age_model_summary.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "stack_summary.csv"), index=False)
    pd.DataFrame(out_pred).to_parquet(os.path.join(args.out, "test_predictions.parquet"))   # per recording: stays local

    # subgroups of the test split (aggregates), for the baseline, the control and the headline set
    headline = args.headline or next((n for n in (f"feature model: {EVERYTHING}", f"feature model: {PRIMARY}",
                                                  f"age model: {EVERYTHING}", f"age model: {PRIMARY}", EVERYTHING, PRIMARY)
                                      if n in out_pred), PRIMARY)
    r = test["rows"]
    ds = np.where(r.corpus == "heedb", "harvard", "temple")
    label = r.label.astype(object).where(r.label.notna(), None).to_numpy()
    if args.labels:                                  # Temple report verdicts live in the age metadata
        lab = {k: v.get("label") for k, v in json.load(open(args.labels))["recordings"].items()}
        label = np.array([lab.get(s) or l for s, l in zip(test["stems"], label)], dtype=object)
    status = np.array([l if l in ("normal", "abnormal") else "no report" for l in label])
    groups = [("all", np.ones(len(ds), bool))]
    groups += [(d, ds == d) for d in ("temple", "harvard")]
    groups += [(f"{d} {c}", (ds == d) & (status == c)) for d in ("temple", "harvard")
               for c in ("normal", "abnormal", "no report")]
    groups += [(f"age {lo}-{hi - 1}" if hi < 200 else f"age {lo}+", (test["age"] >= lo) & (test["age"] < hi))
               for lo, hi in AGE_BINS]
    sub = []
    for g, m in groups:
        if m.sum() < 20:
            continue
        row = {"group": g, "n": int(m.sum()), "mean_age": float(test["age"][m].mean())}
        for name in ("base", "recalibration only", headline):
            if name not in out_pred:
                continue
            e = out_pred[name][m] - test["age"][m]
            row[f"mae {name}"], row[f"bias {name}"] = float(np.abs(e).mean()), float(e.mean())
        if headline in out_pred:
            d = np.abs(out_pred[headline][m] - test["age"][m]) - np.abs(test["pred"][m] - test["age"][m])
            row["delta"], (row["delta_lo"], row["delta_hi"]) = float(d.mean()), paired_bootstrap(d, test["patient"][m])
        sub.append(row)
    sub = pd.DataFrame(sub)
    sub.to_csv(os.path.join(args.out, "stack_subgroups.csv"), index=False)
    pd.set_option("display.width", 250)
    print(sub.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stems")
    p.add_argument("--fold-dir", required=True)
    p.add_argument("--cohort", required=True)
    p.add_argument("--out-dir", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--cohort", required=True)
    p.add_argument("--stems", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--kind", choices=sorted(KINDS), default="main",
                   help="main: handcrafted families + covariances; state: periodic / aperiodic spectra per alpha state")
    p.add_argument("--jobs", type=int, default=7)
    p.add_argument("--chunk", type=int, default=512)
    p = sub.add_parser("merge")
    p.add_argument("--parts", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--cohort", required=True)
    p.add_argument("--preds", required=True, help="eval_cv_folds.py --preds-out JSON")
    p.add_argument("--model", default="P4_L1")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--val", required=True, help="merged features of the validation fold")
    p.add_argument("--test", required=True, help="merged features of the test split")
    p.add_argument("--train-stems", default=None, help="the fold's training recordings (shared patients are not fitted on)")
    p.add_argument("--labels", default=None, help="age_metadata.json with the report verdicts")
    p.add_argument("--feature-model", default=None,
                   help="feature_model_predictions.parquet of the feature-model step: its ages become feature sets")
    p.add_argument("--val-state", default=None, help="merged state-spectra features (extract --kind state) of the validation fold")
    p.add_argument("--test-state", default=None, help="... and of the test split")
    p.add_argument("--val-study", nargs="+", default=None,
                   help="merged method-study features (extract --kind study / study2) of the validation fold")
    p.add_argument("--test-study", nargs="+", default=None, help="... and of the test split, same order")
    p.add_argument("--set-prefix", nargs="+", default=None, help="also run every feature set whose name starts with one of these")
    p.add_argument("--combine", nargs="+", default=None, metavar="NAME=SET+SET",
                   help="extra feature sets made of the columns of existing ones (set names without '+')")
    p.add_argument("--two-stage", action="store_true",
                   help="each feature set first predicts age on its own (cross-fitted on the validation fold); "
                        "the stacker then combines that one column with the fold model")
    p.add_argument("--headline", default=None, help="feature set of the subgroup table")
    p.add_argument("--sets", nargs="+", default=None, help="feature sets to run (default: all)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("feature-model")
    p.add_argument("--cohort", required=True)
    p.add_argument("--train", required=True, help="merged features of the fold's training recordings")
    p.add_argument("--val", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--train-state", default=None, help="merged state-spectra features of the training recordings")
    p.add_argument("--val-state", default=None)
    p.add_argument("--test-state", default=None)
    p.add_argument("--ref-rows", type=int, default=10000, help="training recordings behind the scaling and Riemann means")
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    {"stems": cmd_stems, "extract": cmd_extract, "merge": cmd_merge, "fit": cmd_fit,
     "feature-model": cmd_feature_model}[args.cmd](args)


if __name__ == "__main__":
    main()
