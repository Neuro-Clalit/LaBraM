# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Brain-age error analysis: per-window / per-recording predictions, error by age,
# post-hoc bias correction and spectral correlates of the residual.
# ---------------------------------------------------------
"""Error analysis for a trained brain-age (``TUAB_AGE``) fine-tune.

Drives ``notebooks/age_error_analysis.ipynb``. :func:`load_age_run` rebuilds a
run's model and its exact val/test samples (window selection, preprocessing)
from ``run_config.yaml``; :func:`predict_windows` scores every window and keeps
its dataset index, recording and window number, so any prediction can be traced
back to the EEG it came from (:func:`load_window`). The remaining functions are
pure pandas/numpy over those predictions.

Artifact-aware pooling (:func:`window_artifact_features`,
:func:`artifact_pooled`, :func:`artifact_pooling_sweep`) re-pools cached window
predictions after dropping or down-weighting windows with high EMG (muscle)
power, which the model reads as "young". ``scripts/age_artifact_pooling.py``
runs it over trained runs without re-scoring the model.

Bias-correction conventions (``pred = slope * age + intercept`` fitted on val):

* ``"cole"`` -- ``(pred - intercept) / slope`` (Cole et al. 2018). Uses only the
  prediction, so it is a deployable age estimate.
* ``"recalibrate"`` -- least squares of age on prediction. Deployable and
  MSE-optimal, but it shrinks toward the mean even more.
* ``"age_level"`` -- ``pred - (slope - 1) * age - intercept`` (de Lange & Cole
  2020). Needs the true age: valid for brain-age-gap analyses, not as an age
  estimate.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

AGE_BIN_EDGES = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90)
BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
         "beta": (13.0, 30.0), "gamma": (30.0, 45.0)}
POSTERIOR = ("O1", "O2", "P3", "P4", "PZ", "T5", "T6")
# Muscle (EMG) artifact index band. TUAB is 0.1-75 Hz band-passed with a 50 Hz
# notch (dataset_maker/make_TUAB.py) but recorded on 60 Hz mains, so the band
# stops below 60 Hz line noise; 30-45 Hz is little affected by beta rhythm.
EMG_BAND = (30.0, 45.0)
POOLING_METHODS = ("reject", "weight")
CORRECTIONS = ("none", "cole", "recalibrate", "age_level")
_WINDOW_RE = re.compile(r"_(\d+)\.pkl$")


@dataclass
class AgeRun:
    """A trained brain-age run, rebuilt for analysis."""
    run_dir: str
    config: object
    bundle: object
    model: torch.nn.Module
    labels: Dict[str, str]          # recording stem -> "normal" / "abnormal"
    device: torch.device
    epoch: Optional[int] = None
    to_scalar: Optional[object] = None

    @property
    def target_stats(self) -> Tuple[float, float]:
        return self.bundle.target_stats

    @property
    def ch_names(self) -> List[str]:
        return list(self.bundle.ch_names)

    def split(self, name: str):
        return getattr(self.bundle, name)


def load_age_bundle(run_dir: str, data_path: str):
    """``(run config, window-selected dataset bundle)`` of ``run_dir`` -- the
    exact samples the run was evaluated on, without building its model."""
    from labram.data import get_dataset_bundle
    from labram.data.window_selection import WindowSelection, apply_window_selection
    from labram.eval.loading import load_run_config

    cfg = load_run_config(os.path.join(run_dir, "run_config.yaml"))
    cfg.data.data_path = data_path
    bundle = get_dataset_bundle(cfg.data.dataset, data_path, data_format=cfg.data.data_format)
    return cfg, apply_window_selection(bundle, WindowSelection.from_data_config(cfg.data))


def load_age_run(run_dir: str, data_path: str, checkpoint: str = "checkpoint-best.pth",
                 device: Optional[str] = None) -> AgeRun:
    """Rebuild ``run_dir``'s model (EMA weights when the run selected on them)
    and its window-selected train/val/test datasets."""
    import labram.models.registry  # noqa: F401  (registers the timm models)
    from labram.data.tuh_metadata import load_label_lookup_for
    from labram.losses import build_downstream_criterion, regression_output, soft_label_n_bins
    from labram.runs.run_finetune import get_model

    cfg, bundle = load_age_bundle(run_dir, data_path)
    cfg.model.nb_classes, cfg.model.task = bundle.nb_classes, bundle.task
    to_scalar = None
    if cfg.loss.regression_loss == "soft_label":
        cfg.model.nb_classes = soft_label_n_bins(cfg.loss)
        crit = build_downstream_criterion("regression", cfg.model.nb_classes, cfg.loss,
                                          target_stats=bundle.target_stats)
        to_scalar = lambda out: regression_output(crit, out)   # noqa: E731
    model = get_model(cfg)
    state = torch.load(os.path.join(run_dir, checkpoint), map_location="cpu", weights_only=False)
    key = "model_ema" if (cfg.evaluation.use_ema and "model_ema" in state) else "model"
    model.load_state_dict(state[key])
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev).eval()
    labels = load_label_lookup_for(getattr(bundle.test, "root", data_path))
    return AgeRun(run_dir, cfg, bundle, model, labels, dev, state.get("epoch"), to_scalar)


def window_number(filename: str) -> int:
    """Window index within its recording, from ``<split>/<stem>_<k>.pkl``."""
    m = _WINDOW_RE.search(filename)
    return int(m.group(1)) if m else -1


def subsample_per_recording(dataset, per_recording: int, seed: int = 0) -> np.ndarray:
    """Dataset indices keeping at most ``per_recording`` random windows of each
    recording (e.g. to score the large train split cheaply)."""
    from labram.data.tuh_metadata import recording_stem
    rng = np.random.default_rng(seed)
    by_rec: Dict[str, List[int]] = {}
    for i, f in enumerate(dataset.files):
        by_rec.setdefault(recording_stem(f), []).append(i)
    keep = []
    for idx in by_rec.values():
        keep.extend(idx if len(idx) <= per_recording
                    else rng.choice(idx, per_recording, replace=False).tolist())
    return np.sort(np.asarray(keep, dtype=np.int64))


def predict_windows(run: AgeRun, split: str, indices: Optional[Sequence[int]] = None,
                    batch_size: int = 256, num_workers: int = 8) -> pd.DataFrame:
    """One row per scored window: ``split, idx, recording, window, cohort, age, pred``
    (ages in years). ``indices`` restricts scoring to those dataset indices."""
    from einops import rearrange
    import labram.utils as utils
    from labram.data.tuh_metadata import recording_stem

    dataset = run.split(split)
    idx = np.arange(len(dataset)) if indices is None else np.asarray(indices, dtype=np.int64)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, idx.tolist()), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=run.device.type == "cuda")
    channel_indices = utils.get_channel_indices(run.ch_names)
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            x = rearrange(x.float().to(run.device) / 100, "B N (A T) -> B N A T", T=200)
            with torch.amp.autocast(run.device.type, enabled=run.device.type == "cuda"):
                out = run.model(x, channel_indices=channel_indices, classify_only=True)
            out = getattr(out, "logits", out)
            if run.to_scalar is not None:
                out = run.to_scalar(out)
            preds.append(out.float().reshape(len(y), -1)[:, 0].cpu())
            targets.append(y.float().reshape(-1))
    mean, std = run.target_stats
    files = [dataset.files[i] for i in idx]
    recs = [recording_stem(f) for f in files]
    return pd.DataFrame({
        "split": split, "idx": idx, "recording": recs,
        "window": [window_number(f) for f in files],
        "cohort": [run.labels.get(r, "unknown") for r in recs],
        "age": torch.cat(targets).numpy() * std + mean,
        "pred": torch.cat(preds).numpy() * std + mean,
    })


def add_errors(df: pd.DataFrame, pred: str = "pred") -> pd.DataFrame:
    """Signed ``err = pred - age`` and ``abs_err`` columns (in place, returned)."""
    df["err"] = df[pred] - df["age"]
    df["abs_err"] = df["err"].abs()
    return df


def per_recording(windows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate window predictions per recording: mean (the reported metric),
    median and spread of the window predictions, plus their errors."""
    g = windows.groupby(["split", "recording"], sort=False)
    rec = g.agg(cohort=("cohort", "first"), age=("age", "first"), pred=("pred", "mean"),
                pred_median=("pred", "median"), pred_std=("pred", "std"),
                pred_min=("pred", "min"), pred_max=("pred", "max"),
                n_windows=("pred", "size")).reset_index()
    rec["pred_std"] = rec["pred_std"].fillna(0.0)
    add_errors(rec)
    rec["abs_err_median_pool"] = (rec["pred_median"] - rec["age"]).abs()
    return rec


def regression_summary(df: pd.DataFrame, pred: str = "pred") -> Dict[str, float]:
    """MAE / median AE / P90 AE / RMSE / R^2 / r / mean bias / bias slope."""
    p, t = df[pred].to_numpy(float), df["age"].to_numpy(float)
    e = p - t
    ae = np.abs(e)
    ss_tot = float(((t - t.mean()) ** 2).sum())
    slope = float(np.polyfit(t, e, 1)[0]) if len(t) > 1 and t.std() > 0 else float("nan")
    return {
        "n": int(len(t)), "mae": float(ae.mean()), "median_ae": float(np.median(ae)),
        "p90_ae": float(np.percentile(ae, 90)), "rmse": float(np.sqrt((e ** 2).mean())),
        "r2": 1.0 - float((e ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan"),
        "pearson_r": float(np.corrcoef(p, t)[0, 1]) if len(t) > 1 else float("nan"),
        "mean_err": float(e.mean()), "median_err": float(np.median(e)),
        "bias_slope": slope,
    }


def summary_table(frames: Dict[str, pd.DataFrame], pred: str = "pred") -> pd.DataFrame:
    """:func:`regression_summary` for every ``level -> frame`` x split x cohort
    (cohort ``all`` included)."""
    rows = []
    for level, df in frames.items():
        for split, d in df.groupby("split", sort=False):
            for cohort, c in [("all", d)] + list(d.groupby("cohort", sort=True)):
                rows.append({"level": level, "split": split, "cohort": cohort,
                             **regression_summary(c, pred)})
    return pd.DataFrame(rows).set_index(["level", "split", "cohort"])


def age_bin_labels(edges: Sequence[float] = AGE_BIN_EDGES) -> List[str]:
    return [f"{int(a)}-{int(b) - 1}" for a, b in zip(edges[:-1], edges[1:])]


def assign_age_bin(age, edges: Sequence[float] = AGE_BIN_EDGES) -> pd.Categorical:
    return pd.cut(age, bins=list(edges), right=False, labels=age_bin_labels(edges))


def error_by_age_bin(df: pd.DataFrame, edges: Sequence[float] = AGE_BIN_EDGES,
                     err: str = "err", by: Optional[List[str]] = None) -> pd.DataFrame:
    """Per age bin: count, mean true age, mean/median/IQR of the predicted age
    (when ``df`` has ``pred``), mean/median signed error, MAE, median AE and IQR
    of the absolute error. ``by`` adds grouping columns (e.g. ``["split"]``)."""
    d = df.assign(age_bin=assign_age_bin(df["age"], edges), _ae=df[err].abs())
    keys = (by or []) + ["age_bin"]
    g = d.groupby(keys, observed=True)
    pred_stats = {}
    if "pred" in d:
        pred_stats = dict(pred_mean=("pred", "mean"), pred_median=("pred", "median"),
                          pred_q25=("pred", lambda s: s.quantile(0.25)),
                          pred_q75=("pred", lambda s: s.quantile(0.75)))
    out = g.agg(n=(err, "size"), age_mean=("age", "mean"), **pred_stats,
                mean_err=(err, "mean"), median_err=(err, "median"),
                mae=("_ae", "mean"), median_ae=("_ae", "median"),
                ae_q25=("_ae", lambda s: s.quantile(0.25)),
                ae_q75=("_ae", lambda s: s.quantile(0.75)))
    return out


def fit_bias_correction(pred, age) -> Dict[str, float]:
    """Fit on a calibration set (val): ``pred ~ slope * age + intercept`` and
    ``age ~ a * pred + b``."""
    pred, age = np.asarray(pred, float), np.asarray(age, float)
    slope, intercept = np.polyfit(age, pred, 1)
    a, b = np.polyfit(pred, age, 1)
    return {"slope": float(slope), "intercept": float(intercept),
            "recal_a": float(a), "recal_b": float(b)}


def apply_bias_correction(pred, params: Dict[str, float], method: str,
                          age=None) -> np.ndarray:
    """Corrected predictions (see the module docstring for the methods)."""
    pred = np.asarray(pred, float)
    if method == "none":
        return pred
    if method == "cole":
        return (pred - params["intercept"]) / params["slope"]
    if method == "recalibrate":
        return params["recal_a"] * pred + params["recal_b"]
    if method == "age_level":
        if age is None:
            raise ValueError("age_level correction needs the true age")
        return pred - (params["slope"] - 1.0) * np.asarray(age, float) - params["intercept"]
    raise ValueError(f"unknown correction {method!r} (expected one of {CORRECTIONS})")


def load_window(dataset, idx: int) -> np.ndarray:
    """The window's signal as stored (``[channels, samples]``, microvolts)."""
    item = dataset[int(idx)]
    return item[0].numpy() if torch.is_tensor(item[0]) else np.asarray(item[0])


def common_average_reference(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


def band_powers(x: np.ndarray, fs: float = 200.0, bands: Dict[str, Tuple[float, float]] = BANDS,
                total: Tuple[float, float] = (1.0, 45.0)) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Welch PSD of ``x`` (``[..., samples]``) and its relative band powers.

    Returns ``(relative power per band [...], freqs, psd [..., freqs])``.
    """
    from scipy.signal import welch
    freqs, psd = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    in_total = (freqs >= total[0]) & (freqs < total[1])
    denom = psd[..., in_total].sum(-1)
    rel = {name: psd[..., (freqs >= lo) & (freqs < hi)].sum(-1) / denom
           for name, (lo, hi) in bands.items()}
    return rel, freqs, psd


def peak_alpha_frequency(freqs: np.ndarray, psd: np.ndarray,
                         band: Tuple[float, float] = (7.0, 13.0)) -> np.ndarray:
    """Frequency of the PSD maximum inside ``band`` (last axis = frequency)."""
    m = (freqs >= band[0]) & (freqs <= band[1])
    return freqs[m][np.argmax(psd[..., m], axis=-1)]


def recording_spectral_features(dataset, windows: pd.DataFrame, ch_names: Sequence[str],
                                max_windows: int = 6, fs: float = 200.0,
                                seed: int = 0) -> pd.DataFrame:
    """Per recording (CAR applied): channel-averaged relative band powers,
    posterior peak alpha frequency, theta/alpha ratio and mean amplitude over
    up to ``max_windows`` of its scored windows."""
    post = [i for i, c in enumerate(ch_names) if c.upper() in POSTERIOR]
    rng = np.random.default_rng(seed)
    rows = []
    for (split, rec), d in windows.groupby(["split", "recording"], sort=False):
        idx = d["idx"].to_numpy()
        if len(idx) > max_windows:
            idx = rng.choice(idx, max_windows, replace=False)
        x = np.stack([common_average_reference(load_window(dataset, i)) for i in idx])
        rel, freqs, psd = band_powers(x, fs)
        mean_psd = psd.mean(0)                                   # [channels, freqs]
        row = {"split": split, "recording": rec,
               "paf": float(peak_alpha_frequency(freqs, mean_psd[post].mean(0))),
               "amp_uv": float(x.std(-1).mean())}
        row.update({f"rel_{b}": float(v.mean()) for b, v in rel.items()})
        row["theta_alpha"] = row["rel_theta"] / max(row["rel_alpha"], 1e-9)
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------- artifact-aware pooling
def window_artifact_features(dataset, windows: pd.DataFrame, fs: float = 200.0,
                             emg_band: Tuple[float, float] = EMG_BAND,
                             total: Tuple[float, float] = (1.0, 45.0)) -> pd.DataFrame:
    """Per scored window of ``dataset`` (CAR applied): ``emg`` = relative
    ``emg_band`` power averaged over channels, ``emg_max`` = its channel maximum
    (muscle artifact is often focal: temporal/frontal) and ``amp_uv``.

    ``windows`` holds that split's rows of :func:`predict_windows`; the result is
    keyed by ``split, idx`` for a merge back onto them.
    """
    rows = windows[["split", "idx"]].drop_duplicates()
    emg, emg_max, amp = [], [], []
    for i in rows["idx"].to_numpy():
        x = common_average_reference(load_window(dataset, i))
        rel, _, _ = band_powers(x, fs, bands={"emg": emg_band}, total=total)
        emg.append(float(rel["emg"].mean()))
        emg_max.append(float(rel["emg"].max()))
        amp.append(float(x.std(-1).mean()))
    return rows.assign(emg=emg, emg_max=emg_max, amp_uv=amp).reset_index(drop=True)


def artifact_weights(score, threshold: float, method: str = "reject") -> np.ndarray:
    """Pooling weight of each window from its artifact ``score``: ``"reject"``
    drops windows above ``threshold``; ``"weight"`` down-weights them smoothly,
    ``1 / (1 + (score / threshold) ** 4)`` (0.5 at the threshold)."""
    score = np.asarray(score, float)
    if method == "reject":
        return (score <= threshold).astype(float)
    if method == "weight":
        if not np.isfinite(threshold):
            return np.ones_like(score)
        return 1.0 / (1.0 + (score / max(threshold, 1e-12)) ** 4)
    raise ValueError(f"unknown pooling method {method!r} (expected one of {POOLING_METHODS})")


def artifact_pooled(windows: pd.DataFrame, score: str = "emg", threshold: float = np.inf,
                    method: str = "reject", min_keep: int = 1) -> pd.DataFrame:
    """Per-recording prediction as the artifact-weighted mean of its windows.

    With ``threshold=inf`` this is the plain mean (:func:`per_recording`). Every
    recording keeps at least its ``min_keep`` cleanest windows at full weight,
    so one that is artifact throughout is still scored. Adds ``n_kept`` (sum of
    weights) and ``frac_kept`` per recording.
    """
    w = artifact_weights(windows[score], threshold, method)
    if min_keep > 0:
        rank = windows.groupby(["split", "recording"], sort=False)[score].rank(method="first")
        w = np.where(rank.to_numpy() <= min_keep, np.maximum(w, 1.0), w)
    d = windows.assign(_w=w, _wp=w * windows["pred"].to_numpy(float))
    g = d.groupby(["split", "recording"], sort=False)
    rec = g.agg(cohort=("cohort", "first"), age=("age", "first"), _wp=("_wp", "sum"),
                n_kept=("_w", "sum"), n_windows=("pred", "size")).reset_index()
    rec["pred"] = rec.pop("_wp") / rec["n_kept"]
    rec["frac_kept"] = rec["n_kept"] / rec["n_windows"]
    return add_errors(rec)


def artifact_pooling_sweep(windows: pd.DataFrame, score: str = "emg",
                           quantiles: Sequence[float] = (1.0, 0.99, 0.97, 0.95, 0.9, 0.8, 0.7, 0.5),
                           methods: Sequence[str] = POOLING_METHODS, ref_split: str = "val",
                           min_keep: int = 1) -> pd.DataFrame:
    """Recording-level metrics of :func:`artifact_pooled` over a grid of
    thresholds, one row per ``method x quantile x split x cohort``.

    Thresholds are quantiles of ``score`` over the ``ref_split`` windows (fixed
    there, applied unchanged to every split); quantile 1.0 means no rejection,
    i.e. the plain-mean baseline. Pick the setting on val
    (:func:`select_pooling`), read its test row.
    """
    ref = windows.loc[windows["split"] == ref_split, score]
    rows = []
    for method in methods:
        for q in quantiles:
            thr = np.inf if q >= 1.0 else float(ref.quantile(q))
            rec = artifact_pooled(windows, score, thr, method, min_keep)
            for split, d in rec.groupby("split", sort=False):
                for cohort, dc in [("all", d)] + list(d.groupby("cohort", sort=True)):
                    s = regression_summary(dc)
                    rows.append({"method": method, "quantile": q, "threshold": thr,
                                 "split": split, "cohort": cohort, "n": s["n"],
                                 "mae": s["mae"], "rmse": s["rmse"], "mean_err": s["mean_err"],
                                 "bias_slope": s["bias_slope"],
                                 "frac_kept": float(dc["frac_kept"].mean())})
    return pd.DataFrame(rows)


def select_pooling(sweep: pd.DataFrame, split: str = "val") -> pd.Series:
    """The ``method, quantile`` with the lowest ``split`` MAE (cohort ``all``),
    with that setting's ``mae_<split>`` for every split."""
    allc = sweep[sweep["cohort"] == "all"]
    wide = allc.pivot_table(index=["method", "quantile"], columns="split", values="mae")
    wide.columns = [f"mae_{c}" for c in wide.columns]
    best = wide[f"mae_{split}"].idxmin()
    return pd.concat([pd.Series({"method": best[0], "quantile": best[1]}), wide.loc[best]])


def read_epoch_log(run_dir: str) -> pd.DataFrame:
    """The run's ``log.txt`` (one JSON object per epoch) as a frame."""
    with open(os.path.join(run_dir, "log.txt")) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    df = pd.DataFrame(rows)
    if "epoch" not in df:
        df.insert(0, "epoch", np.arange(len(df)))
    return df


def pick_examples(rec: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Labelled example recordings: the best, the most over- and the most
    under-predicted ones (by signed recording error)."""
    parts = [rec.nsmallest(n, "abs_err").assign(kind="best"),
             rec.nlargest(n, "err").assign(kind="over-predicted"),
             rec.nsmallest(n, "err").assign(kind="under-predicted")]
    return pd.concat(parts, ignore_index=True)
