# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Brain-age plots: ages on the original scale, real EEG of well / badly predicted
# recordings, and an interactive explorer for notebooks.
# ---------------------------------------------------------
"""Plotting for ``notebooks/age_error_analysis.ipynb``.

Every age axis is in years on the original (de-normalized) scale. The EEG is the
stored signal (microvolts), re-referenced to the common average as the model's
input is (``labram_plus`` CAR), at the same vertical scale in every panel of a
figure so amplitudes compare directly.

:class:`AgeExplorer` bundles a run with its window/recording predictions:

    ex = AgeExplorer(run, win, rec, feats)
    ex.find(age=65, kind="under")            # candidates, as a table
    ex.show("aaaaalvz_s001_t000")             # EEG + per-window predictions + spectrum
    ex.compare_at_age(65)                     # accurate vs under- vs over-predicted, same age
    ex.widget()                               # interactive browser (Jupyter)
"""

import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from labram.eval.age_analysis import (
    POSTERIOR, add_errors, band_powers, common_average_reference, error_by_age_bin, load_window,
    predict_windows, subsample_per_recording)

AGE_LIM = (0, 92)
STANDARD_19 = ("FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ", "C4", "T4",
               "T5", "P3", "PZ", "P4", "T6", "O1", "O2")
CHANNEL_SETS = {"10-20 (19)": STANDARD_19, "posterior": POSTERIOR, "all": None}
KIND_COLORS = {"accurate": "#2ca02c", "under-predicted": "#1f77b4", "over-predicted": "#d62728"}
BAND_COLORS = {"delta": "#8c564b", "theta": "#e377c2", "alpha": "#2ca02c", "beta": "#17becf"}
FS = 200.0
EEG_DPI = 90        # dense EEG traces make large PNGs; this keeps a saved notebook manageable


# ----------------------------------------------------------------- caching
def cached_window_predictions(run_dir: str, data_path: str, splits=("val", "test", "train"),
                              run=None, train_per_recording: int = 10) -> pd.DataFrame:
    """Window predictions of ``run_dir`` (with errors), computed once and cached
    as ``<run_dir>/analysis/window_predictions.parquet``. The cache is redone
    when ``checkpoint-best.pth`` is newer or a requested split is missing. The
    train split is scored on ``train_per_recording`` windows per recording."""
    from labram.eval.age_analysis import load_age_run
    path = os.path.join(run_dir, "analysis", "window_predictions.parquet")
    ckpt = os.path.join(run_dir, "checkpoint-best.pth")
    if os.path.exists(path) and os.path.getmtime(path) >= os.path.getmtime(ckpt):
        df = pd.read_parquet(path)
        if set(splits) <= set(df["split"]):
            return add_errors(df[df["split"].isin(splits)].copy())
    run = run or load_age_run(run_dir, data_path)
    frames = []
    for split in splits:
        idx = (subsample_per_recording(run.split(split), train_per_recording)
               if split == "train" else None)
        frames.append(predict_windows(run, split, idx))
    df = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)
    return add_errors(df)


# ------------------------------------------------------------ age helpers
def identity(ax, lim=AGE_LIM):
    """The diagonal where predicted age = true age (an unbiased model sits on it)."""
    ax.plot(lim, lim, "k--", lw=1, label="predicted = true")
    ax.set(xlim=lim, ylim=lim, xlabel="true age (years)", ylabel="predicted age (years)")


def plot_pred_by_bin(ax, df: pd.DataFrame, label: str, color, min_n: int = 5, marker: str = "o"):
    """Median predicted age per true-age decade (IQR bars) at the decade's mean true age."""
    b = error_by_age_bin(df)
    b = b[b.n >= min_n]
    ax.errorbar(b.age_mean, b.pred_median,
                yerr=[b.pred_median - b.pred_q25, b.pred_q75 - b.pred_median],
                fmt=marker + "-", color=color, ms=4, capsize=2, lw=1.3, label=label)


# -------------------------------------------------------------------- EEG
def channel_indices(ch_names: Sequence[str], channels: Optional[Sequence[str]]):
    if channels is None:
        return list(range(len(ch_names)))
    upper = [c.upper() for c in ch_names]
    return [upper.index(c.upper()) for c in channels if c.upper() in upper]


def eeg_scale(*signals: np.ndarray, factor: float = 5.0) -> float:
    """A shared trace spacing (µV, rounded to 10) for several signals."""
    s = factor * float(np.median(np.concatenate([x.std(-1) for x in signals])))
    return max(10.0, 10.0 * round(s / 10.0))


def plot_eeg(ax, x: np.ndarray, ch_names: Sequence[str], channels: Optional[Sequence[str]] = None,
             spacing: Optional[float] = None, seconds: Tuple[float, float] = (0.0, 10.0),
             color="k", fs: float = FS, title: Optional[str] = None, label_channels: bool = True,
             clip: Optional[float] = 1.0):
    """Stacked EEG traces (``x``: ``[channels, samples]``, µV) with a scale bar.

    ``clip`` (in trace spacings, like a clinical viewer's clip option) keeps a
    high-amplitude artifact from running over its neighbours; ``None`` draws
    the signal unclipped."""
    idx = channel_indices(ch_names, channels)
    spacing = spacing or eeg_scale(x[idx])
    lo, hi = int(seconds[0] * fs), int(seconds[1] * fs)
    t = np.arange(lo, min(hi, x.shape[1])) / fs
    for k, c in enumerate(idx):
        y = x[c, lo:lo + len(t)]
        if clip is not None:
            y = np.clip(y, -clip * spacing, clip * spacing)
        ax.plot(t, y - k * spacing, lw=0.55, color=color)
    ax.set_yticks([-k * spacing for k in range(len(idx))],
                  [ch_names[c] for c in idx] if label_channels else [], fontsize=7)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-(len(idx) - 0.3) * spacing, 1.2 * spacing)
    ax.set_xlabel("s")
    ax.grid(axis="y", lw=0.3)
    # Scale bar: one trace spacing, at the top right.
    x0 = t[-1] - 0.02 * (t[-1] - t[0])
    ax.plot([x0, x0], [0.6 * spacing, 0.6 * spacing - spacing], color="#c00", lw=2)
    ax.text(x0, 0.75 * spacing, f"{spacing:.0f} µV", color="#c00", ha="right", fontsize=7)
    if title:
        ax.set_title(title, fontsize=9)
    return spacing


def posterior_log_psd(windows: np.ndarray, ch_names: Sequence[str], fs: float = FS):
    """``(freqs, log10 PSD)`` averaged over windows and posterior channels."""
    _, freqs, psd = band_powers(windows, fs)
    post = channel_indices(ch_names, POSTERIOR)
    return freqs, np.log10(psd.mean(0)[post].mean(0))


# --------------------------------------------------------------- explorer
class AgeExplorer:
    """A run's predictions joined to its EEG, for browsing examples."""

    def __init__(self, run, windows: pd.DataFrame, recordings: pd.DataFrame,
                 features: Optional[pd.DataFrame] = None):
        self.run = run
        self.win = windows
        rec = recordings[recordings.split != "train"]
        if features is not None:
            rec = rec.merge(features, on=["split", "recording"], how="left")
        self.rec = rec.reset_index(drop=True)
        self.ch_names = run.ch_names

    # -- lookup ------------------------------------------------------------
    def record(self, recording: str, split: Optional[str] = None) -> pd.Series:
        r = self.rec[(self.rec.recording == recording) & ((split is None) | (self.rec.split == split))]
        if r.empty:
            raise KeyError(f"recording {recording!r} is not in the scored val/test recordings")
        return r.iloc[0]

    def windows_of(self, recording: str, split: Optional[str] = None) -> pd.DataFrame:
        split = split or self.record(recording).split
        w = self.win[(self.win.split == split) & (self.win.recording == recording)]
        return w.sort_values("window").reset_index(drop=True)

    def signal(self, split: str, idx: int, car: bool = True) -> np.ndarray:
        x = load_window(self.run.split(split), idx)
        return common_average_reference(x) if car else x

    def find(self, age: Optional[float] = None, tol: float = 3.0, kind: Optional[str] = None,
             cohort: Optional[str] = None, split: Optional[str] = None, n: int = 10) -> pd.DataFrame:
        """Recordings near ``age`` (±tol), optionally of one cohort/split, sorted
        by ``kind``: ``"accurate"`` (smallest |error|), ``"under"`` (most negative
        error: predicted too young), ``"over"`` (predicted too old)."""
        d = self.rec
        if age is not None:
            d = d[(d.age - age).abs() <= tol]
        if cohort:
            d = d[d.cohort == cohort]
        if split:
            d = d[d.split == split]
        key = {"accurate": ("abs_err", True), "under": ("err", True), "over": ("err", False),
               None: ("age", True)}[kind]
        cols = [c for c in ("split", "recording", "cohort", "age", "pred", "err", "pred_std",
                            "n_windows", "paf", "rel_delta", "rel_alpha") if c in d]
        return d.sort_values(key[0], ascending=key[1])[cols].head(n).reset_index(drop=True)

    def representative_window(self, w: pd.DataFrame) -> int:
        """Position (in ``w``) of the window whose prediction is closest to the
        recording's mean prediction."""
        return int((w.pred - w.pred.mean()).abs().to_numpy().argmin())

    # -- one recording -----------------------------------------------------
    def show(self, recording: str, window: Optional[int] = None, split: Optional[str] = None,
             channels: Optional[Sequence[str]] = None, seconds=(0.0, 10.0), car: bool = True,
             reference: bool = True, clip: Optional[float] = 1.0):
        """EEG of one window (default: the most representative one) + the
        prediction of every window across the recording + posterior spectrum
        against age-matched normal recordings + relative band power."""
        import matplotlib.pyplot as plt
        r = self.record(recording, split)
        w = self.windows_of(recording, r.split)
        pos = self.representative_window(w) if window is None else int(window)
        pick = w.iloc[pos]
        x = self.signal(r.split, pick.idx, car)
        fig = plt.figure(figsize=(16, 5.6), dpi=EEG_DPI)
        gs = fig.add_gridspec(2, 3, width_ratios=[2.2, 1, 1])
        a = fig.add_subplot(gs[:, 0])
        plot_eeg(a, x, self.ch_names, channels, seconds=seconds, clip=clip,
                 title=f"{r.split} {r.recording} ({r.cohort}): true age {r.age:.0f}, recording "
                       f"prediction {r.pred:.1f} | window {int(pick.window)} predicts {pick.pred:.1f}"
                       + ("" if car else "  [raw reference]"))
        b = fig.add_subplot(gs[0, 1:])
        tm = w.window * 10 / 60
        b.plot(tm, w.pred, "o-", ms=3, label="window prediction")
        b.axhline(r.age, color="g", lw=1.8, label=f"true age {r.age:.0f}")
        b.axhline(w.pred.mean(), color="purple", ls="--", lw=1, label=f"mean {w.pred.mean():.1f}")
        b.plot([tm.iloc[pos]], [pick.pred], "o", ms=9, mfc="none", mec="k", label="shown window")
        b.set(xlabel="minutes from recording start", ylabel="age (years)", ylim=AGE_LIM,
              title=f"{len(w)} windows: predictions span {w.pred.min():.0f}–{w.pred.max():.0f} y")
        b.legend(fontsize=7, ncol=4, loc="lower right")
        c = fig.add_subplot(gs[1, 1])
        xs = np.stack([self.signal(r.split, i) for i in w.idx.to_numpy()[:8]])
        freqs, lp = posterior_log_psd(xs, self.ch_names)
        keep = (freqs >= 1) & (freqs <= 30)
        if reference:
            rf, ref, label = self.age_matched_reference(r.age)
            c.fill_between(rf[keep], *np.percentile(ref, [25, 75], axis=0)[:, keep], color="g",
                           alpha=0.2, label=label)
            c.plot(rf[keep], np.median(ref, 0)[keep], color="g", lw=1)
        c.plot(freqs[keep], lp[keep], color="k", label="this recording")
        c.set(xlabel="Hz", ylabel="log10 µV²/Hz", title="posterior spectrum")
        c.legend(fontsize=6)
        d = fig.add_subplot(gs[1, 2])
        rel = band_powers(xs)[0]
        d.bar(list(BAND_COLORS), [rel[n].mean() for n in BAND_COLORS], color=list(BAND_COLORS.values()))
        paf = r.get("paf", np.nan)
        d.set(title=f"relative band power (PAF {paf:.1f} Hz)", ylim=(0, 1))
        fig.tight_layout()
        return fig

    def age_matched_reference(self, age: float, n: int = 25, seed: int = 0):
        """Posterior log-PSDs of up to ``n`` normal recordings near ``age``
        (±5 y, widened when the extremes are sparse)."""
        for tol in (5, 10, 20, 100):
            pool = self.rec[(self.rec.cohort == "normal") & ((self.rec.age - age).abs() <= tol)]
            if len(pool) >= 5:
                break
        pool = pool.sample(min(n, len(pool)), random_state=seed)
        psds = []
        for _, r in pool.iterrows():
            idx = self.windows_of(r.recording, r.split).idx.to_numpy()[:4]
            freqs, lp = posterior_log_psd(np.stack([self.signal(r.split, i) for i in idx]),
                                          self.ch_names)
            psds.append(lp)
        return freqs, np.stack(psds), f"normal, age ±{tol} y (n={len(pool)}), IQR"

    # -- same age, three outcomes ------------------------------------------
    def pick_at_age(self, age: float, tol: float = 3.0, split: Optional[str] = None,
                    cohort: Optional[str] = None) -> Dict[str, pd.Series]:
        """The accurate, most under- and most over-predicted recording near ``age``
        (tolerance widened until three distinct recordings exist)."""
        for t in (tol, 2 * tol, 4 * tol, 100):
            d = self.rec[(self.rec.age - age).abs() <= t]
            if split:
                d = d[d.split == split]
            if cohort:
                d = d[d.cohort == cohort]
            if len(d) >= 3:
                break
        best = d.loc[d.abs_err.idxmin()]
        rest = d.drop(best.name)
        return {"accurate": best, "under-predicted": rest.loc[rest.err.idxmin()],
                "over-predicted": rest.loc[rest.err.idxmax()]}

    def compare_at_age(self, age: float, tol: float = 3.0, split: Optional[str] = None,
                       cohort: Optional[str] = None, channels: Optional[Sequence[str]] = STANDARD_19,
                       seconds=(0.0, 10.0), car: bool = True, clip: Optional[float] = 1.0):
        """Real EEG of three recordings of about the same true age: predicted
        accurately, predicted too young, predicted too old. Same µV scale in all
        three; below, their per-window predictions, spectra and band powers."""
        import matplotlib.pyplot as plt
        picks = self.pick_at_age(age, tol, split, cohort)
        sigs, rows = {}, {}
        for kind, r in picks.items():
            w = self.windows_of(r.recording, r.split)
            rows[kind] = (r, w, w.iloc[self.representative_window(w)])
            sigs[kind] = self.signal(r.split, rows[kind][2].idx, car)
        idx = channel_indices(self.ch_names, channels)
        spacing = eeg_scale(*[s[idx] for s in sigs.values()])
        fig = plt.figure(figsize=(18, 10.5), dpi=EEG_DPI)
        gs = fig.add_gridspec(2, 3, height_ratios=[2.6, 1])
        for k, (kind, (r, w, pick)) in enumerate(rows.items()):
            a = fig.add_subplot(gs[0, k])
            plot_eeg(a, sigs[kind], self.ch_names, channels, spacing=spacing, seconds=seconds,
                     color=KIND_COLORS[kind], label_channels=(k == 0), clip=clip)
            note = ""
            if kind == "over-predicted" and r.err < 0:
                note = " (oldest prediction, still below true age)"
            elif kind == "under-predicted" and r.err > 0:
                note = " (youngest prediction, still above true age)"
            a.set_title(f"{kind.upper()}{note}\n{r.split} {r.recording} ({r.cohort})\n"
                        f"true age {r.age:.0f}  →  predicted {r.pred:.1f}  ({r.err:+.1f} y)",
                        fontsize=10, color=KIND_COLORS[kind], fontweight="bold")
        fig.suptitle(f"Real EEG at true age ≈ {age:.0f}: accurate vs. under- vs. over-predicted "
                     f"(CAR, same {spacing:.0f} µV scale"
                     + (f", clipped at ±{clip:g} trace spacing" if clip is not None else "")
                     + ", window closest to each recording's mean prediction)",
                     fontsize=12, fontweight="bold")
        b = fig.add_subplot(gs[1, 0])
        c = fig.add_subplot(gs[1, 1])
        d = fig.add_subplot(gs[1, 2])
        width = 0.27
        for k, (kind, (r, w, _)) in enumerate(rows.items()):
            col = KIND_COLORS[kind]
            b.plot(w.window * 10 / 60, w.pred, "o-", ms=2.5, lw=1, color=col,
                   label=f"{kind}: true {r.age:.0f}")
            xs = np.stack([self.signal(r.split, i) for i in w.idx.to_numpy()[:8]])
            freqs, lp = posterior_log_psd(xs, self.ch_names)
            keep = (freqs >= 1) & (freqs <= 30)
            c.plot(freqs[keep], lp[keep], color=col, lw=1.5, label=kind)
            rel = band_powers(xs)[0]
            d.bar(np.arange(4) + (k - 1) * width, [rel[n].mean() for n in BAND_COLORS], width,
                  color=col, label=kind)
        b.axhspan(age - tol, age + tol, color="grey", alpha=0.2, label="true age range")
        b.set(xlabel="minutes from recording start", ylabel="predicted age (years)", ylim=AGE_LIM,
              title="per-window predictions")
        b.legend(fontsize=7)
        c.set(xlabel="Hz", ylabel="log10 µV²/Hz", title="posterior spectrum (O1/O2/P3/P4/Pz/T5/T6)")
        c.legend(fontsize=7)
        d.set_xticks(range(4), list(BAND_COLORS))
        d.set(ylim=(0, 1), title="relative band power")
        d.legend(fontsize=7)
        fig.tight_layout()
        return fig

    # -- interactive -------------------------------------------------------
    def widget(self):
        """An ipywidgets browser: pick split / outcome / recording / window /
        channels; the figure redraws on every change."""
        import ipywidgets as W
        import matplotlib.pyplot as plt
        from IPython.display import display

        split = W.Dropdown(options=["val", "test"], value="test", description="split")
        kind = W.Dropdown(options=["over-predicted", "under-predicted", "accurate", "all (by age)"],
                          value="over-predicted", description="show")
        cohort = W.Dropdown(options=["any", "normal", "abnormal"], value="any", description="cohort")
        age = W.IntRangeSlider(value=(0, 90), min=0, max=90, step=1, description="true age",
                               continuous_update=False)
        rec = W.Dropdown(description="recording", layout=W.Layout(width="620px"))
        window = W.IntSlider(min=0, max=29, value=0, description="window #",
                             continuous_update=False)
        rep = W.Checkbox(value=True, description="representative window")
        chans = W.Dropdown(options=list(CHANNEL_SETS), value="10-20 (19)", description="channels")
        car = W.Checkbox(value=True, description="common average reference")
        clip = W.Checkbox(value=True, description="clip artifacts")
        out = W.Output()

        def candidates():
            d = self.rec[self.rec.split == split.value]
            d = d[(d.age >= age.value[0]) & (d.age <= age.value[1])]
            if cohort.value != "any":
                d = d[d.cohort == cohort.value]
            order = {"over-predicted": ("err", False), "under-predicted": ("err", True),
                     "accurate": ("abs_err", True), "all (by age)": ("age", True)}[kind.value]
            return d.sort_values(order[0], ascending=order[1])

        def refresh_list(*_):
            d = candidates()
            rec.options = [(f"{r.recording} | {r.cohort:8s} | true {r.age:4.0f} → pred {r.pred:5.1f} "
                            f"({r.err:+5.1f})", r.recording) for r in d.itertuples()]

        def redraw(*_):
            out.clear_output(wait=True)
            if rec.value is None:
                return
            w = self.windows_of(rec.value, split.value)
            window.max = max(len(w) - 1, 0)
            window.disabled = rep.value
            with out:
                fig = self.show(rec.value, None if rep.value else window.value, split.value,
                                channels=CHANNEL_SETS[chans.value], car=car.value,
                                clip=1.0 if clip.value else None)
                plt.show(fig)

        for wd in (split, kind, cohort, age):
            wd.observe(refresh_list, names="value")
        for wd in (rec, window, rep, chans, car, clip):
            wd.observe(redraw, names="value")
        refresh_list()
        redraw()
        controls = W.VBox([W.HBox([split, kind, cohort]), W.HBox([age, chans]), rec,
                           W.HBox([window, rep, car, clip])])
        display(W.VBox([controls, out]))
