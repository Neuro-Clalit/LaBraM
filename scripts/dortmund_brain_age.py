#!/usr/bin/env python
"""Zero-shot brain age on the Dortmund Vital Study cohort (OpenNeuro ds005385).

The cohort has four 3-minute runs per session (eyes closed / eyes open, before
/ after a 2-hour block of cognitive tasks) and, for 208 of the 608 healthy
participants, a second session about 5 years later. From an
``eval_age_by_cohort.py`` output (``--eval``) and the cohort manifest this
script writes three tables to ``--out-dir``:

* ``dortmund_accuracy.csv`` -- MAE with a 95 % subject-bootstrap interval, bias,
  r, R^2 per model, scoring unit and subset. Units: the **session** (mean of its
  runs' predictions; all four runs, or only the pre-task / post-task / eyes-closed
  / eyes-open ones), sessions 1 and 2 apart, the single **run** by condition, the
  session restricted to runs without a recording flag (a discontinuity, stray
  triggers or a lost-connection note), and the session after the label-free
  rejection of the ``--reject`` least certain sessions (``rejection_scores.py``:
  ``total`` for an ensemble, ``spread`` for a single model; needs ``--windows``).
  Session subsets: age bands.
* ``dortmund_state_effects.csv`` -- paired differences of the predicted age inside
  a session: eyes open - eyes closed and after - before the task block (mean, SD,
  95 % interval over subjects).
* ``dortmund_longitudinal.csv`` -- for participants with both sessions: the
  predicted change ``pred_2 - pred_1`` against the elapsed time (difference of
  the recording years), the correlation of the two sessions' gaps (is the gap a
  stable property of the person?), and the same within session 1 between the
  pre- and the post-task runs (short-term repeatability).

Ages only score: nothing is fitted on this cohort. Figures (``--fig-dir``):
predicted against true age per model, and the longitudinal change.

    python scripts/dortmund_brain_age.py --eval .../dortmund_all_models.json \\
        --cohort /data/datasets/EEG-public/DORTMUND/cohorts/dortmund_v1 \\
        --windows .../windows/dortmund_all --models E2L P3ens P4ens \\
        --out-dir docs/data/external_eval/dortmund --fig-dir docs/figures/dortmund
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tdbrain_brain_age import metrics  # noqa: E402

BANDS = [("20-29", 20, 30), ("30-39", 30, 40), ("40-49", 40, 50), ("50-59", 50, 60), ("60+", 60, 200)]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def load_runs(cohort, pred):
    """One row per scored run: manifest fields + prediction."""
    meta = pd.DataFrame(json.load(open(os.path.join(cohort, "manifest.json")))["recordings"]).set_index("stem")
    meta = meta[meta["age_session_matched"].astype(bool)]
    d = meta.join(pd.Series(pred, dtype=float).rename("pred"), how="inner")
    d["flagged"] = (d["n_boundaries"] > 1) | (d["n_triggers"] > 0) | d["usb_note"].astype(bool)
    return d


def sessions(runs):
    """Session units: mean prediction of the given runs of each (subject, session)."""
    return runs.groupby(["subject", "session"]).agg(pred=("pred", "mean"), age=("age", "first"),
                                                    n_runs=("pred", "size")).reset_index()


def mean_ci(x, subjects, n_boot=2000, seed=0):
    """Mean with a 95 % bootstrap interval over subjects."""
    x, subjects = np.asarray(x, float), np.asarray(subjects)
    uniq, inv = np.unique(subjects, return_inverse=True)
    total, count = np.bincount(inv, x), np.bincount(inv)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        boot.append((w * total).sum() / (w * count).sum())
    return x.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def icc_agreement(a, b):
    """Two-way random, absolute-agreement, single-measure ICC(2,1) of two measurements per subject."""
    y = np.column_stack([a, b]).astype(float)
    n, k = y.shape
    grand = y.mean()
    msr = k * ((y.mean(1) - grand) ** 2).sum() / (n - 1)
    msc = n * ((y.mean(0) - grand) ** 2).sum() / (k - 1)
    mse = ((y - y.mean(1, keepdims=True) - y.mean(0, keepdims=True) + grand) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def accuracy_rows(model, runs, scores, reject):
    rows = []

    def add(unit, subset, d, **extra):
        if len(d) >= 10:
            rows.append({"model": model, "unit": unit, "subset": subset, **metrics(d), **extra})

    ses = sessions(runs)
    add("session, all runs", "all", ses)
    for name, lo, hi in BANDS:
        add("session, all runs", f"age {name}", ses[(ses.age >= lo) & (ses.age < hi)])
    for s in sorted(runs["session"].unique()):
        add("session, all runs", f"session {s}", ses[ses.session == s])
    for label, sel in (("pre-task runs", runs.acq == "pre"), ("post-task runs", runs.acq == "post"),
                       ("eyes-closed runs", runs.task == "EC"), ("eyes-open runs", runs.task == "EO")):
        add(f"session, {label}", "all", sessions(runs[sel]))
    add("session, runs without a recording flag", "all", sessions(runs[~runs.flagged]),
        runs_dropped=int(runs.flagged.sum()))
    add("run", "all", runs)
    for (task, acq), g in runs.groupby(["task", "acq"]):
        add("run", f"{task}, {acq}-task", g)
    if scores is not None:
        col = "total" if scores["fold_sd"].abs().sum() > 0 else "spread"
        s = ses.merge(scores[["subject", "session", col]], on=["subject", "session"], how="inner")
        keep = s[s[col] <= s[col].quantile(1 - reject)]
        add(f"session, reject {reject:.0%} by {col} (label-free)", "all", keep, kept_share=len(keep) / len(s))
    return rows


def state_rows(model, runs):
    """Paired prediction differences inside a session."""
    wide = runs.pivot_table(index=["subject", "session"], columns=["task", "acq"], values="pred")
    subj = wide.index.get_level_values("subject")
    rows = []
    pairs = [("eyes open - eyes closed, pre-task", ("EO", "pre"), ("EC", "pre")),
             ("eyes open - eyes closed, post-task", ("EO", "post"), ("EC", "post")),
             ("post - pre, eyes closed", ("EC", "post"), ("EC", "pre")),
             ("post - pre, eyes open", ("EO", "post"), ("EO", "pre"))]
    for name, a, b in pairs:
        if a not in wide or b not in wide:
            continue
        diff = (wide[a] - wide[b]).dropna()
        m, lo, hi = mean_ci(diff, subj[wide.index.isin(diff.index)])
        rows.append({"model": model, "contrast": name, "n": len(diff), "mean": m, "lo": lo, "hi": hi,
                     "sd": diff.std(), "r_between_the_two": np.corrcoef(wide.loc[diff.index, a], wide.loc[diff.index, b])[0, 1]})
    return rows


def longitudinal_rows(model, runs):
    ses = sessions(runs)
    w = ses.pivot(index="subject", columns="session", values=["pred", "age"]).dropna()
    rows = []
    if len(w) >= 10 and ("pred", "2") in w:
        d_pred, d_age = w[("pred", "2")] - w[("pred", "1")], w[("age", "2")] - w[("age", "1")]
        g1, g2 = w[("pred", "1")] - w[("age", "1")], w[("pred", "2")] - w[("age", "2")]
        m, lo, hi = mean_ci(d_pred, w.index)
        rows.append({"model": model, "comparison": "session 2 vs session 1", "n": len(w),
                     "elapsed_years_mean": d_age.mean(), "pred_change_mean": m, "pred_change_lo": lo,
                     "pred_change_hi": hi, "pred_change_sd": d_pred.std(), "share_older": (d_pred > 0).mean(),
                     "r_pred": np.corrcoef(w[("pred", "1")], w[("pred", "2")])[0, 1],
                     "r_gap": np.corrcoef(g1, g2)[0, 1], "icc_gap": icc_agreement(g1, g2),
                     "gap_sd_session1": g1.std(), "gap_sd_session2": g2.std()})
    s1 = runs[runs.session == "1"]
    pre, post = sessions(s1[s1.acq == "pre"]).set_index("subject"), sessions(s1[s1.acq == "post"]).set_index("subject")
    both = pre.join(post, lsuffix="_pre", rsuffix="_post", how="inner")
    if len(both) >= 10:
        g1, g2 = both.pred_pre - both.age_pre, both.pred_post - both.age_post
        m, lo, hi = mean_ci(both.pred_post - both.pred_pre, both.index)
        rows.append({"model": model, "comparison": "session 1: post-task vs pre-task runs (2 h apart)", "n": len(both),
                     "elapsed_years_mean": 0.0, "pred_change_mean": m, "pred_change_lo": lo, "pred_change_hi": hi,
                     "pred_change_sd": (both.pred_post - both.pred_pre).std(),
                     "share_older": ((both.pred_post - both.pred_pre) > 0).mean(),
                     "r_pred": np.corrcoef(both.pred_pre, both.pred_post)[0, 1],
                     "r_gap": np.corrcoef(g1, g2)[0, 1], "icc_gap": icc_agreement(g1, g2),
                     "gap_sd_session1": g1.std(), "gap_sd_session2": g2.std()})
    return rows, w


def binned(age, value, width=5.0, min_bin=10):
    edges = np.arange(np.floor(age.min() / width) * width, age.max() + width, width)
    idx = np.digitize(age, edges)
    rows = [(age[idx == k].mean(), value[idx == k].mean()) for k in np.unique(idx) if (idx == k).sum() >= min_bin]
    return np.array(rows).T if rows else np.empty((2, 0))


def style(ax, title):
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.grid(color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def figures(per_model, fig_dir, models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)
    show = [m for m in models if m in per_model][:1] + [m for m in models if m in per_model][-1:]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.0))
    for ax, model, tag in zip(axes, show, "ab"):
        ses = sessions(per_model[model]["runs"])
        s1 = ses[ses.session == "1"]
        ax.plot([15, 80], [15, 80], color=MUTED, lw=1, ls=":", zorder=1)
        ax.scatter(s1.age, s1.pred, s=9, color=BLUE, alpha=0.35, lw=0, zorder=2, label="session 1, one point per person")
        x, y = binned(s1.age.to_numpy(), s1.pred.to_numpy())
        ax.plot(x, y, color=ORANGE, lw=2.2, marker="o", ms=5, zorder=3, label="mean per 5-year age bin")
        mae = (s1.pred - s1.age).abs().mean()
        style(ax, f"({tag}) {model}: MAE {mae:.2f} y, r {np.corrcoef(s1.pred, s1.age)[0, 1]:.2f} (n = {len(s1)})")
        ax.set_xlim(15, 80)
        ax.set_ylim(15, 80)
        ax.set_aspect("equal")
        ax.set_xlabel("true age (y)")
        ax.set_ylabel("predicted age (y)")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "dortmund_pred_vs_age.png"), dpi=170)
    print(os.path.join(fig_dir, "dortmund_pred_vs_age.png"))

    model = show[-1]
    w = per_model[model]["wide"]
    if len(w) >= 10:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
        ax = axes[0]
        ax.plot([15, 85], [15, 85], color=MUTED, lw=1, ls=":", zorder=1)
        ax.scatter(w[("pred", "1")], w[("pred", "2")], s=14, color=BLUE, alpha=0.5, lw=0, zorder=2)
        style(ax, f"(a) {model}: predicted age at the two sessions (n = {len(w)})")
        ax.set_xlabel("predicted age, session 1 (y)")
        ax.set_ylabel("predicted age, session 2 (y)")
        ax.set_xlim(15, 85)
        ax.set_ylim(15, 85)
        ax.set_aspect("equal")
        d_pred, d_age = w[("pred", "2")] - w[("pred", "1")], w[("age", "2")] - w[("age", "1")]
        ax = axes[1]
        ax.hist(d_pred, bins=np.arange(-20, 30.01, 2), color=BLUE, alpha=0.85, zorder=2, edgecolor="white")
        ax.axvline(d_pred.mean(), color=ORANGE, lw=2, zorder=3, label=f"mean predicted change {d_pred.mean():+.1f} y")
        ax.axvline(d_age.mean(), color=INK, lw=1.5, ls="--", zorder=3, label=f"mean elapsed time {d_age.mean():.1f} y")
        style(ax, "(b) change of the predicted age between the sessions")
        ax.set_xlabel("predicted age at session 2 minus session 1 (y)")
        ax.set_ylabel("participants")
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "dortmund_longitudinal.png"), dpi=170)
        print(os.path.join(fig_dir, "dortmund_longitudinal.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--windows", default=None, help="dir of the --save-windows parquets, for the rejection rows")
    ap.add_argument("--reject", type=float, default=0.2)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fig-dir", default="")
    args = ap.parse_args()
    res = json.load(open(args.eval))
    acc, state, longi, per_model = [], [], [], {}
    for model in args.models:
        runs = load_runs(args.cohort, res[model]["splits"]["test"]["predictions"])
        scores = None
        members = res[model].get("members", [model])
        if args.windows and all(os.path.exists(os.path.join(args.windows, f"{m}_test.parquet")) for m in members):
            import rejection_scores as rs
            meta = runs.assign(pathological=False, healthy=True)[["subject", "session", "task", "age", "healthy", "pathological"]]
            scores = rs.cohort_units(meta, args.windows, members)
        acc += accuracy_rows(model, runs, scores, args.reject)
        state += state_rows(model, runs)
        rows, wide = longitudinal_rows(model, runs)
        longi += rows
        per_model[model] = {"runs": runs, "wide": wide}
    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in (("accuracy", acc), ("state_effects", state), ("longitudinal", longi)):
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(args.out_dir, f"dortmund_{name}.csv"), index=False)
        with pd.option_context("display.width", 260, "display.max_columns", 30, "display.max_rows", 500):
            print(df.round(3).to_string(index=False))
    if args.fig_dir:
        figures(per_model, args.fig_dir, args.models)


if __name__ == "__main__":
    main()
