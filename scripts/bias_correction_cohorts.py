#!/usr/bin/env python
"""Age-bias correction of the brain-age models on every cohort, and true-vs-predicted
age figures by model and by correction method.

Calibration (never the scored cohort): the **validation** recordings of the
training cohort with a normal EEG report, adults. For E2L these are its saved
validation predictions (``--val MODEL=json``); for a fold ensemble, each fold
model's predictions on its own validation fold (``--val-windows``, the
``val_bag_weights`` parquets), pooled over the folds. On them
(``labram/eval/age_analysis.py::fit_bias_correction``):

    pred ~ a * age + b            (prediction on age)
    age  ~ c * pred + d           (age on prediction)
    pred - age ~ c0 + c1 * age + c2 * age^2

Corrections applied to a scored cohort (``apply_bias_correction``):

* ``none``          raw prediction;
* ``cole``          (pred - b) / a            -- Cole et al. 2018; needs no true age;
* ``recalibrate``   c * pred + d              -- OLS recalibration; needs no true age;
* ``age_level``     pred - ((a - 1) age + b)  -- linear gap removed (Beheshti et al. 2019;
                                                 de Lange & Cole 2020); uses the true age;
* ``quadratic_gap`` pred - (c0 + c1 age + c2 age^2) -- Gemein et al. 2024; uses the true age.

``cole`` and ``recalibrate`` give a new age estimate, so their MAE is an
accuracy. ``age_level`` and ``quadratic_gap`` use the true age: they give a
corrected brain-age gap for group comparisons, and their "MAE" is the mean
absolute corrected gap, not a deployable accuracy.

``--by-group OUT.csv`` writes the same metrics per subset of each cohort: ``normal``
(a normal EEG report, or the cohort's healthy controls), ``abnormal`` (an
abnormal report, or the cohort's patients) and ``unlabelled`` (no report), next
to ``all``. The correction is the one fitted on the normal-report validation
predictions in every subset; subsets below ``--min-group`` units are left out.

Per cohort, model and method: MAE, mean gap (bias), slope of the prediction on
age (1 = no compression toward the mean), Pearson r of the gap with age (0 = no
age bias), R^2, and the corrected-gap difference pathological - healthy with a
Welch test. Figures: (1) predicted vs true age per cohort for each model,
(2) the same for one model under each correction; curves are means per 5-year
age bin with at least ``--min-bin`` units, points are drawn only for the open
cohorts.

    python scripts/bias_correction_cohorts.py --set ... --models E2L P3ens P4ens \\
        --val E2L=/data/runs/heedb/external_eval/own_all_val_E2L.json \\
        --val-windows P4ens=<val parquets> --val-windows P3ens=<val parquets> \\
        --val-meta /data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1/age_metadata.json \\
        --out docs/data/external_eval/tdbrain/bias_correction.csv --fig-dir docs/figures/tdbrain
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from external_cohort_table import load_meta, units  # noqa: E402
from labram.eval.age_analysis import apply_bias_correction, fit_bias_correction  # noqa: E402

METHODS = ["none", "cole", "recalibrate", "age_level", "quadratic_gap"]
USES_AGE = {"age_level", "quadratic_gap"}
METHOD_LABEL = {"none": "raw", "cole": "Cole inversion", "recalibrate": "recalibration",
                "age_level": "linear gap (uses age)", "quadratic_gap": "quadratic gap (uses age)"}
OPEN_COHORTS = {"lemon", "chbp", "mipdb", "dortmund", "srm"}   # open licences: per-subject points may be drawn
NAMES = {"temple": "Temple (TUH), in-domain", "harvard": "Harvard (HEEDB), in-domain", "tdbrain": "TDBRAIN",
         "lemon": "LEMON", "chbp": "CHBP", "mipdb": "MIPDB", "dortmund": "Dortmund", "srm": "SRM"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def calibration(model, vals, val_windows, val_meta, min_age):
    """(pred, age) of the normal-report adult validation recordings of ``model``."""
    lab = json.load(open(val_meta))["recordings"]
    normal = {k for k, v in lab.items() if v.get("label") == "normal" and (v.get("age") or 0) >= min_age}
    if model in vals:
        res = json.load(open(vals[model]))[model]
        p = pd.Series(res["splits"]["val"]["predictions"], dtype=float)
        p = p[p.index.isin(normal)]
        return p.to_numpy(), np.array([lab[s]["age"] for s in p.index], float), "saved validation predictions"
    import rejection_scores as rs
    preds, ages = [], []
    for path in val_windows[model]:
        f = rs.fold_scores(path)
        f = f[f.index.isin(normal)]
        preds.append(f["yhat"].to_numpy())
        ages.append(np.array([lab[s]["age"] for s in f.index], float))
    return np.concatenate(preds), np.concatenate(ages), f"own validation folds of {len(val_windows[model])} fold model(s)"


def describe(d, corrected):
    age, e = d["age"].to_numpy(float), corrected - d["age"].to_numpy(float)
    sst = ((age - age.mean()) ** 2).sum()
    out = {"n": len(d), "mae": np.abs(e).mean(), "bias": e.mean(), "slope": np.polyfit(age, corrected, 1)[0],
           "r_gap_age": np.corrcoef(e, age)[0, 1], "r2": 1 - (e ** 2).sum() / sst}
    h, p = e[d["healthy"].to_numpy()], e[d["pathological"].to_numpy()]
    if len(h) >= 10 and len(p) >= 10:
        out.update(gap_path_minus_healthy=p.mean() - h.mean(), welch_p=stats.ttest_ind(p, h, equal_var=False).pvalue)
    return out


def describe_subset(d, corrected, mask):
    """MAE, bias, slope, r of the gap with age and R^2 of the units in ``mask``."""
    age, c = d["age"].to_numpy(float)[mask], corrected[mask]
    e = c - age
    sst = ((age - age.mean()) ** 2).sum()
    return {"n": int(mask.sum()), "age_mean": age.mean(), "age_sd": age.std(), "mae": np.abs(e).mean(),
            "bias": e.mean(), "slope": np.polyfit(age, c, 1)[0] if sst > 0 else np.nan,
            "r_gap_age": np.corrcoef(e, age)[0, 1] if sst > 0 else np.nan,
            "r2": 1 - (e ** 2).sum() / sst if sst > 0 else np.nan}


def subsets(d):
    """{name: boolean array} of the report / health subsets present in ``d``."""
    h, p = d["healthy"].to_numpy(bool), d["pathological"].to_numpy(bool)
    out = {"all": np.ones(len(d), bool)}
    if p.any() or (~(h | p)).any():
        out.update({"normal": h, "abnormal": p, "unlabelled": ~(h | p)})
    return {k: v for k, v in out.items() if v.any()}


def binned(age, value, min_bin, width=5.0):
    """Mean and SD of ``value`` per ``width``-year age bin with at least ``min_bin`` units."""
    edges = np.arange(np.floor(age.min() / width) * width, age.max() + width, width)
    idx = np.digitize(age, edges)
    rows = [(age[idx == k].mean(), value[idx == k].mean(), value[idx == k].std()) for k in np.unique(idx)
            if (idx == k).sum() >= min_bin]
    return np.array(rows).T if rows else np.empty((3, 0))


def panel_grid(n):
    import matplotlib.pyplot as plt
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(12.5, 4.6 * rows), squeeze=False)
    return fig, axes


def style(ax, title, lo=15, hi=90):
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls=":", zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("true age (y)")
    ax.set_ylabel("predicted age (y)")
    ax.set_title(title, fontsize=11, color=INK)
    ax.grid(color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", action="append", required=True, metavar="NAME:COHORT:EVAL.json:WINDOWS_DIR[:FAMILY]")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--val", action="append", default=[], metavar="MODEL=JSON")
    ap.add_argument("--val-windows", action="append", default=[], metavar="MODEL=PARQUET[,PARQUET...]")
    ap.add_argument("--val-meta", required=True)
    ap.add_argument("--min-age", type=float, default=18.0)
    ap.add_argument("--min-bin", type=int, default=10)
    ap.add_argument("--figure-model", default=None, help="model of the correction figure (default: the last of --models)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--by-group", default=None, help="also write the metrics per normal / abnormal / unlabelled subset")
    ap.add_argument("--min-group", type=int, default=20)
    ap.add_argument("--fig-dir", default=None)
    args = ap.parse_args()
    vals = dict(v.split("=", 1) for v in args.val)
    val_windows = {k: v.split(",") for k, v in (e.split("=", 1) for e in args.val_windows)}
    params, rows, data, grouped = {}, [], {}, []
    for model in args.models:
        pred, age, source = calibration(model, vals, val_windows, args.val_meta, args.min_age)
        params[model] = fit_bias_correction(pred, age)
        print(f"{model}: calibration on {len(pred):,} normal-report validation predictions ({source}); "
              f"pred = {params[model]['slope']:.3f} age + {params[model]['intercept']:.2f}")
        rows.append({"cohort": "calibration (val, normal)", "model": model, "method": "none", "n": len(pred),
                     "mae": np.abs(pred - age).mean(), "bias": (pred - age).mean(), "slope": params[model]["slope"],
                     "r_gap_age": np.corrcoef(pred - age, age)[0, 1]})
    for spec in args.set:
        name, cdir, ev, _wdir, *family = spec.split(":")
        meta = load_meta(cdir, family[0] if family else None, args.min_age)
        res = json.load(open(ev))
        for model in args.models:
            d = units(meta, pd.Series(res[model]["splits"]["test"]["predictions"], dtype=float))
            data[(name, model)] = d
            for method in METHODS:
                c = apply_bias_correction(d["pred"].to_numpy(), params[model], method, age=d["age"].to_numpy())
                rows.append({"cohort": name, "model": model, "method": method, "uses_true_age": method in USES_AGE,
                             **describe(d, c)})
                for group, mask in subsets(d).items():
                    if mask.sum() >= args.min_group:
                        grouped.append({"cohort": name, "model": model, "method": method, "group": group,
                                        "uses_true_age": method in USES_AGE, **describe_subset(d, c, mask)})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(df.round(3).to_string(index=False))
    if args.by_group:
        pd.DataFrame(grouped).to_csv(args.by_group, index=False)
        print(args.by_group)
    if not args.fig_dir:
        return
    import matplotlib
    matplotlib.use("Agg")
    from labram.eval.age_plots import assert_max_columns, panel_labels
    os.makedirs(args.fig_dir, exist_ok=True)
    cohorts = [s.split(":")[0] for s in args.set]
    # Figure 1: models
    fig, axes = panel_grid(len(cohorts))
    for ax, c in zip(axes.flat, cohorts):
        last = data[(c, args.models[-1])]
        if c in OPEN_COHORTS:
            ax.scatter(last["age"], last["pred"], s=9, color=MUTED, alpha=0.35, zorder=2, linewidths=0)
        for model, color in zip(args.models, PALETTE):
            d = data[(c, model)]
            x, m, s = binned(d["age"].to_numpy(), d["pred"].to_numpy(), args.min_bin)
            mae = np.abs(d["pred"] - d["age"]).mean()
            ax.plot(x, m, color=color, lw=2, marker="o", ms=4, zorder=4, label=f"{model} (MAE {mae:.2f})")
            if model == args.models[-1]:
                ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, zorder=3, linewidth=0)
        style(ax, f"{NAMES.get(c, c)} (n = {len(last):,})")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    for ax in axes.flat[len(cohorts):]:
        ax.set_visible(False)
    fig.tight_layout()
    panel_labels(list(axes.flat[:len(cohorts)]))
    assert_max_columns(fig)
    path = os.path.join(args.fig_dir, "pred_vs_age_models.png")
    fig.savefig(path, dpi=140)
    print(path)
    # Figure 2: corrections of one model
    model = args.figure_model or args.models[-1]
    fig, axes = panel_grid(len(cohorts))
    for ax, c in zip(axes.flat, cohorts):
        d = data[(c, model)]
        for method, color in zip(METHODS, PALETTE):
            corr = apply_bias_correction(d["pred"].to_numpy(), params[model], method, age=d["age"].to_numpy())
            x, m, _ = binned(d["age"].to_numpy(), corr, args.min_bin)
            mae = np.abs(corr - d["age"]).mean()
            ax.plot(x, m, color=color, lw=2, marker="o", ms=4, zorder=4, ls="--" if method in USES_AGE else "-",
                    label=f"{METHOD_LABEL[method]} ({mae:.2f})")
        style(ax, f"{NAMES.get(c, c)}: {model}")
        ax.legend(frameon=False, fontsize=8, loc="upper left", title="correction (MAE or mean |gap|)", title_fontsize=8)
    for ax in axes.flat[len(cohorts):]:
        ax.set_visible(False)
    fig.tight_layout()
    panel_labels(list(axes.flat[:len(cohorts)]))
    assert_max_columns(fig)
    path = os.path.join(args.fig_dir, "pred_vs_age_corrections.png")
    fig.savefig(path, dpi=140)
    print(path)


if __name__ == "__main__":
    main()
