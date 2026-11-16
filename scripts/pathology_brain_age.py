#!/usr/bin/env python
"""Brain-age accuracy and brain-age gap per pathology group and EEG-report status
on the training cohort's held-out data (Temple + Harvard, ``tuh_heedb_v1``).

Two sources of predictions, both for recordings the scoring model never trained on:

* ``test``  the shared test split, any model of an ``eval_age_by_cohort.py``
            output (``--test-eval``, ``--models``; ensembles included);
* ``oof``   out-of-fold: each fold model's predictions on its own validation
            fold at its best epoch (``--oof NAME=parquet,...``, the
            ``val_bag_weights`` files), pooled over the folds. One model per
            recording, no ensemble; the fold was used to pick that model's best
            epoch, nothing else. It covers about five times more recordings
            than the test split, which the rarer groups need (the Temple test
            split has 272 recordings).

Groups (adults, ``--min-age``):

* report status: ``normal`` / ``abnormal`` EEG report, ``unlabelled`` (no report);
  labels the cohort table leaves empty (the TUH rows) come from ``age_metadata.json``;
* ``reference``: a normal report **and** a health record with none of the
  pathology groups (the comparison group of every adjusted difference);
* each pathology group of ``multitask_targets_v2.npz`` (NEDC sheet groups for
  Temple, ICD-10 groups for Harvard; a patient can be in several), all its
  recordings and those with a normal / an abnormal report.

Per group: recordings, subjects, mean age, MAE with a 95 % subject-bootstrap
interval, mean gap (prediction - age), and against the reference group

    y_r = b1 * 1[r in group] + c_band(r) + b2 * age_r + b3 * male_r + e_r

fitted on group + reference, with one intercept ``c`` per 5-year age band (85+
is one band), once with ``y`` = gap (``gap_adj`` = b1: how much older the group
looks at the same age and sex) and once with ``y`` = |gap| (``abs_adj``: its
extra absolute error). The age bands allow for the non-linear age bias of the
models (predictions are compressed toward the mean age, so the gap falls with
age). ``b1`` is an average over the group's ages: restricted to ages 60+ the
estimates of the oldest groups (dementia, stroke) are about half as large.
Standard errors are clustered by
subject (several recordings per patient); ``q`` is the Benjamini-Hochberg
adjusted p-value over the pathology groups of one source, model and dataset.
Cells with fewer than ``--min-subjects`` subjects or ``--min-n`` recordings are
suppressed (data use agreements: aggregates only).

    python scripts/pathology_brain_age.py --models E2L P3ens P4ens \\
        --oof P3=<fold parquets> --out docs/data/external_eval/tdbrain/pathology_brain_age.csv \\
        --figure docs/figures/tdbrain/pathology_gap.png
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tdbrain_brain_age import metrics  # noqa: E402

COHORT = "/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1"
LABEL = {"clinical_seizure": "clinical seizures", "electrographic_seizure": "electrographic seizures",
         "epilepsy": "epilepsy", "stroke": "stroke", "tbi": "traumatic brain injury", "substance": "substance use",
         "bipolar": "bipolar disorder", "anxiety": "anxiety", "depression": "depression",
         "schizophrenia": "schizophrenia", "dementia": "dementia", "ms": "multiple sclerosis",
         "other": "other diagnosis", "headache_disorders": "headache / migraine", "sleep_disorders": "sleep disorders",
         "intracranial_and_spinal_tumors": "CNS tumours", "movement_and_cerebellar_disorders": "movement disorders"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
AGE_BAND = 5.0          # years; the adjustment compares a group with the reference inside age bands


def load_cohort(root, targets, min_age):
    """Adult recordings: subject, age, sex, family, report label and one 0/1/NaN column per pathology group."""
    d = pd.read_parquet(os.path.join(root, "cohort.parquet")).set_index("stem")
    side = json.load(open(os.path.join(root, "age_metadata.json")))["recordings"]   # the TUH rows' report labels
    d["label"] = d["label"].astype(object).combine_first(pd.Series({k: v.get("label") for k, v in side.items()}, dtype=object))
    z = np.load(os.path.join(root, targets))
    t = pd.DataFrame(z["targets"], index=z["stems"], columns=[str(n).split("/")[-1] for n in z["names"]])
    groups = [c for c in t.columns if c != "status_abnormal"]
    d = d.join(t[groups])
    d["family"] = np.where(d["corpus"] == "heedb", "harvard", "temple")
    d["label"] = d["label"].astype(object).where(d["label"].notna(), "unlabelled")
    d["has_record"] = d[groups].notna().any(axis=1)
    d["any_pathology"] = (d[groups] == 1).any(axis=1)
    d["reference"] = (d["label"] == "normal") & d["has_record"] & ~d["any_pathology"]
    return d[d["age"] >= min_age], groups


def cluster_ols(y, X, clusters):
    """OLS coefficients, cluster-robust (CR1) standard errors and the number of clusters."""
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    u = y - X @ beta
    _, inv = np.unique(clusters, return_inverse=True)
    g = inv.max() + 1
    s = np.zeros((g, X.shape[1]))
    np.add.at(s, inv, X * u[:, None])
    n, k = X.shape
    cov = xtx_inv @ (s.T @ s) @ xtx_inv * (g / (g - 1)) * ((n - 1) / (n - k))
    return beta, np.sqrt(np.diag(cov)), g


def adjusted(d, in_group, in_ref):
    """b1 of the docstring model for the gap and for |gap|, with 95 % intervals and p."""
    x = d[(in_group | in_ref) & d["sex"].isin(["M", "F"])]
    flag = in_group.loc[x.index].to_numpy(float)
    if flag.sum() < 10 or (1 - flag).sum() < 10:
        return {}
    age = x["age"].to_numpy(float)
    band = np.unique(np.minimum(age, 85.0) // AGE_BAND, return_inverse=True)[1]     # 85+ is one band
    X = np.column_stack([flag, np.eye(band.max() + 1)[band], (age - age.mean()) / 10.0,
                         (x["sex"] == "M").to_numpy(float)])
    out = {}
    for name, y in (("gap", x["gap"].to_numpy(float)), ("abs", x["gap"].abs().to_numpy(float))):
        beta, se, g = cluster_ols(y, X, x["subject"].to_numpy())
        t = beta[0] / se[0]
        out.update({f"{name}_adj": beta[0], f"{name}_adj_lo": beta[0] - 1.96 * se[0],
                    f"{name}_adj_hi": beta[0] + 1.96 * se[0],
                    f"{name}_adj_p": 2 * stats.t.sf(abs(t), g - 1)})
    return out


def bh(p):
    """Benjamini-Hochberg adjusted p-values (NaN kept)."""
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    if len(ok):
        order = ok[np.argsort(p[ok])]
        adj = p[order] * len(ok) / np.arange(1, len(ok) + 1)
        q[order] = np.minimum(np.minimum.accumulate(adj[::-1])[::-1], 1.0)
    return q


def table(d, groups, min_n, min_subjects):
    """Rows of one (source, model, family) table."""
    rows, ref = [], d["reference"]
    sets = [("report", "normal", d["label"] == "normal"), ("report", "abnormal", d["label"] == "abnormal"),
            ("report", "unlabelled", d["label"] == "unlabelled"), ("report", "reference", ref)]
    for g in groups:
        pos = d[g] == 1
        sets += [("pathology", g, pos), ("pathology_normal", g, pos & (d["label"] == "normal")),
                 ("pathology_abnormal", g, pos & (d["label"] == "abnormal"))]
    for kind, name, mask in sets:
        x = d[mask]
        row = {"kind": kind, "group": name, "n": len(x), "n_subjects": x["subject"].nunique()}
        if row["n"] < min_n or row["n_subjects"] < min_subjects:
            if row["n"]:                                    # small cell: suppressed, not reported
                rows.append({"kind": kind, "group": name, "n": np.nan, "n_subjects": np.nan, "suppressed": True})
            continue
        m = metrics(x.assign(pred=x["pred"]), n_boot=1000)
        row.update({k: m[k] for k in ("age_mean", "mae", "mae_lo", "mae_hi", "bias", "r2")})
        if not (kind == "report" and name in ("reference", "normal")):
            row.update(adjusted(d, mask & ~ref, ref))
        rows.append(row)
    out = pd.DataFrame(rows)
    if "gap_adj_p" in out:
        sel = (out["kind"] == "pathology").to_numpy()
        for name in ("gap", "abs"):
            out.loc[sel, f"{name}_adj_q"] = bh(out.loc[sel, f"{name}_adj_p"])
    return out


def forest(ax, t, title, color):
    """Adjusted gap per pathology group (rows sorted by the estimate) with 95 % intervals."""
    t = t[(t["kind"] == "pathology") & t["gap_adj"].notna()].sort_values("gap_adj")
    y = np.arange(len(t))
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    ax.hlines(y, t["gap_adj_lo"], t["gap_adj_hi"], color=color, lw=2, zorder=2)
    sig = (t["gap_adj_q"] < 0.05).to_numpy()
    ax.scatter(t["gap_adj"][sig], y[sig], s=46, color=color, zorder=3, label="q < 0.05")
    ax.scatter(t["gap_adj"][~sig], y[~sig], s=46, facecolor="white", edgecolor=color, lw=1.6, zorder=3, label="n.s.")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{LABEL.get(g, g)} ({int(n):,})" for g, n in zip(t["group"], t["n"])], fontsize=9)
    ax.set_xlabel("brain-age gap vs reference, adjusted (years)")
    ax.set_title(title, fontsize=10.5, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cohort", default=COHORT)
    ap.add_argument("--targets", default="multitask_targets_v2.npz")
    ap.add_argument("--test-eval", default="/data/runs/heedb/external_eval/own_5min_models.json")
    ap.add_argument("--models", nargs="*", default=["E2L", "P3ens", "P4ens"])
    ap.add_argument("--oof", action="append", default=[], metavar="NAME=PARQUET[,PARQUET...]")
    ap.add_argument("--min-age", type=float, default=18.0)
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--min-subjects", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figure", default="")
    ap.add_argument("--figure-panels", nargs=2, default=["oof:P3:temple", "oof:P3:harvard"],
                    metavar="SOURCE:MODEL:FAMILY")
    args = ap.parse_args()
    cohort, groups = load_cohort(args.cohort, args.targets, args.min_age)
    preds = {}
    if args.models:
        res = json.load(open(args.test_eval))
        for m in args.models:
            preds[("test", m)] = pd.Series(res[m]["splits"]["test"]["predictions"], dtype=float)
    if args.oof:
        import rejection_scores as rs
        for name, paths in (e.split("=", 1) for e in args.oof):
            p = pd.concat([rs.fold_scores(f)["yhat"] for f in paths.split(",")])
            if p.index.has_duplicates:
                raise SystemExit(f"--oof {name}: a recording is in two validation folds")
            preds[("oof", name)] = p
    out = []
    for (source, model), p in preds.items():
        d = cohort.join(p.rename("pred"), how="inner")
        d["gap"] = d["pred"] - d["age"]
        for family in ("temple", "harvard"):
            t = table(d[d["family"] == family], groups, args.min_n, args.min_subjects)
            out.append(t.assign(source=source, model=model, family=family))
    df = pd.concat(out, ignore_index=True)
    lead = ["source", "model", "family", "kind", "group"]
    df = df[lead + [c for c in df.columns if c not in lead]]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    show = ["source", "model", "family", "kind", "group", "n", "n_subjects", "age_mean", "mae", "bias",
            "gap_adj", "gap_adj_lo", "gap_adj_hi", "gap_adj_q", "abs_adj", "abs_adj_q"]
    with pd.option_context("display.width", 260, "display.max_rows", 2000):
        print(df[df["kind"].isin(["report", "pathology"])][[c for c in show if c in df]].round(2).to_string(index=False))
    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
        names = {"temple": "Temple", "harvard": "Harvard"}
        for ax, spec, color, tag in zip(axes, args.figure_panels, (BLUE, ORANGE), "ab"):
            source, model, family = spec.split(":")
            t = df[(df["source"] == source) & (df["model"] == model) & (df["family"] == family)]
            what = "out-of-fold" if source == "oof" else "test split"
            forest(ax, t, f"({tag}) {names[family]}, {what}, {model} (recordings)", color)
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(args.figure)), exist_ok=True)
        fig.savefig(args.figure, dpi=170)
        print(args.figure)


if __name__ == "__main__":
    main()
