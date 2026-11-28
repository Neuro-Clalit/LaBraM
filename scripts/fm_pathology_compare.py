#!/usr/bin/env python
"""Brain age per pathology group for several models on the same recordings, paired.

The comparison of ``docs/fm_encoder_summary.md`` (P4 / LaBraM against the fine-tuned CBraMod and
REVE models): every model is scored on the same held-out recordings of the training cohort
(Temple + Harvard, ``tuh_heedb_v1``), adults only, split by source dataset and by group.

Groups (``scripts/pathology_brain_age.py::load_cohort``): the EEG-report status (normal, abnormal,
unlabelled), the reference group (normal report and a health record with none of the pathology
groups) and each pathology group of ``multitask_targets_v2.npz`` (NEDC sheet groups for Temple,
ICD-10 groups for Harvard; a patient can be in several).

Per group and model: the MAE (Eq. 5.1 of the summary), the mean gap (prediction - age), the
gap adjusted for age band and sex against the reference group (``pathology_brain_age.adjusted``:
how much older the group looks at the same age and sex, subject-clustered errors), and that
adjusted gap in units of the reference group's spread (``gapadj_sd_<model>``: divided by the
standard deviation of the reference group's gap around its age-band means), which puts models
with different accuracy on one scale. Per group and
model other than ``--reference``: the paired MAE difference against the reference model on the
same recordings, with a 95 % interval from a bootstrap over subjects (a patient's recordings are
resampled together). Cells with fewer than ``--min-n`` recordings or ``--min-subjects`` subjects
are suppressed (data use agreements: aggregates only).

Predictions: ``--pred MODEL:SOURCE=PATH``, PATH a bag-weight parquet of a MIL run
(``analysis/<split>_bag_weights_epochNN.parquet``) or an ``eval_age_by_cohort.py`` output as
``file.json#RUN`` (its ``SOURCE`` split). Sources are scored alone and pooled (``pooled``: all
sources of a model together; a recording must not be in two sources).

    python scripts/fm_pathology_compare.py --reference P4 \\
        --pred P4:val=p4_val.json#P4_L1_f0 --pred P4:test=p4_test.json#P4_L1_f0 \\
        --pred REVE:val=<run>/analysis/val_bag_weights_epoch10.parquet ... --out pathology.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_bag_mae import recording_predictions  # noqa: E402
from pathology_brain_age import COHORT, LABEL, adjusted, bh, load_cohort  # noqa: E402


def read_predictions(path, source):
    """recording -> predicted age (years)."""
    if "#" in path:
        file, run = path.split("#", 1)
        return pd.Series(json.load(open(file))[run]["splits"][source]["predictions"], dtype=float)
    return recording_predictions(path)["pred"]


def paired_delta(err_a, err_b, subjects, n_boot=2000, seed=0):
    """MAE(a) - MAE(b) on the same recordings, with a 95 % interval from a bootstrap over subjects."""
    d = np.abs(err_a) - np.abs(err_b)
    _, inv = np.unique(subjects, return_inverse=True)
    s = np.bincount(inv, weights=d)
    n = np.bincount(inv).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(s), size=(n_boot, len(s)))
    boot = s[idx].sum(1) / n[idx].sum(1)
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def reference_spread(d, model, band_years=5.0):
    """SD of the reference group's gap around its age-band means (85+ is one band), years."""
    x = d[d["reference"]]
    gap = x[f"pred_{model}"] - x["age"]
    band = np.minimum(x["age"], 85.0) // band_years
    return float((gap - gap.groupby(band).transform("mean")).std())


def group_sets(d, groups):
    sets = [("report", g, d["label"] == g) for g in ("normal", "abnormal", "unlabelled")]
    sets.append(("report", "reference", d["reference"]))
    sets += [("pathology", g, d[g] == 1) for g in groups]
    return sets


def table(d, groups, models, reference, min_n, min_subjects):
    """One row per group of one (source, family) frame ``d`` holding ``pred_<model>``."""
    rows = []
    spread = {m: reference_spread(d, m) for m in models}
    for kind, name, mask in group_sets(d, groups):
        x = d[mask]
        row = {"kind": kind, "group": name, "label": LABEL.get(name, name)}
        if len(x) < min_n or x["subject"].nunique() < min_subjects:
            if len(x):
                rows.append({**row, "suppressed": True})
            continue
        row.update(n=len(x), n_subjects=x["subject"].nunique(), age_mean=x["age"].mean(), suppressed=False)
        for m in models:
            gap = x[f"pred_{m}"] - x["age"]
            row[f"mae_{m}"] = gap.abs().mean()
            row[f"bias_{m}"] = gap.mean()
            if m != reference:
                row[f"d_{m}"], row[f"d_{m}_lo"], row[f"d_{m}_hi"] = paired_delta(
                    gap.to_numpy(), (x[f"pred_{reference}"] - x["age"]).to_numpy(), x["subject"].to_numpy())
            if not (kind == "report" and name in ("reference", "normal")):
                adj = adjusted(d.assign(gap=d[f"pred_{m}"] - d["age"]), mask & ~d["reference"], d["reference"])
                row.update({f"{k.replace('gap_adj', 'gapadj')}_{m}": v for k, v in adj.items()
                            if k.startswith("gap_adj")})
                if "gap_adj" in adj:
                    row[f"gapadj_sd_{m}"] = adj["gap_adj"] / spread[m]
        rows.append(row)
    out = pd.DataFrame(rows)
    sel = (out["kind"] == "pathology").to_numpy() & ~out["suppressed"].astype(bool).to_numpy()
    for m in models:
        if f"gapadj_p_{m}" in out:
            out.loc[sel, f"gapadj_q_{m}"] = bh(out.loc[sel, f"gapadj_p_{m}"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", action="append", required=True, metavar="MODEL:SOURCE=PATH")
    ap.add_argument("--reference", required=True, help="the model every paired difference is taken against")
    ap.add_argument("--cohort", default=COHORT)
    ap.add_argument("--targets", default="multitask_targets_v2.npz")
    ap.add_argument("--min-age", type=float, default=18.0)
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--min-subjects", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    preds = {}                                          # (model, source) -> series
    for spec in args.pred:
        key, path = spec.split("=", 1)
        model, source = key.split(":", 1)
        preds[(model, source)] = read_predictions(path, source)
    models = list(dict.fromkeys(m for m, _ in preds))
    sources = list(dict.fromkeys(s for _, s in preds))
    if args.reference not in models:
        raise SystemExit(f"--reference {args.reference} has no --pred")
    cohort, groups = load_cohort(args.cohort, args.targets, args.min_age)

    frames = {}
    for source in sources:
        cols = {f"pred_{m}": preds[(m, source)] for m in models if (m, source) in preds}
        if len(cols) != len(models):
            raise SystemExit(f"source {source}: every model needs predictions")
        frames[source] = cohort.join(pd.DataFrame(cols), how="inner").dropna(subset=list(cols))
    pooled = pd.concat(frames.values())
    if pooled.index.has_duplicates:
        raise SystemExit("a recording is in two sources")
    if len(sources) > 1:
        frames["pooled"] = pooled

    out = []
    for source, d in frames.items():
        for family in ("harvard", "temple"):
            t = table(d[d["family"] == family], groups, models, args.reference, args.min_n, args.min_subjects)
            out.append(t.assign(source=source, family=family))
            print(f"{source:7s} {family:8s} recordings {int((d['family'] == family).sum()):6d}")
    df = pd.concat(out, ignore_index=True)
    lead = ["source", "family", "kind", "group", "label", "suppressed", "n", "n_subjects", "age_mean"]
    df = df[lead + [c for c in df.columns if c not in lead]]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    show = lead[:2] + ["group", "n", "age_mean"] + [f"mae_{m}" for m in models] + \
        [c for m in models if m != args.reference for c in (f"d_{m}", f"d_{m}_lo", f"d_{m}_hi")] + \
        [f"gapadj_{m}" for m in models]
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(df[~df["suppressed"].astype(bool)][[c for c in show if c in df]].round(2).to_string(index=False))
    print("->", args.out)


if __name__ == "__main__":
    main()
