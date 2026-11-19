#!/usr/bin/env python
"""Error by age group, model, age-bias correction and rejection (Tables 13-15 of
``docs/tech_report/``).

For every cohort set (as in ``external_cohort_table.py``; predictions may come
from several evaluation JSONs) and model:

* units: recordings, or sessions for TDBRAIN / Dortmund; subsets ``all``,
  ``normal`` (normal report or healthy), ``abnormal`` (abnormal report or
  patient) and ``unlabelled``;
* corrections fitted on the normal-report adult **validation** predictions of the
  training cohort (``bias_correction_cohorts.calibration`` +
  ``labram/eval/age_analysis.py::fit_bias_correction``) and applied unchanged:
  ``none``, ``cole``, ``recalibrate`` (no true age) and ``age_level``,
  ``quadratic_gap`` (use the true age; their MAE is a mean absolute corrected gap);
* rejection: the 20 % of a cohort with the highest label-free score set aside,
  the threshold over the whole cohort (``total`` for an ensemble, ``spread`` for
  a single model whose windows were saved; models without saved windows get no
  rejection rows);
* age bands 18-29, 30-39, ..., 70+ and ``all``.

Rows: cohort, model, method, rejection, group, band, n, kept share of the band,
mean age, MAE, bias. Pools: ``indomain`` (Temple + Harvard) and ``external``
(TDBRAIN, LEMON, Dortmund, SRM: the healthy-volunteer and outpatient cohorts)
are added as cohorts by concatenating their units.

    python scripts/report_age_groups.py --set ... --models E2L L1 W4 P3ens P4ens \\
        --val E2L=<json> --val L1=<json> --val W4=<json> --val-windows P4ens=<parquets> ... \\
        --val-meta <tuh_heedb_v1>/age_metadata.json --out <csv>
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rejection_scores as rs  # noqa: E402
from bias_correction_cohorts import calibration  # noqa: E402
from external_cohort_table import load_meta, units  # noqa: E402
from labram.eval.age_analysis import apply_bias_correction, fit_bias_correction  # noqa: E402

METHODS = ["none", "cole", "recalibrate", "age_level", "quadratic_gap"]
BANDS = [(18, 30, "18–29"), (30, 40, "30–39"), (40, 50, "40–49"), (50, 60, "50–59"), (60, 70, "60–69"), (70, 200, "70+")]
POOLS = {"indomain": ["temple", "harvard"], "external": ["tdbrain", "lemon", "dortmund", "srm"]}
REJECT = 0.2


def unit_scores(meta, wdir, members):
    """Rejection score per unit (total for an ensemble, spread for one model), or None."""
    if not wdir or not all(os.path.exists(os.path.join(wdir, f"{m}_test.parquet")) for m in members):
        return None
    sc = rs.cohort_units(meta, wdir, members)
    key = ["subject", "session"] if "task" in meta else ["stem"]
    return sc[key + ["total" if len(members) > 1 else "spread"]].rename(columns={"total": "score", "spread": "score"})


def rows_for(name, model, d, method, rejection):
    out = []
    h, p = d["healthy"].to_numpy(bool), d["pathological"].to_numpy(bool)
    groups = {"all": np.ones(len(d), bool)}
    if p.any() or (~(h | p)).any():
        groups.update({"normal": h, "abnormal": p, "unlabelled": ~(h | p)})
    age = d["age"].to_numpy(float)
    keep = d["keep"].to_numpy(bool)
    err = d["corr"].to_numpy(float) - age
    for g, gm in groups.items():
        for lo, hi, band in [(0, 200, "all")] + BANDS:
            bm = gm & (age >= lo) & (age < hi)
            km = bm & keep
            if km.sum() < 10:
                continue
            out.append({"cohort": name, "model": model, "method": method, "rejection": rejection, "group": g,
                        "band": band, "n": int(km.sum()), "kept_share": km.sum() / bm.sum(),
                        "age_mean": age[km].mean(), "mae": np.abs(err[km]).mean(), "bias": err[km].mean()})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", action="append", required=True,
                    metavar="NAME:COHORT:EVAL.json[,EVAL.json...]:WINDOWS_DIR[:FAMILY]")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--val", action="append", default=[], metavar="MODEL=JSON")
    ap.add_argument("--val-windows", action="append", default=[], metavar="MODEL=PARQUET[,PARQUET...]")
    ap.add_argument("--val-meta", required=True)
    ap.add_argument("--min-age", type=float, default=18.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    vals = dict(v.split("=", 1) for v in args.val)
    val_windows = {k: v.split(",") for k, v in (e.split("=", 1) for e in args.val_windows)}
    params = {}
    for m in args.models:
        pred, age, _ = calibration(m, vals, val_windows, args.val_meta, args.min_age)
        params[m] = fit_bias_correction(pred, age)
        print(f"{m}: correction fitted on {len(pred):,} validation predictions, slope {params[m]['slope']:.3f}", flush=True)
    units_by = {}
    for spec in args.set:
        name, cdir, evs, wdir, *family = spec.split(":")
        meta = load_meta(cdir, family[0] if family else None, args.min_age)
        res = {}
        for ev in evs.split(","):
            res.update(json.load(open(ev)))
        for m in args.models:
            if m not in res:
                print(f"{name}: {m} not scored", flush=True)
                continue
            pred = pd.Series(res[m]["splits"]["test"]["predictions"], dtype=float)
            if "task" in meta:
                d = units(meta, pred)
            else:
                d = units(meta.assign(stem=meta.index), pred, extra=["stem"])
            sc = unit_scores(meta, wdir, res[m].get("members", [m]))
            if sc is not None:
                key = ["subject", "session"] if "task" in meta else ["stem"]
                d = d.merge(sc, on=key, how="left")
            units_by[(name, m)] = d
    rows = []
    # pooled sets: concatenate units of the member cohorts (rejection threshold stays per cohort)
    for pool, members in POOLS.items():
        for m in args.models:
            parts = [units_by[(c, m)].assign(_cohort=c) for c in members if (c, m) in units_by]
            if len(parts) == len(members):
                units_by[(pool, m)] = pd.concat(parts, ignore_index=True)
    for (name, m), d in units_by.items():
        has_score = "score" in d and d["score"].notna().all()
        if has_score:
            by = d["_cohort"] if "_cohort" in d else pd.Series(name, index=d.index)
            thr = d.groupby(by)["score"].transform(lambda s: s.quantile(1 - REJECT))
            keep80 = (d["score"] <= thr).to_numpy()
        for method in METHODS:
            corr = apply_bias_correction(d["pred"].to_numpy(), params[m], method, age=d["age"].to_numpy())
            base = d.assign(corr=corr)
            rows += rows_for(name, m, base.assign(keep=True), method, "none")
            if has_score:
                rows += rows_for(name, m, base.assign(keep=keep80), method, f"reject {REJECT:.0%}")
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(args.out, len(df))
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        t = df[(df.method == "none") & (df.rejection == "none") & (df.group == "all")]
        print(t.pivot_table(index=["cohort", "band"], columns="model", values="mae").round(2).to_string())


if __name__ == "__main__":
    main()
