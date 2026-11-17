#!/usr/bin/env python
"""Merge ``eval_age_by_cohort.py`` outputs and add fold ensembles as models.

An ensemble's prediction for recording ``r`` is the mean of its member models'
predictions, ``p_r = (1/K) sum_k p_{r,k}`` over the ``K`` members that scored
``r`` (Eq. (13.1) of ``docs/mil_stage1_results.md``, the 5-fold ensembles of
the phased CV). The output has the input's structure
(``{model: {"splits": {split: {"predictions": {stem: years}}}}}``), so
``external_cohort_comparison.py``, ``tdbrain_brain_age.py`` and
``external_eval_report.py`` read the ensembles like any model. Only the
predictions of an ensemble are written; its metrics come from those scripts.

    python scripts/ensemble_predictions.py \\
        --eval tdbrain_all_p34.json tdbrain_all_p4f3.json \\
        --ensemble P3ens='P3_L1_f*' --ensemble P4ens='P4_L1_f*' --ensemble P34ens='P[34]_L1_f*' \\
        --out tdbrain_all_p34_ens.json
"""
import argparse
import fnmatch
import json

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval", nargs="+", required=True, help="eval_age_by_cohort.py outputs to merge")
    ap.add_argument("--ensemble", action="append", default=[], metavar="NAME=GLOB",
                    help="ensemble NAME = mean of the models whose names match GLOB")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    merged = {}
    for path in args.eval:
        for model, res in json.load(open(path)).items():
            if model in merged:
                raise SystemExit(f"model {model!r} appears in two inputs")
            merged[model] = res
    for spec in args.ensemble:
        name, pattern = spec.split("=", 1)
        members = sorted(m for m in merged if fnmatch.fnmatch(m, pattern))
        if not members:
            raise SystemExit(f"ensemble {name}: no model matches {pattern!r}")
        preds = pd.DataFrame({m: pd.Series(merged[m]["splits"][args.split]["predictions"], dtype=float)
                              for m in members})
        merged[name] = {"members": members,
                        "splits": {args.split: {"predictions": preds.mean(axis=1).to_dict()}}}
        print(f"{name}: mean of {len(members)} models {members}, {len(preds)} recordings")
    json.dump(merged, open(args.out, "w"))
    print(args.out)


if __name__ == "__main__":
    main()
