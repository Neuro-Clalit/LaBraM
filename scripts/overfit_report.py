#!/usr/bin/env python
"""Overfitting / headroom table for fine-tune runs (labram/eval/overfit.py).

    python scripts/overfit_report.py --run "Exp4-aug=checkpoints/tuh_exp4_.../log.txt:2" \\
        --run "Exp4base=/tmp/exp4base/finetune/log.txt:2" [--per-epoch] [--out report.json]

``NAME=LOG[:WARMUP]``: a run's per-epoch log.txt and its warm-up epochs.
"""
import argparse
import dataclasses
import json
import sys

import pandas as pd

from labram.eval.overfit import analyze, load_log


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, metavar="NAME=LOG[:WARMUP]")
    p.add_argument("--metric", default="mae")
    p.add_argument("-k", type=int, default=3)
    p.add_argument("--per-epoch", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    rows, full = [], {}
    for item in args.run:
        name, rest = item.split("=", 1)
        path, _, warm = rest.partition(":")
        r = analyze(load_log(path), args.metric, k=args.k, warmup_epochs=int(warm or 0))
        full[name] = dataclasses.asdict(r)
        rows.append({"run": name, "epochs": r.epochs, "best_ep": r.best_val_epoch,
                     "best_val": r.best_val, "test@best": r.test_at_best_val,
                     "regret": r.selection_regret, "val~test rho": r.val_test_spearman,
                     "gap": r.final_gap, "gap%": 100 * r.relative_gap, "gap_slope": r.gap_slope_last_k,
                     "d_val": r.last_val_improvement, "acc_val": r.last_val_acceleration,
                     "d_train": r.last_train_improvement, "overfit_ep": len(r.overfit_epochs),
                     "streak": r.overfit_streak, "headroom": r.headroom_estimate,
                     "verdict": r.verdict})
        if args.per_epoch:
            print(f"== {name}")
            print(pd.DataFrame(r.per_epoch).round(3).to_string(index=False))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    if args.out:
        json.dump(full, open(args.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    sys.exit(main())
