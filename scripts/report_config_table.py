#!/usr/bin/env python
"""Every configuration field on which the report's models differ, read from each run's
saved ``run_config.yaml`` (Table 16 of ``docs/tech_report/``). Bookkeeping fields
(output, logging, ClearML, SageMaker, data paths, evaluation reporting) are left out
(``labram.eval.research_summary.NON_SCIENTIFIC``), and the ``mil.*`` fields of runs
without bag training (``mil.enabled`` false) are blanked, since such runs never read them.

    python scripts/report_config_table.py --run E2L=<run dir> --run P4=<fold dir> ... --out <csv>
"""
import argparse
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from labram.eval.research_summary import NON_SCIENTIFIC, flatten  # noqa: E402

SKIP = NON_SCIENTIFIC + ("evaluation.", "sagemaker", "finetune_checkpoint.finetune")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="append", required=True, metavar="NAME=RUN_DIR")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs = dict(r.split("=", 1) for r in args.run)
    flat = {}
    for name, d in runs.items():
        f = flatten(yaml.safe_load(open(os.path.join(d, "run_config.yaml"))))
        if not f.get("mil.enabled", False):
            f = {k: v for k, v in f.items() if not k.startswith("mil.")}
        flat[name] = f
    rows = []
    for key in sorted(set().union(*flat.values())):
        if key.startswith(SKIP):
            continue
        vals = [str(flat[n].get(key, "–")) for n in runs]
        rows.append({"field": key, "differs": len(set(vals)) > 1, **dict(zip(runs, vals))})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200, "display.max_colwidth", 30):
        print(df[df.differs].drop(columns="differs").to_string(index=False))
    print(args.out)


if __name__ == "__main__":
    main()
