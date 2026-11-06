#!/usr/bin/env python
"""Per-epoch metrics of fine-tune runs, parsed from their console logs.

One row per evaluated epoch: val MAE (overall, Harvard, Temple), test MAE
(overall, Harvard = HEEDB test, Temple = TUAB eval with its normal / abnormal
recordings), train-eval MAE (overall, per dataset), the learning rate and the
wall time of the training pass and of the evaluation. The epoch number is the
order of the ``Val EEG mae:`` lines (a resumed run appends to the same log).
Aggregate numbers only, so the CSV can live in git.

    python scripts/epoch_metrics_from_log.py \\
        --run E2=/data/runs/heedb/e2_r2/finetune/log/run.log \\
        --run E2L=/data/runs/heedb/e2l/train.log --out docs/data/heedb_epoch_metrics.csv
"""
import argparse
import re
from datetime import datetime

import pandas as pd

TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]")
KV = re.compile(r"(\w+)=([\d.]+)")


def _ts(line):
    m = TS.match(line)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") if m else None


def parse(path):
    rows, cur, train_start, eval_start = [], {}, None, None
    for line in open(path, errors="replace"):
        t = _ts(line)
        if "Start training" in line:          # (re)start or resume: the pass begins here
            train_start = t
        if "Averaged stats" in line:
            lr = re.search(r"lr: ([\d.e-]+)", line)
            cur["lr"] = float(lr.group(1)) if lr else None
            if train_start and t:
                cur["train_min"] = (t - train_start).total_seconds() / 60
            eval_start = t
        elif "Val EEG mae:" in line:
            cur["val"] = float(line.rsplit(":", 1)[1])
        elif "Test EEG mae:" in line:
            cur["test"] = float(line.rsplit(":", 1)[1])
        elif "TrainEval EEG mae:" in line:
            cur["train_eval"] = float(line.rsplit(":", 1)[1])
        elif "EEG mae by dataset/cohort" in line:
            kv = dict((k, float(v)) for k, v in KV.findall(line.split("cohort:", 1)[1]))
            split = ("val" if "Val EEG" in line else "test" if "Test EEG" in line else "train_eval")
            for k in ("harvard", "temple", "temple_normal", "temple_abnormal", "normal", "abnormal"):
                if k in kv:
                    cur[f"{split}_{k}"] = kv[k]
        elif "Best mae val" in line or "Best " in line and " val: " in line:
            if "val" in cur:
                if eval_start and t:
                    cur["eval_min"] = (t - eval_start).total_seconds() / 60
                rows.append(cur)
            cur, train_start, eval_start = {}, t, None
    df = pd.DataFrame(rows)
    df.insert(0, "epoch", range(1, len(df) + 1))
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="append", required=True, metavar="NAME=LOG")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    parts = []
    for item in args.run:
        name, path = item.split("=", 1)
        parts.append(parse(path).assign(run=name))
    df = pd.concat(parts, ignore_index=True)
    df = df[["run"] + [c for c in df.columns if c != "run"]]
    df = df.round({c: 3 for c in df.columns if c not in ("run", "lr")})
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False, max_cols=12))


if __name__ == "__main__":
    main()
