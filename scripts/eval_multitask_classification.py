#!/usr/bin/env python
"""Per-recording class probabilities of trained multi-task runs (status +
diagnosis groups), joined with their labels, for ROC / PR curves and
threshold metrics (best F1, errors).

For every run of ``--manifest`` (as ``scripts/eval_runs.py``: a JSON list of
``{"name", "model", "data_path"}``), rebuild the model and its val / test
samples from its own ``run_config.yaml`` (``eval_age_by_cohort.evaluate_run``),
score ``checkpoint-best.pth`` and average each recording's window
probabilities per class, as training's per-recording metrics do. Labels come
from the cohort's ``data.multitask_targets`` file (NaN = unknown). Writes
``--out`` (csv.gz): one row per run x split x recording x target with
``p`` (probability) and ``y`` (label). ``--metrics-out`` also writes
``labram.eval.binary_metrics.classification_table`` of it; ``--append`` adds
the manifest's runs to an existing ``--out`` (re-scored runs replace their rows).

    python scripts/eval_multitask_classification.py --manifest manifest_mt.json \\
        --out docs/data/multitask_class_predictions.csv.gz \\
        --metrics-out docs/data/multitask_classification_f1.csv
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
from eval_age_by_cohort import evaluate_run  # noqa: E402
from eval_runs import fetch_run  # noqa: E402

from labram.eval.binary_metrics import classification_table  # noqa: E402
from labram.eval.loading import load_run_config  # noqa: E402


def label_table(run_dir: str, data_path: str) -> pd.DataFrame:
    """recording stem x target labels of the run's ``data.multitask_targets``."""
    mt = load_run_config(os.path.join(run_dir, "run_config.yaml")).data.multitask_targets
    with np.load(mt if os.path.isabs(mt) else os.path.join(data_path, mt)) as z:
        names = [str(n).split("/")[-1] for n in z["names"]]
        return pd.DataFrame(z["targets"], index=[str(s) for s in z["stems"]], columns=names)


def score(name: str, run_dir: str, data_path: str, args, device) -> pd.DataFrame:
    res = evaluate_run(run_dir, data_path, "checkpoint-best.pth", tuple(args.splits), device,
                       args.batch_size, "npy", tokenizer_weight=args.tokenizer_weight,
                       stage_dir="none", num_workers=args.num_workers, class_probs=True)
    if "class_names" not in res:
        raise ValueError(f"{name}: not a multi-task run")
    names = [n.split("/")[-1] for n in res["class_names"]]
    labels = label_table(run_dir, data_path)[names]
    frames = []
    for split in args.splits:
        cp = res["splits"][split]["class_probs"]
        recs = list(cp)
        prob = pd.DataFrame(np.stack([cp[r] for r in recs]), index=recs, columns=names)
        lab = labels.reindex(recs)
        for target in names:
            frames.append(pd.DataFrame({"run": name, "split": split, "recording": recs, "target": target,
                                        "p": prob[target].to_numpy(), "y": lab[target].to_numpy()}))
        print(f"[{name}] {split}: {len(recs)} recordings", flush=True)
    return pd.concat(frames, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True, help="JSON list of {name, model, data_path}")
    ap.add_argument("--data-path", default=None, help="data root when an entry has no data_path")
    ap.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    ap.add_argument("--out", required=True, help="per-recording probabilities + labels (csv.gz)")
    ap.add_argument("--metrics-out", default=None, help="threshold / curve metrics table (csv)")
    ap.add_argument("--append", action="store_true",
                    help="keep the other runs already in --out (a re-scored run replaces its rows)")
    ap.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples for the CIs")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--tokenizer-weight", default="./checkpoints/vqnsp.pth")
    args = ap.parse_args(argv)
    with open(args.manifest) as f:
        manifest = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = []
    for m in manifest:
        with tempfile.TemporaryDirectory(prefix="eval_mt_", dir=args.work_dir) as tmp:
            frames.append(score(m["name"], fetch_run(m["model"], tmp),
                                m.get("data_path") or args.data_path, args, device))
    preds = pd.concat(frames, ignore_index=True)
    if args.append and os.path.exists(args.out):
        old = pd.read_csv(args.out)
        preds = pd.concat([old[~old.run.isin(set(preds.run))], preds], ignore_index=True)
    preds.to_csv(args.out, index=False, float_format="%.6g")
    if args.metrics_out:
        classification_table(preds, n_boot=args.n_boot).to_csv(args.metrics_out, index=False,
                                                                float_format="%.6g")


if __name__ == "__main__":
    main()
