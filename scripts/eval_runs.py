#!/usr/bin/env python
"""Re-evaluate trained brain-age runs in eval mode only: no training.

For every run of ``--manifest`` (a JSON list of ``{"name", "model"}``, where
``model`` is a SageMaker ``model.tar.gz`` on S3, a local tarball or a run
directory), fetch the run, rebuild its model and data from its own
``run_config.yaml`` (``scripts/eval_age_by_cohort.py::evaluate_run``) on
``--data-path`` and score ``checkpoint-best.pth`` on the train, val and test
splits. Writes to ``--out-dir``:

* ``metrics.csv``: per run x split x cohort (all / normal / abnormal):
  recordings, MAE, RMSE, R², Pearson r (recording level: the mean of each
  recording's window predictions, in years). Split ``train`` scores every
  training window; ``train_eval_budget`` only the windows the run's own val /
  test scoring uses per recording (the first ``eval_minutes`` after the trim),
  so it compares directly with val and test.
* ``split_check.csv``: per run x split, the recordings the rebuilt split holds
  against the run's recorded ``data_split.json``: counts, identical or not,
  and how many differ each way.
* ``predictions.csv.gz``: per run x split x recording, predicted and true age.

Runs are processed one at a time and the files are rewritten after each, so a
partial result survives an interruption; ``--skip-done`` resumes.

    python scripts/eval_runs.py --manifest runs.json --data-path /data/.../edf \\
        --out-dir eval_out --data-format npy
"""
import argparse
import gc
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
from eval_age_by_cohort import evaluate_run  # noqa: E402

SPLITS = ("train", "val", "test")
COHORTS = ("all", "normal", "abnormal")


def fetch_run(model: str, workdir: str) -> str:
    """A local run directory for ``model`` (an ``s3://`` or local
    ``model.tar.gz``, or a directory). Tarballs are extracted under ``workdir``;
    SageMaker's tarballs hold the run under ``finetune/``."""
    if os.path.isdir(model):
        return model
    tar_path = model
    if model.startswith("s3://"):
        import boto3
        bucket, _, key = model[5:].partition("/")
        tar_path = os.path.join(workdir, "model.tar.gz")
        boto3.client("s3").download_file(bucket, key, tar_path)
    out = os.path.join(workdir, "run")
    with tarfile.open(tar_path) as tar:
        members = [m for m in tar.getmembers() if os.path.basename(m.name) in
                   ("checkpoint-best.pth", "run_config.yaml", "data_split.json", "log.txt")]
        tar.extractall(out, members=members)
    for root, _, files in os.walk(out):
        if "run_config.yaml" in files:
            return root
    raise FileNotFoundError(f"{model}: no run_config.yaml inside")


def metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    e = pred - true
    if len(true) < 2:
        return {"n": len(true), "MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "pearson_r": np.nan}
    return {"n": len(true), "MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
            "R2": float(1 - (e ** 2).sum() / ((true - true.mean()) ** 2).sum()),
            "pearson_r": float(np.corrcoef(pred, true)[0, 1])}


def per_recording(w: dict, keep=None) -> pd.DataFrame:
    """Mean window prediction per recording (years) over the windows ``keep``."""
    d = pd.DataFrame({"recording": w["recs"], "name": w["names"], "pred": w["pred"], "true": w["true"],
                      "cohort": w["labels"]})
    if keep is not None:
        d = d[keep(d)]
    return d.groupby("recording", sort=False).agg(pred=("pred", "mean"), true=("true", "first"),
                                                 cohort=("cohort", "first"), windows=("pred", "size"))


def eval_budget(data: dict):
    """The windows val / test scoring keeps per recording: the first
    ``eval_minutes`` of windows after the trim (all windows when it is 0)."""
    n = int(round(float(data.get("eval_minutes") or 0) * 60 / float(data.get("window_sec") or 10)))
    if n <= 0:
        return None

    def keep(d):
        k = d["name"].str.extract(r"_(\d+)\.\w+$")[0].astype(int)
        rank = k.groupby(d["recording"]).rank(method="first")
        return rank <= n
    return keep


def score_run(name: str, run_dir: str, args, device) -> tuple:
    res = evaluate_run(run_dir, args.data_path, "checkpoint-best.pth", SPLITS, device, args.batch_size,
                       args.data_format, return_windows=True, tokenizer_weight=args.tokenizer_weight,
                       stage_dir="none", num_workers=args.num_workers)
    rows, preds = [], []
    recorded = {}
    split_json = os.path.join(run_dir, "data_split.json")
    if os.path.exists(split_json):
        with open(split_json) as f:
            recorded = json.load(f)
    budget = eval_budget(res["data"])
    for split in SPLITS:
        w = res["splits"][split]["windows"]
        views = [(split, per_recording(w))]
        if split == "train":
            views.append(("train_eval_budget", per_recording(w, budget)))
        for view, r in views:
            for cohort in COHORTS:
                s = r if cohort == "all" else r[r.cohort == cohort]
                rows.append({"run": name, "split": view, "cohort": cohort, "epoch": res["epoch"],
                             "windows": int(s.windows.sum()), **metrics(s.pred.to_numpy(), s.true.to_numpy())})
            if view == split:
                preds.append(r.reset_index().assign(run=name, split=split))
    checks = []
    for split in SPLITS:
        got = set(res["splits"][split]["predictions"])
        want = set(recorded.get(split, {}).get("recordings", [])) if recorded else set()
        checks.append({"run": name, "split": split, "recordings eval": len(got),
                       "recordings recorded": len(want) if recorded else np.nan,
                       "identical": (got == want) if recorded else np.nan,
                       "only in eval": len(got - want) if recorded else np.nan,
                       "only in recorded": len(want - got) if recorded else np.nan})
    return rows, checks, pd.concat(preds, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True, help="JSON list of {name, model}")
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-format", choices=["pickle", "npy"], default="npy")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4, help="data-loader workers (memory)")
    ap.add_argument("--mmap-cache", type=int, default=8,
                    help="open recording memory maps per loader process. Scoring every training "
                         "window touches whole recordings, and the default 256 per process exceeds "
                         "a 32-64 GB SageMaker instance's memory limit; eval reads recordings in order")
    ap.add_argument("--work-dir", default=None,
                    help="where models are downloaded / extracted (default: the system temp dir, "
                         "outside --out-dir so a crash leaves no checkpoint in the outputs)")
    ap.add_argument("--tokenizer-weight", default="./checkpoints/vqnsp.pth",
                    help="VQNSP weights for codebook-regularized runs")
    ap.add_argument("--only", nargs="*", default=None, help="evaluate only these run names")
    ap.add_argument("--skip-done", action="store_true", help="skip runs already in metrics.csv")
    args = ap.parse_args(argv)
    with open(args.manifest) as f:
        manifest = json.load(f)
    if args.only:
        manifest = [m for m in manifest if m["name"] in set(args.only)]
    os.makedirs(args.out_dir, exist_ok=True)
    paths = {k: os.path.join(args.out_dir, f) for k, f in
             (("metrics", "metrics.csv"), ("checks", "split_check.csv"), ("preds", "predictions.csv.gz"))}
    old = {k: pd.read_csv(p) for k, p in paths.items() if os.path.exists(p)}
    done = set(old["metrics"].run) if args.skip_done and "metrics" in old else set()
    acc = {k: [old[k]] if k in old else [] for k in paths}
    from labram.data.tuh_datasets import NpyWindowsMixin
    NpyWindowsMixin.MMAP_CACHE = args.mmap_cache
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for m in manifest:
        if m["name"] in done:
            print(f"[skip] {m['name']}", flush=True)
            continue
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="eval_runs_", dir=args.work_dir) as tmp:
            try:
                rows, checks, preds = score_run(m["name"], fetch_run(m["model"], tmp), args, device)
            except Exception as exc:          # one broken run must not stop the others
                print(f"[error] {m['name']}: {type(exc).__name__}: {exc}", flush=True)
                acc["checks"].append(pd.DataFrame([{"run": m["name"], "split": "ERROR", "identical": False,
                                                    "error": f"{type(exc).__name__}: {exc}"}]))
                pd.concat(acc["checks"], ignore_index=True).to_csv(paths["checks"], index=False)
                continue
        gc.collect()                          # free the run's model, data and loader workers
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        acc["metrics"].append(pd.DataFrame(rows))
        acc["checks"].append(pd.DataFrame(checks))
        acc["preds"].append(preds)
        for k, p in paths.items():
            pd.concat(acc[k], ignore_index=True).to_csv(p, index=False)
        test = [r for r in rows if r["split"] == "test" and r["cohort"] == "all"][0]
        train = [r for r in rows if r["split"] == "train" and r["cohort"] == "all"][0]
        ok = all(c["identical"] for c in checks if c["identical"] == c["identical"])
        print(f"[done] {m['name']:10s} {time.time() - t0:6.0f} s  train MAE {train['MAE']:.2f}  "
              f"test MAE {test['MAE']:.2f}  split identical: {ok}", flush=True)


if __name__ == "__main__":
    main()
