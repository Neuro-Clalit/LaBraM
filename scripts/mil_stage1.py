#!/usr/bin/env python
"""MIL Stage 1: bag pooling variants on cached (frozen) window features.

Runs every variant of ``--variants`` (default: the grid below) for each run
and protocol, against the two benchmarks ``M0-run`` (the run's own mean
pooling) and ``M0`` (mean pooling with the head refit), and writes

* ``metrics.csv``   -- MAE / RMSE / r per variant x split x group (all, TUH,
  HEEDB, dataset x cohort) with the paired-bootstrap change vs ``M0``;
* ``analytics.csv`` -- per variant (test): pooling weight vs the window's own
  error (within-recording Spearman, excess error by weight decile), weight vs
  descriptors (EMG, amplitude, flat channel, channel spread, minute, theta
  bursts), concentration, the weighted-sum identity, and recording confidence
  from the weights;
* ``<run>/<protocol>/<variant>_test_windows.parquet`` -- per-window weights.

    python scripts/mil_stage1.py --cache /data/runs/heedb/mil_cache --run E2L --run E2 \\
        --protocol val_cf --protocol train --out-dir /data/runs/heedb/mil_stage1
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from labram.layers.bag_pool import DescriptorNorm
from labram.mil import stage1 as s1
from labram.mil.cache import Cache

V = s1.Variant
GRID = [
    V("M0-run", kind="mean", fit_head=False),
    V("M0", kind="mean"),
    V("M1-h", kind="gated", score_input="h"),
    V("M1-q", kind="gated", score_input="q"),
    V("M1-hq", kind="gated", score_input="hq"),
    V("M1-hq-time", kind="gated", score_input="hq", time_encoding=True),
    V("M1-hq-noent", kind="gated", score_input="hq", entropy_weight=0.0),
    V("M2-h", kind="gated", score_input="h", branches=4, attn_dim=64, topk_mask_prob=0.3),
    V("M2-hq", kind="gated", score_input="hq", branches=4, attn_dim=64, topk_mask_prob=0.3),
    V("M3-h", kind="sigmoid", score_input="h", attn_dim=64),
    V("M3-q", kind="sigmoid", score_input="q", attn_dim=64),
    V("M3-hq", kind="sigmoid", score_input="hq", attn_dim=64),
    V("M4-hq", kind="query", score_input="hq"),
    V("M4-hq-sa", kind="query", score_input="hq", self_attention=True),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--protocol", action="append", choices=["val_cf", "train"])
    ap.add_argument("--variants", nargs="*", default=None, help="names from the grid (default: all)")
    ap.add_argument("--epochs", type=int, default=None, help="override every variant's epochs")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    protocols = args.protocol or ["val_cf"]
    grid = [v for v in GRID if args.variants is None or v.name in args.variants]
    if args.epochs:
        for v in grid:
            v.epochs = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    all_metrics, all_analytics = [], []

    for run in args.run:
        cache = Cache(args.cache, run)
        stats = tuple(cache.info["pred_stats"])
        head = {k: v.float() for k, v in cache.head_state().items()}
        splits = ["val", "test"] + (["train"] if "train" in protocols else [])
        t0 = time.time()
        data = {s: s1.load_bags(cache, s, stats, device) for s in splits}
        print(f"{run}: bags " + ", ".join(f"{s} {len(b)}" for s, b in data.items())
              + f" ({time.time() - t0:.0f}s)", flush=True)
        ref = data.get("train", data["val"])
        norm = DescriptorNorm(ref.q.shape[-1]).fit(ref.q[ref.mask].cpu()).to(device)

        for protocol in protocols:
            outdir = os.path.join(args.out_dir, run, protocol)
            os.makedirs(outdir, exist_ok=True)
            results = {}
            for v in grid:
                t0 = time.time()
                res = s1.run_variant(v, data, head, norm, stats, protocol, args.seed)
                results[v.name] = res
                print(f"  {run}/{protocol}/{v.name}: test MAE "
                      f"{np.abs(res['test']['pred'] - data['test'].rec['age'].to_numpy()).mean():.3f}"
                      f"  val {np.abs(res['val']['pred'] - data['val'].rec['age'].to_numpy()).mean():.3f}"
                      f"  ({time.time() - t0:.0f}s)", flush=True)
            ref_pred = {s: results["M0"][s]["pred"] for s in ("val", "test")}
            for name, res in results.items():
                for split in ("val", "test"):
                    all_metrics += [dict(r, run=run, protocol=protocol) for r in s1.metrics_rows(
                        name, split, data[split].rec, res[split]["pred"],
                        None if name == "M0" else ref_pred[split])]
                head_used = res.get("head_after", head)
                win = s1.window_frame(data["test"], res["test"]["alpha"], head_used, stats,
                                      res["test"].get("gate"))
                win.to_parquet(os.path.join(outdir, f"{name}_test_windows.parquet"))
                an = s1.attention_analytics(win, data["test"].rec, res["test"]["pred"])
                an.update(run=run, protocol=protocol, variant=name,
                          **{f"cfg_{k}": val for k, val in s1.variant_record(
                              next(v for v in grid if v.name == name)).items()})
                if "info" in res:
                    an.update(best_epoch=res["info"]["best_epoch"])
                all_analytics.append(an)
                for split in ("val", "test"):
                    pd.DataFrame({"recording": data[split].rec["recording"], "pred": res[split]["pred"],
                                  "age": data[split].rec["age"]}).to_parquet(
                        os.path.join(outdir, f"{name}_{split}_pred.parquet"))
            pd.DataFrame(all_metrics).to_csv(os.path.join(args.out_dir, "metrics.csv"), index=False)
            pd.DataFrame(all_analytics).drop(columns=["excess_by_decile"]).to_csv(
                os.path.join(args.out_dir, "analytics.csv"), index=False)
            with open(os.path.join(args.out_dir, "excess_by_decile.json"), "w") as fh:
                json.dump({f"{a['run']}/{a['protocol']}/{a['variant']}": a["excess_by_decile"]
                           for a in all_analytics}, fh, indent=1)

    m = pd.DataFrame(all_metrics)
    with pd.option_context("display.width", 250, "display.max_rows", 400):
        print(m[m.group.isin(["all", "temple", "harvard"])].pivot_table(
            index=["run", "protocol", "variant"], columns=["split", "group"], values="mae").round(3))


if __name__ == "__main__":
    main()
