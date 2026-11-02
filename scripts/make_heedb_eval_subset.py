#!/usr/bin/env python
"""A small local copy of an S3 HEEDB cohort's evaluation split, for scoring models.

Scoring a recording reads only its first ``eval_minutes`` after the start trim
(``WindowSelection``: windows ``[trim, min(trim + eval, n - trim))``). Keeping
each evaluated recording's first ``trim + eval + trim`` windows, with
``n_windows`` set to that length, leaves this selection unchanged, and is about
4 MB per recording instead of about 20 MB. Only the sampled ``--split``
recordings are fetched. The loader still needs a training split for its target
stats, so ``--keep-train`` training rows stay listed, without their recordings.
Those stats cancel out for the true ages, and predictions are de-normalized with
the scored model's own training stats
(``eval_age_by_cohort.py --pred-stats-from-log``). The other evaluation split is
dropped, which keeps the loader from building millions of window names.

    python scripts/make_heedb_eval_subset.py \\
        --cohort s3://eeg-data-public/HEEDB/v4.1/cohorts/heedb_routine_v1 \\
        --out /tmp/heedb_eval --split test --per-label 700
"""
import argparse
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataset_maker import heedb_index as hx  # noqa: E402
from dataset_maker.make_heedb_cohort import split_s3  # noqa: E402

W = 2000                     # samples per 10 s window at 200 Hz


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort", required=True, help="s3:// cohort dir")
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--per-label", type=int, default=700, help="recordings per report label (normal/abnormal/none)")
    p.add_argument("--trim-sec", type=int, default=60)
    p.add_argument("--eval-minutes", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-train", type=int, default=300, help="training rows kept for the target stats")
    args = p.parse_args(argv)
    client = hx.s3_client(None)
    bucket, prefix = split_s3(args.cohort)

    def get(name):
        return client.get_object(Bucket=bucket, Key=f"{prefix}/{name}")["Body"].read()

    manifest = json.loads(get("manifest.json"))
    meta = json.loads(get("age_metadata.json"))
    split = json.loads(get("age_split.json"))
    cohort = pd.read_parquet(io.BytesIO(get("cohort.parquet")))
    ev = cohort[cohort["split"] == args.split].assign(lab=lambda d: d["label"].fillna("none"))
    pick = (ev.groupby("lab", group_keys=False)
              .apply(lambda g: g.sample(n=min(args.per_label, len(g)), random_state=args.seed)))
    keep = set(pick["stem"])
    other = "val" if args.split == "test" else "test"
    train_rows = cohort[cohort["split"] == "train"].sample(
        n=min(args.keep_train, int((cohort["split"] == "train").sum())), random_state=args.seed)
    train_keep = set(train_rows["stem"])
    trim = args.trim_sec // 10
    keep_windows = 2 * trim + int(args.eval_minutes * 6)

    os.makedirs(os.path.join(args.out, "recordings"), exist_ok=True)
    rows = []
    for r in manifest["recordings"]:
        if (r["source_split"] == args.split and r["stem"] not in keep) or r["source_split"] == other \
                or (r["source_split"] == "train" and r["stem"] not in train_keep):
            continue
        r = dict(r)
        if r["stem"] in keep:
            r["n_windows"] = min(r["n_windows"], keep_windows)
            r["n_samples"] = r["n_windows"] * W
        rows.append(r)
    stems = {r["stem"] for r in rows}
    manifest.update(recordings=rows, n_recordings=len(rows), n_windows=sum(r["n_windows"] for r in rows),
                    eval_subset={"cohort": args.cohort, "split": args.split, "per_label": args.per_label,
                                 "keep_windows": keep_windows})
    split["files"][args.split] = [f for f in split["files"][args.split]
                                  if f.split("/")[1].rsplit("_", 1)[0] in keep]
    split["files"]["train"] = [f for f in split["files"]["train"]
                               if f.split("/")[1].rsplit("_", 1)[0] in train_keep]
    split["files"][other] = []
    meta["recordings"] = {k: v for k, v in meta["recordings"].items() if k in stems}
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"))
    json.dump(meta, open(os.path.join(args.out, "age_metadata.json"), "w"))
    json.dump(split, open(os.path.join(args.out, "age_split.json"), "w"))
    pick.drop(columns="lab").to_parquet(os.path.join(args.out, "eval_subset.parquet"), index=False)

    def fetch(stem):
        dst = os.path.join(args.out, "recordings", f"{stem}.npy")
        if os.path.exists(dst):
            return 0
        body = client.get_object(Bucket=bucket, Key=f"{prefix}/recordings/{stem}.npy")["Body"].read()
        x = np.load(io.BytesIO(body))
        np.save(dst, np.ascontiguousarray(x[:keep_windows * W]))
        return x[:keep_windows * W].nbytes

    with ThreadPoolExecutor(32) as pool:
        total = sum(pool.map(fetch, sorted(keep)))
    print(json.dumps({"recordings": len(keep), "by_label": pick["lab"].value_counts().to_dict(),
                      "GB": round(total / 1e9, 2), "out": args.out}))


if __name__ == "__main__":
    sys.exit(main())
