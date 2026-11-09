#!/usr/bin/env python
"""Recording MAE of every aggregation experiment per dataset, TUAB membership and cohort.

Collects per-recording val/test age predictions of the reference run (E2L),
post-hoc A3, the Stage 1 pools, the end-to-end bag runs (from their saved
per-window weights: prediction = sum_i alpha_i * window prediction, exact for
the linear head without time encoding) and the window-level controls (from
``scripts/eval_age_by_cohort.py`` scores), and reports MAE per group:

* TUAB normal / abnormal -- TUH recordings that belong to TUAB (test TUH is
  exactly the TUAB eval set; val holds TUAB train recordings);
* TUEG (non-TUAB TUH) normal / abnormal / unlabelled;
* HEEDB normal / abnormal / unlabelled; plus all / TUH / HEEDB.

    python scripts/mil_cohort_summary.py --out docs/data/mil_summary/all_experiments_by_tuab_cohort.csv
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

DATA = "/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1"
TUH_INDEX = "/data/datasets/EEG-public/TUH_EEG/index/tuh_index.parquet"
SCORES = "/data/runs/heedb/scores"
WINDOWS = "/data/runs/heedb/window_conf/E2L_windows.parquet"
S1 = "/data/runs/heedb/mil_summary/stage1_train/E2L/train"
ART = "/data/runs/heedb/mil_sagemaker/artifacts"
LOCAL = "/data/runs/heedb/mil_local"
WINDOW_SCORES = "/data/runs/heedb/mil_summary/F0_W0_scores.json"
GROUPS = ["all", "TUH", "HEEDB", "TUAB normal", "TUAB abnormal", "TUEG normal", "TUEG abnormal",
          "TUEG unlabelled", "HEEDB normal", "HEEDB abnormal", "HEEDB unlabelled"]


def recording_table():
    from labram.data.tuh_metadata import load_label_lookup_for
    man = pd.DataFrame(json.load(open(os.path.join(DATA, "manifest.json")))["recordings"])[["stem", "corpus"]]
    labels = load_label_lookup_for(DATA)
    idx = pd.read_parquet(TUH_INDEX, columns=["stem", "corpora"])
    tuab = set(idx.loc[idx["corpora"].astype(str).str.contains("tuab"), "stem"])
    man["cohort"] = [labels.get(s) if labels.get(s) in ("normal", "abnormal") else "unlabelled" for s in man.stem]
    man["source"] = np.where(man.corpus == "heedb", "HEEDB", np.where(man.stem.isin(tuab), "TUAB", "TUEG"))
    return man.rename(columns={"stem": "recording"}).set_index("recording")


def ages():
    w = pd.read_parquet(WINDOWS, columns=["split", "recording", "age"])
    return {s: g.groupby("recording")["age"].first() for s, g in w.groupby("split")}


def from_weights(path):
    w = pd.read_parquet(path)
    return (w["alpha"] * w["window_pred"]).groupby(w["recording"]).sum()


def collect():
    preds = {}                                     # (experiment) -> {split: Series}
    for split in ("val", "test"):
        p = {}
        for f in ("E2L_tuab_valtest.json", "E2L_heedb_valtest.json"):
            p.update(json.load(open(os.path.join(SCORES, f)))["E2L"]["splits"][split]["predictions"])
        preds.setdefault("E2L (reference)", {})[split] = pd.Series(p)
    # A3 (base descriptors), val cross-fitted, as in window_confidence_results.md
    from labram.eval import window_confidence as wc
    df = wc.add_tta_stats(pd.read_parquet(WINDOWS).reset_index(drop=True))
    w, _ = wc.a3_weights(df)
    rec = wc.pool(df, w)
    for split, g in rec.groupby("split"):
        preds.setdefault("A3 post-hoc", {})[split] = g.set_index("recording")["pred"]
    for f in sorted(glob.glob(os.path.join(S1, "*_val_pred.parquet"))):
        name = os.path.basename(f)[:-len("_val_pred.parquet")]
        for split in ("val", "test"):
            d = pd.read_parquet(f.replace("_val_", f"_{split}_"))
            preds.setdefault(f"S1 {name}", {})[split] = d.set_index("recording")["pred"]
    l1 = glob.glob(os.path.join(LOCAL, "L1_abmil_*", "analysis"))[0]
    bag = {"S3 L1": (l1, 3)}
    for k in ("w1", "w2", "w3", "w4", "w5", "w6", "w7"):
        bag[f"S3 {k.upper()}"] = (os.path.join(ART, k, "finetune", "analysis"), 3)
    f2 = glob.glob(os.path.join(LOCAL, "F2_*", "analysis"))
    if f2:
        bag["S4 F2"] = (f2[0], 10)
    for name, (d, ep) in bag.items():
        for split in ("val", "test"):
            path = os.path.join(d, f"{split}_bag_weights_epoch{ep:02d}.parquet")
            if os.path.exists(path):
                preds.setdefault(name, {})[split] = from_weights(path)
    if os.path.exists(WINDOW_SCORES):
        s = json.load(open(WINDOW_SCORES))
        for run, label in (("W0", "S3 W0"), ("F0", "S4 F0")):
            if run in s:
                for split in ("val", "test"):
                    preds.setdefault(label, {})[split] = pd.Series(s[run]["splits"][split]["predictions"])
    return preds


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    recs, age, preds = recording_table(), ages(), collect()
    rows = []
    for name, by_split in preds.items():
        for split, p in by_split.items():
            d = pd.DataFrame({"pred": p}).join(recs, how="inner")
            d["age"] = age[split].reindex(d.index)
            d = d.dropna(subset=["age"])
            err = (d["pred"] - d["age"]).abs()
            row = {"experiment": name, "split": split, "n": len(d)}
            for g in GROUPS:
                if g == "all":
                    m = np.ones(len(d), bool)
                elif g == "TUH":
                    m = (d.source != "HEEDB").to_numpy()
                elif g == "HEEDB":
                    m = (d.source == "HEEDB").to_numpy()
                else:
                    src, coh = g.split()
                    m = ((d.source == src) & (d.cohort == coh)).to_numpy()
                row[g] = float(err[m].mean()) if m.sum() >= 2 else np.nan
                row[f"n {g}"] = int(m.sum())
            rows.append(row)
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    for split in ("test", "val"):
        print(split)
        print(out[out.split == split].set_index("experiment")[GROUPS].round(2).to_string())
    print(out[out.experiment == "E2L (reference)"][["split"] + [f"n {g}" for g in GROUPS]].to_string(index=False))


if __name__ == "__main__":
    main()
