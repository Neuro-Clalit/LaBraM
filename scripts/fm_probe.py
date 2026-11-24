#!/usr/bin/env python
"""Ridge probe of frozen pretrained encoders (LaBraM, CBraMod, REVE) for brain age.

Stage A of ``docs/fm_encoder_onboarding.md``: each encoder is frozen, its window
embeddings are cached (``scripts/mil_cache.py --pretrained``), averaged per
recording, and a ridge regression of age is fitted on training recordings of
one fold of the best model's CV split, so every encoder sees the same
recordings as that model.

``sample``: pick the recordings to cache. Fold ``k`` of ``cv_split.json``
(groups = subjects) is the validation set and the other folds the training
set, as in the fine-tune runs; a seeded sample of each, proportional per
source dataset, is written as ``{"cache": {cohort split: stems}, "probe":
{"train": stems, "val": stems}}``. The held-out test split is cached whole.

    python scripts/fm_probe.py sample --cohort /data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1 \\
        --cv-split /data/runs/heedb/mil_phased_cv/P4_L1/fold_0/cv/fold_0/cv_split.json --fold 0 \\
        --n-train 20000 --n-val 5000 --out /data/runs/fm_probe/recordings.json

``fit``: per encoder and feature set (``mean``: the mean window embedding;
``mean_std``: mean and standard deviation over windows), standardise with the
training recordings, choose the ridge penalty by leave-one-out on them, and
score validation, the test split per source dataset, and every external cohort
cached under ``<root>/<cohort>/<encoder>/test``. Writes the metric table, the
predictions in the ``eval_age_by_cohort.py`` JSON layout, and a paired
bootstrap of each encoder against ``--reference``.

    python scripts/fm_probe.py fit --root /data/runs/fm_probe --encoders LaBraM CBraMod REVE \\
        --recordings /data/runs/fm_probe/recordings.json --cohorts lemon dortmund --out /data/runs/fm_probe/probe
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ALPHAS = np.logspace(-1, 5, 25)
FEATURES = ("mean", "mean_std")


def subject_of(stem: str) -> str:
    """CV group of a recording (``labram.data.cross_validation.group_id_for``)."""
    return stem.split("_")[0]


def sample_recordings(manifest: dict, cv_split: dict, fold: int, n_train: int, n_val: int, seed: int = 0) -> dict:
    """Seeded sample of fold ``fold``'s training and validation recordings,
    proportional per source corpus; see the module docstring for the layout."""
    val_groups = set(next(f["groups"] for f in cv_split["folds"] if f["fold"] == fold))
    pool_groups = {g for f in cv_split["folds"] for g in f["groups"]}
    rng = np.random.default_rng(seed)
    rows = pd.DataFrame([{"stem": r["stem"], "corpus": r.get("corpus", ""), "split": r["source_split"]}
                         for r in manifest["recordings"] if r["source_split"] in ("train", "val")])
    rows["group"] = rows["stem"].map(subject_of)
    rows = rows[rows["group"].isin(pool_groups)]
    rows["role"] = np.where(rows["group"].isin(val_groups), "val", "train")
    probe, cache = {}, {"train": [], "val": []}
    for role, n in (("train", n_train), ("val", n_val)):
        part = rows[rows["role"] == role]
        frac = min(1.0, n / max(len(part), 1)) if n > 0 else 1.0
        picked = []
        for _, grp in part.groupby("corpus", sort=True):
            k = len(grp) if frac >= 1.0 else int(round(len(grp) * frac))
            picked.append(grp.iloc[np.sort(rng.choice(len(grp), size=k, replace=False))])
        picked = pd.concat(picked)
        probe[role] = sorted(picked["stem"])
        for split in ("train", "val"):
            cache[split] += sorted(picked.loc[picked["split"] == split, "stem"])
    assert not {subject_of(s) for s in probe["train"]} & {subject_of(s) for s in probe["val"]}
    return {"fold": fold, "seed": seed, "cache": cache, "probe": probe}


def recording_features(cache_dir: str, split: str, kind: str = "mean") -> pd.DataFrame:
    """Per-recording features of one cached split: the mean window embedding
    (``mean``) or mean and standard deviation over windows (``mean_std``);
    columns ``age``, ``dataset``, ``cohort`` and ``n_windows`` are appended."""
    d = os.path.join(cache_dir, split)
    # Memory-mapped: a per-channel cache of the training split is tens of GB.
    h = np.load(os.path.join(d, "h.npy"), mmap_mode="r")
    meta = pd.read_parquet(os.path.join(d, "meta.parquet"))
    groups = meta.groupby("recording", sort=True).indices
    stems = list(groups)
    mean = np.empty((len(stems), h.shape[1]))
    std = np.empty_like(mean) if kind == "mean_std" else None
    for i, s in enumerate(stems):
        rows = np.asarray(h[np.sort(groups[s])], dtype=np.float64)
        mean[i] = rows.mean(0)
        if std is not None:
            std[i] = rows.std(0)
    feats = mean if std is None else np.hstack([mean, std])
    out = pd.DataFrame(feats, index=stems)
    first = meta.groupby("recording", sort=True).first()
    out["age"], out["dataset"], out["cohort"] = first["age"], first["dataset"], first["cohort"]
    out["n_windows"] = meta.groupby("recording", sort=True).size()
    return out


def _xy(df: pd.DataFrame):
    return df.drop(columns=["age", "dataset", "cohort", "n_windows"]).to_numpy(), df["age"].to_numpy(float)


def metrics(pred, true) -> dict:
    err = np.asarray(pred, float) - np.asarray(true, float)
    ss = ((true - true.mean()) ** 2).sum()
    return {"n": int(len(true)), "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "r2": float(1 - (err ** 2).sum() / ss) if ss > 0 else float("nan"),
            "r": float(np.corrcoef(pred, true)[0, 1]) if len(true) > 2 else float("nan")}


def fit_probe(train: pd.DataFrame):
    """StandardScaler + RidgeCV (leave-one-out over ``ALPHAS``) of z-scored age
    on the training recordings; returns ``predict(df) -> years`` and the penalty."""
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = _xy(train)
    mu, sd = y.mean(), y.std()
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    model.fit(X, (y - mu) / sd)
    return (lambda df: model.predict(_xy(df)[0]) * sd + mu), float(model[-1].alpha_)


def run_fit(root: str, encoders, recordings: dict, cohorts=(), main: str = "main", reference: str = None,
            min_age: float = 18.0, out: str = None, features=FEATURES):
    """Fit and score every encoder on each feature set in ``features``; returns
    ``(metric table, paired table)``."""
    from labram.eval.cv_compare import bootstrap_delta_mae
    roles = recordings["probe"]
    rows, preds, truth = [], {}, {}
    for enc in encoders:
        base = os.path.join(root, main, enc)
        for kind in features:
            pool = pd.concat([recording_features(base, s, kind) for s in ("train", "val")
                              if os.path.exists(os.path.join(base, s, "h.npy"))])
            pool = pool[~pool.index.duplicated()]
            train = pool.loc[pool.index.intersection(roles["train"])]
            sets = {"val": pool.loc[pool.index.intersection(roles["val"])]}
            for ds in sorted(set(sets["val"]["dataset"])):
                sets[f"val_{ds}"] = sets["val"][sets["val"]["dataset"] == ds]
            if os.path.exists(os.path.join(base, "test", "h.npy")):      # absent in a screening cache
                test = recording_features(base, "test", kind)
                sets["test"] = test
                for ds in sorted(set(test["dataset"])):
                    sets[f"test_{ds}"] = test[test["dataset"] == ds]
            for cohort in cohorts:
                ext = recording_features(os.path.join(root, cohort, enc), "test", kind)
                sets[cohort] = ext[ext["age"] >= min_age]
            predict, alpha = fit_probe(train)
            name = f"{enc}/{kind}"
            for group, df in sets.items():
                if len(df) < 1:
                    continue
                p = predict(df)
                rows.append({"encoder": enc, "features": kind, "dim": df.shape[1] - 4, "alpha": alpha,
                             "n_train": len(train), "group": group, **metrics(p, df["age"].to_numpy(float))})
                preds.setdefault(group, {})[name] = pd.Series(p, index=df.index)
                truth[group] = df["age"]
    table = pd.DataFrame(rows)
    paired = []
    reference = reference or encoders[0]
    for group, by_model in preds.items():
        for kind in features:
            ref = by_model.get(f"{reference}/{kind}")
            for enc in encoders:
                cur = by_model.get(f"{enc}/{kind}")
                if enc == reference or ref is None or cur is None:
                    continue
                idx = ref.index.intersection(cur.index)
                t = truth[group].loc[idx].to_numpy(float)
                paired.append({"group": group, "features": kind, "a": enc, "b": reference, "n": len(idx),
                               **bootstrap_delta_mae(cur.loc[idx] - t, ref.loc[idx] - t)})
    paired = pd.DataFrame(paired)
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        table.to_csv(out + "_metrics.csv", index=False)
        paired.to_csv(out + "_paired.csv", index=False)
        for group, by_model in preds.items():       # eval_age_by_cohort.py layout
            payload = {name: {"splits": {"test": {"predictions": {s: float(v) for s, v in p.items()}}}}
                       for name, p in by_model.items()}
            with open(f"{out}_pred_{group}.json", "w") as fh:
                json.dump(payload, fh)
    return table, paired


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="recordings to cache for the probe")
    s.add_argument("--cohort", required=True, help="training cohort folder (manifest.json)")
    s.add_argument("--cv-split", required=True)
    s.add_argument("--fold", type=int, default=0)
    s.add_argument("--n-train", type=int, default=20000, help="0 = every training recording of the fold")
    s.add_argument("--n-val", type=int, default=5000, help="0 = every validation recording of the fold")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    f = sub.add_parser("fit", help="fit and score the ridge probes")
    f.add_argument("--root", required=True, help="<root>/<main>/<encoder>/{train,val,test}, <root>/<cohort>/<encoder>/test")
    f.add_argument("--main", default="main", help="folder of the training-cohort cache under --root")
    f.add_argument("--encoders", nargs="+", required=True)
    f.add_argument("--recordings", required=True, help="the JSON written by `sample`")
    f.add_argument("--cohorts", nargs="*", default=[], help="external test cohorts cached under --root")
    f.add_argument("--reference", default=None, help="encoder the others are compared with (default: the first)")
    f.add_argument("--min-age", type=float, default=18.0, help="external cohorts: adults only")
    f.add_argument("--features", nargs="+", choices=FEATURES, default=list(FEATURES),
                   help="recording feature sets (mean_std doubles the width)")
    f.add_argument("--out", default=None, help="output prefix")
    args = ap.parse_args()
    if args.cmd == "sample":
        with open(os.path.join(args.cohort, "manifest.json")) as fh:
            manifest = json.load(fh)
        with open(args.cv_split) as fh:
            cv_split = json.load(fh)
        res = sample_recordings(manifest, cv_split, args.fold, args.n_train, args.n_val, args.seed)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(res, fh)
        print({f"probe_{k}": len(v) for k, v in res["probe"].items()},
              {f"cache_{k}": len(v) for k, v in res["cache"].items()})
        return
    with open(args.recordings) as fh:
        recordings = json.load(fh)
    table, paired = run_fit(args.root, args.encoders, recordings, args.cohorts, args.main,
                            args.reference, args.min_age, args.out, args.features)
    wide = table.pivot_table(index=["encoder", "features"], columns="group", values="mae", sort=False)
    print("MAE (years)\n" + wide.round(2).to_string())
    if len(paired):
        print("\npaired dMAE vs reference (a - b; negative = a better)\n"
              + paired.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
