#!/usr/bin/env python
"""Extend a cohort with more training recordings, keeping its split intact.

The base cohort (``make_cohort_dataset.py``) stays exactly as it is: every
recording keeps its split, every val / test file list is unchanged, and so
every number on val or test stays comparable. Added are the recordings the
base selection left out only because of their missing channels: with
``--allow-missing`` (default: every channel) a recording may lack any channels
(zero-filled in the npy; ``channel_drop`` leaves them out per sample). Of
those, only recordings of **training** patients are added -- the patient's
split in the base cohort, else in the base's ``patient_split`` /
``tuab_split`` -- so no val or test patient gains a recording, and the
patient-disjointness check of the base still holds. Per patient the base's
``max_per_subject`` cap applies to base + added training recordings (base
recordings are never removed; added ones fill the free slots, seeded).

The base's selection parameters (ages, minimum length, train minutes, trim)
are reused, so an added recording is selected exactly as a base recording
would have been. Writes ``<out>`` like ``make_cohort_dataset.py`` (manifest,
age_metadata, age_split, cohort.parquet, recordings/ as hard links) and
``added_recordings.json`` (stem, subject, missing channels).

    python -m dataset_maker.extend_cohort --base .../cohorts/all_both_v3 \\
        --source .../tueg/v2.0.2/processed_npy_interp50-60_float16 \\
        --out .../cohorts/all_both_v3_anymiss

``--compact-base`` / ``--compact-out`` also write the extension of the base's
compact copy (``make_compact_cohort.py``): the base's compact rows, files and
recordings unchanged, plus the added recordings as **full** rows (no
``window_index``; the npy loader resolves compact vs full per row) with their
listed training files appended to the age split. Only the added recordings
are new, so the compact copy on S3 extends by a server-side copy of the base
copy plus the added recordings.
"""
import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter

import pandas as pd

from dataset_maker import make_cohort_dataset as mc
from dataset_maker import tuh_clinical as tc

ALL_CHANNELS = ("FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4",
                "T5", "T6", "A1", "A2", "FZ", "CZ", "PZ", "T1", "T2")


def extra_train_recordings(j, base_rows, cohort, allow_missing, tuab_split, patient_split, seed):
    """Rows of ``j`` (index joined with the clinical sheet) to add to the
    base's training split, with the reason each other candidate is skipped."""
    pool, _ = mc.select(j, cohort["experiment"], min_age=cohort["min_age"], max_age=cohort["max_age"],
                        min_minutes=cohort["min_minutes"], allow_missing=allow_missing)
    base_split = {r["stem"]: r["source_split"] for r in base_rows}
    subj_split = {}
    for r in base_rows:
        subj_split.setdefault(r["stem"].split("_")[0], set()).add(r["source_split"])
    cand = j[pool & ~j.stem.isin(base_split)].copy()

    def patient_role(s):
        if s in subj_split:                          # the base already places this patient
            roles = subj_split[s]
            return next(iter(roles)) if len(roles) == 1 else "mixed"
        if s in tuab_split:
            return tuab_split[s]
        return patient_split.get(s, "unassigned")
    cand["role"] = cand.subject.map(patient_role)
    skipped = Counter(cand.loc[cand.role != "train", "role"])
    cand = cand[cand.role == "train"]
    if cohort["experiment"] == "normal_tueg":
        cand = cand[~cand.subject.isin(tuab_split)]
    have = Counter(r["stem"].split("_")[0] for r in base_rows if r["source_split"] == "train")
    rng = random.Random(seed)
    keep = []
    for subj, g in cand.groupby("subject", sort=True):
        idx = list(g.index)
        rng.shuffle(idx)
        free = max(0, cohort["max_per_subject"] - have.get(subj, 0))
        keep += idx[:free]
        if len(idx) > free:
            skipped["per-patient cap"] += len(idx) - free
    return cand.loc[sorted(keep)], skipped


def extend(args):
    base = json.load(open(os.path.join(args.base, "manifest.json")))
    cohort = base["cohort"]
    if "compact" in base:
        raise SystemExit("--base must be a full cohort, not a compact copy")
    split = json.load(open(os.path.join(args.base, "age_split.json")))
    meta = json.load(open(os.path.join(args.base, "age_metadata.json")))["recordings"]
    src_rows = {r["stem"]: r for r in json.load(open(os.path.join(args.source, "manifest.json")))["recordings"]}
    j = tc.join_index(pd.read_parquet(args.index), tc.load_sheet(args.sheet))
    j = j[j.stem.isin(src_rows)].reset_index(drop=True)
    tuab_split = mc.tuab_subject_split(args.tuab_split)
    patient_split = json.load(open(cohort["patient_split"]))["subjects"] if cohort.get("patient_split") else {}
    add, skipped = extra_train_recordings(j, base["recordings"], cohort, args.allow_missing, tuab_split,
                                          patient_split, args.seed)
    trim = args.trim_sec // mc.WINDOW_SEC
    train_windows = cohort["train_minutes"] * 60 // mc.WINDOW_SEC
    rows, files, added = list(base["recordings"]), {k: list(v) for k, v in split["files"].items()}, []
    for r in add.itertuples():
        row = dict(src_rows[r.stem])
        row["source_split"] = "train"
        rows.append(row)
        files["train"] += mc.window_files(r.stem, row["n_windows"], "train",
                                          train_windows=train_windows, trim_windows=trim)
        meta[r.stem] = {"stem": r.stem, "subject": r.subject, "session": r.session, "token": r.token,
                        "age": int(r.age), "sex": r.sex, "year": None if pd.isna(r.year) else int(r.year),
                        "raw_age": None if pd.isna(r.raw_age) else int(r.raw_age),
                        "label": mc.label_of(r._asdict())}
        added.append({"stem": r.stem, "subject": r.subject, "missing_channels": sorted(r.missing_channels)})
    subjects = {s: {r["stem"].split("_")[0] for r in rows if r["source_split"] == s} for s in files}
    for a in files:
        for b in files:
            if a < b and subjects[a] & subjects[b]:
                raise AssertionError(f"patients in both {a} and {b}: {sorted(subjects[a] & subjects[b])[:5]}")
    for s in ("val", "test"):
        assert files[s] == split["files"][s], f"{s} changed"
    counts = {s: {"n_windows": len(files[s]), "n_recordings": sum(r["source_split"] == s for r in rows),
                  "n_subjects": len(subjects[s])} for s in files}
    manifest = {**{k: v for k, v in base.items() if k != "recordings"},
                "n_recordings": len(rows), "n_windows": sum(r["n_windows"] for r in rows),
                "cohort": {**cohort, "counts": counts,
                           "extended_from": {"base": os.path.abspath(args.base),
                                             "allow_missing": list(args.allow_missing), "seed": args.seed,
                                             "added_train_recordings": len(added),
                                             "skipped": dict(skipped)}},
                "recordings": rows}
    print(f"added {len(added)} training recordings of {len({a['subject'] for a in added})} patients; "
          f"skipped {dict(skipped)}")
    print("missing-channel patterns:", Counter(tuple(a["missing_channels"]) for a in added).most_common(6))
    print("counts:", counts)
    if args.dry_run:
        return
    out = args.out
    os.makedirs(os.path.join(out, "recordings"), exist_ok=True)
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    json.dump({"version": 1, "recordings": meta}, open(os.path.join(out, "age_metadata.json"), "w"))
    json.dump({**split, "counts": counts, "files": files}, open(os.path.join(out, "age_split.json"), "w"))
    json.dump(added, open(os.path.join(out, "added_recordings.json"), "w"), indent=1)
    base_parquet = os.path.join(args.base, "cohort.parquet")
    if os.path.exists(base_parquet):
        c = pd.read_parquet(base_parquet)
        new = add.assign(split="train", source_split="train")[list(c.columns)]
        pd.concat([c, new], ignore_index=True).to_parquet(os.path.join(out, "cohort.parquet"), index=False)
    base_stems = {r["stem"] for r in base["recordings"]}
    for r in rows:
        dst = os.path.join(out, "recordings", f"{r['stem']}.npy")
        if not os.path.exists(dst):              # hard links: no extra space on the same filesystem
            src = os.path.join(args.base if r["stem"] in base_stems else args.source,
                               "recordings", f"{r['stem']}.npy")
            os.link(os.path.realpath(src), dst)


def extend_compact(compact_base, extended, out):
    """The compact copy of ``extended`` = ``compact_base`` + the rows that
    ``extended`` added (see the module doc). Recordings are hard links."""
    cb = json.load(open(os.path.join(compact_base, "manifest.json")))
    ex = json.load(open(os.path.join(extended, "manifest.json")))
    added = {a["stem"] for a in json.load(open(os.path.join(extended, "added_recordings.json")))}
    have = {r["stem"] for r in cb["recordings"]}
    assert not added & have, "added recordings already in the compact base"
    new_rows = [dict(r) for r in ex["recordings"] if r["stem"] in added]
    rows = cb["recordings"] + new_rows
    manifest = {**{k: v for k, v in cb.items() if k != "recordings"},
                "n_recordings": len(rows), "n_windows": sum(r["n_windows"] for r in rows),
                "n_stored_windows": cb.get("n_stored_windows", 0) + sum(r["n_windows"] for r in new_rows),
                "cohort": ex["cohort"],
                "compact": {**cb["compact"], "extended_with_full_rows": len(new_rows),
                            "extended_from": os.path.abspath(extended)},
                "recordings": rows}
    split = json.load(open(os.path.join(compact_base, "age_split.json")))
    ex_split = json.load(open(os.path.join(extended, "age_split.json")))
    new_train = [f for f in ex_split["files"]["train"] if os.path.basename(f).rsplit("_", 1)[0] in added]
    files = {k: list(v) for k, v in split["files"].items()}
    files["train"] += new_train
    os.makedirs(os.path.join(out, "recordings"), exist_ok=True)
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"))
    json.dump({**split, "counts": ex_split.get("counts"), "files": files},
              open(os.path.join(out, "age_split.json"), "w"))
    for name in ("age_metadata.json", "cohort.parquet", "added_recordings.json") + tuple(
            f for f in os.listdir(extended) if f.endswith(".npz") and f != "window_stats.npz"):
        if os.path.exists(os.path.join(extended, name)):
            shutil.copy2(os.path.join(extended, name), os.path.join(out, name))
    for r in rows:
        dst = os.path.join(out, "recordings", f"{r['stem']}.npy")
        if not os.path.exists(dst):
            src = os.path.join(extended if r["stem"] in added else compact_base, "recordings", f"{r['stem']}.npy")
            os.link(os.path.realpath(src), dst)
    print(f"{out}: {len(cb['recordings'])} compact + {len(new_rows)} full rows; train files "
          f"{len(split['files']['train'])} + {len(new_train)}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="full cohort dir (make_cohort_dataset.py output)")
    p.add_argument("--source", required=True, help="npy dir with manifest.json the base was built from")
    p.add_argument("--out", required=True)
    p.add_argument("--allow-missing", nargs="*", default=list(ALL_CHANNELS), metavar="CHANNEL",
                   help="channels an added recording may lack (default: any)")
    p.add_argument("--index", default=mc.DEFAULT_INDEX)
    p.add_argument("--sheet", default=mc.DEFAULT_SHEET)
    p.add_argument("--tuab-split", default=mc.DEFAULT_TUAB_SPLIT)
    p.add_argument("--trim-sec", type=int, default=60, help="must match the base (data.trim_*_sec)")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--compact-base", default=None, help="the base's compact copy (make_compact_cohort.py)")
    p.add_argument("--compact-out", default=None, help="where to write its extension")
    args = p.parse_args(argv)
    extend(args)
    if args.compact_base and not args.dry_run:
        extend_compact(args.compact_base, args.out, args.compact_out)


if __name__ == "__main__":
    sys.exit(main())
