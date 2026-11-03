#!/usr/bin/env python
"""Join cohorts of several corpora (TUH + HEEDB) into one training cohort in S3.

Each ``--part NAME=SIDECARS,RECORDINGS`` names a cohort: ``SIDECARS`` is its
dir (local or ``s3://``) with ``manifest.json``, ``age_metadata.json`` and
``age_split.json`` (``cohort.parquet`` optional), and ``RECORDINGS`` the
``s3://`` prefix its ``<stem>.npy`` files are copied from (server-side), or,
for a local ``--out``, the local dir they are hard-linked from (no extra
space; local training on /data). The output has the
TUH cohort layout (one prefix: sidecars + ``recordings/``), so the TUAB_AGE npy
loader and SageMaker read it like any cohort. Each part keeps its own patient
split: TUH keeps its TUAB-eval test set, HEEDB its held-out test patients.

Checks: every part has the same channels, sample rate, window length and
units; no recording stem or patient id appears in two parts (patients of
different corpora cannot be the same person, but their ids must not collide in
the loader's subject-disjointness checks). Every manifest row gets ``corpus`` = NAME.

    python -m dataset_maker.make_joint_cohort \\
        --part tuh=/data/datasets/EEG-public/TUH_EEG/cohorts/all_both_v3_anymiss,s3://eeg-data-public/TUH_EEG/tueg/v2.0.2/processed_npy_interp50-60_float16/recordings \\
        --part heedb=s3://eeg-data-public/HEEDB/v4.1/cohorts/heedb_routine_v1,s3://eeg-data-public/HEEDB/v4.1/cohorts/heedb_routine_v1/recordings \\
        --out s3://eeg-data-public/HEEDB/v4.1/cohorts/tuh_heedb_v1
    # local, hard links:
    python -m dataset_maker.make_joint_cohort \\
        --part tuh=/data/datasets/EEG-public/TUH_EEG/cohorts/all_both_v3_anymiss,/data/datasets/EEG-public/TUH_EEG/cohorts/all_both_v3_anymiss/recordings \\
        --part heedb=/data/datasets/HEEDB/v4.1/cohorts/heedb_routine_v1,/data/datasets/HEEDB/v4.1/cohorts/heedb_routine_v1/recordings \\
        --out /data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1
"""
import argparse
import concurrent.futures as cf
import io
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataset_maker import heedb_index as hx  # noqa: E402
from dataset_maker.make_heedb_cohort import split_s3  # noqa: E402

SAME = ("channels", "sample_rate", "window_samples", "units")


def _reader(client, root):
    def read(name):
        if root.startswith("s3://"):
            b, p = split_s3(root)
            try:
                return client.get_object(Bucket=b, Key=f"{p}/{name}")["Body"].read()
            except client.exceptions.NoSuchKey:
                return None
        path = os.path.join(root, name)
        return open(path, "rb").read() if os.path.exists(path) else None
    return read


def join(parts):
    """``parts``: list of (name, manifest, meta, split, cohort-or-None) ->
    (manifest, meta, split, cohort table)."""
    ref = parts[0][1]
    rows, meta, files, tables, owner = [], {}, {}, [], {}
    patients = {}
    for name, man, md, sp, coh in parts:
        for k in SAME:
            if man.get(k) != ref.get(k):
                raise ValueError(f"{name}: {k} {man.get(k)!r} differs from {parts[0][0]}'s {ref.get(k)!r}")
        for r in man["recordings"]:
            if r["stem"] in owner:
                raise ValueError(f"stem {r['stem']} in both {owner[r['stem']]} and {name}")
            owner[r["stem"]] = name
            rows.append({**r, "corpus": name})
        subs = {m.get("subject") for m in md["recordings"].values()}
        for other, s in patients.items():
            if subs & s:
                raise ValueError(f"patient ids shared by {other} and {name}: {sorted(subs & s)[:3]}")
        patients[name] = subs
        meta.update(md["recordings"])
        for split, names in sp["files"].items():
            files.setdefault(split, []).extend(names)
        if coh is not None:
            tables.append(coh.assign(corpus=name))
    manifest = {**{k: v for k, v in ref.items() if k not in ("recordings", "n_recordings", "n_windows", "cohort")},
                "n_recordings": len(rows), "n_windows": sum(r["n_windows"] for r in rows),
                "line_removal": sorted({str(p[1].get("line_removal")) for p in parts}),
                "joint": {name: {"recordings": len(man["recordings"]),
                                 "cohort": man.get("cohort", {}).get("experiment")}
                          for name, man, *_ in parts},
                "recordings": rows}
    split = {"version": 1, "files": files,
             "counts": {s: {"n_windows": len(f)} for s, f in files.items()}}
    cohort = pd.concat(tables, ignore_index=True) if tables else None
    return manifest, {"version": 1, "recordings": meta}, split, cohort


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--part", action="append", required=True, metavar="NAME=SIDECARS,RECORDINGS")
    p.add_argument("--out", required=True, help="output cohort dir: s3:// (copies) or local (hard links)")
    p.add_argument("--threads", type=int, default=128)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    from dataset_maker.download_heedb import check_private
    client = hx.s3_client(None)
    local = not args.out.startswith("s3://")
    if not local:
        ob, op = split_s3(args.out)
        check_private(client, ob)
    parts, sources = [], {}
    for spec in args.part:
        name, rest = spec.split("=", 1)
        side, recs = rest.split(",", 1)
        read = _reader(client, side)
        coh = read("cohort.parquet")
        parts.append((name, json.loads(read("manifest.json")), json.loads(read("age_metadata.json")),
                      json.loads(read("age_split.json")),
                      pd.read_parquet(io.BytesIO(coh)) if coh else None))
        sources[name] = recs.rstrip("/")
    manifest, meta, split, cohort = join(parts)
    print(json.dumps({"recordings": manifest["n_recordings"], "parts": manifest["joint"],
                      "files": {s: len(f) for s, f in split["files"].items()}}, indent=1))
    if args.dry_run:
        return

    def put(name, data):
        if local:
            os.makedirs(args.out, exist_ok=True)
            with open(os.path.join(args.out, name), "wb") as fh:
                fh.write(data)
        else:
            client.put_object(Bucket=ob, Key=f"{op}/{name}", Body=data)

    put("manifest.json", json.dumps(manifest).encode())
    put("age_metadata.json", json.dumps(meta).encode())
    put("age_split.json", json.dumps(split).encode())
    if cohort is not None:
        buf = io.BytesIO()
        cohort.astype({c: "string" for c in cohort.columns if cohort[c].dtype == object}).to_parquet(buf, index=False)
        put("cohort.parquet", buf.getvalue())

    def link(row):
        src = os.path.join(sources[row["corpus"]], f"{row['stem']}.npy")
        dst = os.path.join(args.out, "recordings", f"{row['stem']}.npy")
        if os.path.lexists(dst):
            return 1
        try:
            os.link(os.path.realpath(src), dst)
        except OSError:                    # another filesystem: symlink instead
            os.symlink(os.path.realpath(src), dst)
        return 1

    def copy(row):
        if local:
            return link(row)
        sb, sp = split_s3(sources[row["corpus"]])
        client.copy_object(Bucket=ob, Key=f"{op}/recordings/{row['stem']}.npy",
                           CopySource={"Bucket": sb, "Key": f"{sp}/{row['stem']}.npy"})
        return 1

    if local:
        os.makedirs(os.path.join(args.out, "recordings"), exist_ok=True)
    with cf.ThreadPoolExecutor(args.threads) as pool:
        n = sum(pool.map(copy, manifest["recordings"]))
    print(f"wrote {args.out}: sidecars + {n} recordings {'linked' if local else 'copied'}")


if __name__ == "__main__":
    sys.exit(main())
