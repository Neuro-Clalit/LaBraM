#!/usr/bin/env python
"""Download Synapse-hosted EEG corpora for the external brain-age evaluation.

Two datasets (see ``docs/chbp_onboarding.md``):

* ``chbp`` -- Cuban Human Brain Mapping Project (Valdes-Sosa et al. 2021,
  ``syn22324937``). Raw resting EEG (BIDS, EDF) of healthy adults 18-68. Two
  releases on Synapse: ``v1.0`` (``ds_bids_chbmp``, 2020, 282 subjects, the one
  Engemann et al. 2022 benchmarked) and ``v1.1`` (``ds_bids_cbm_loris_24_11_21``,
  279 subjects, EEG already filtered 0.5-50 Hz + 60 Hz notch, adds channel
  status, electrode positions and artifact annotations under ``derivatives``).
  Only the ``eeg`` folders are taken; the MRI folders (``anat``, ``dwi``,
  ``fmap``) are skipped.
* ``mncs`` -- HarMNqEEG multinational norms (Li et al. 2022, ``syn26712693``):
  cross-spectra only (.mat), no raw signal; a spectral-baseline test set.

The token comes from ``SYNAPSE_AUTH_TOKEN`` or ``~/.synapseConfig`` (never in
code). It needs the ``view`` and ``download`` scopes. A 403 means the terms on
the dataset's Synapse page have not been accepted.

Resumable: files already present with the listed size are skipped. ``--list``
only writes ``<out>/listing.jsonl`` (path, Synapse id, size, md5) and prints the
totals.

    python -m dataset_maker.download_synapse chbp --list
    python -m dataset_maker.download_synapse chbp --out /data/datasets/EEG-public/CHBP/raw --jobs 8
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time

# name -> {part: (Synapse id, local sub-folder)}
DATASETS = {
    "chbp": {
        "meta": ("syn22324942", "meta/cognitive_scales"),
        "electrodes": ("syn22324953", "meta/electrode_positions"),
        "v1.0": ("syn22324960", "bids_v1.0"),
        "v1.1": ("syn28129311", "bids_v1.1"),
    },
    "mncs": {"spectra": ("syn26718114", "ShareRawData")},
}
SKIP_FOLDERS = {"anat", "dwi", "fmap", "func"}


def login():
    import synapseclient
    syn = synapseclient.Synapse(skip_checks=True, silent=True)
    token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    syn.login(authToken=token, silent=True) if token else syn.login(silent=True)
    return syn


def walk(syn, syn_id, rel=""):
    """Yield {path, id, size, md5} for every file under ``syn_id``, skipping MRI folders."""
    for child in syn.getChildren(syn_id, includeTypes=["folder", "file"]):
        path = f"{rel}/{child['name']}" if rel else child["name"]
        if child["type"].endswith("Folder"):
            if child["name"] not in SKIP_FOLDERS:
                yield from walk(syn, child["id"], path)
        else:
            yield {"path": path, "id": child["id"]}


def file_handle(syn, row):
    ent = syn.get(row["id"], downloadFile=False)
    fh = ent.get("_file_handle", {}) or {}
    return {**row, "size": fh.get("contentSize"), "md5": fh.get("contentMd5"), "version": ent.versionNumber}


def build_listing(syn, name, parts, jobs):
    rows = []
    for part in parts:
        syn_id, sub = DATASETS[name][part]
        files = [{**r, "path": f"{sub}/{r['path']}", "part": part} for r in walk(syn, syn_id)]
        print(f"  {part} ({syn_id}): {len(files)} files", flush=True)
        with cf.ThreadPoolExecutor(jobs) as pool:
            rows += list(pool.map(lambda r: file_handle(syn, r), files))
    return rows


def download_one(syn, row, out):
    dest = os.path.join(out, row["path"])
    if os.path.exists(dest) and os.path.getsize(dest) == row["size"]:
        return "skip"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(4):
        try:
            ent = syn.get(row["id"], downloadLocation=os.path.dirname(dest), ifcollision="overwrite.local")
            if ent.path != dest:                 # Synapse keeps the stored name; ours is the listed one
                os.replace(ent.path, dest)
            return "ok"
        except Exception as exc:                 # transient HTTP errors: back off and retry
            if attempt == 3 or "403" in str(exc) or "401" in str(exc):
                return f"error: {type(exc).__name__}: {str(exc)[:200]}"
            time.sleep(2 ** (attempt + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset", choices=sorted(DATASETS))
    ap.add_argument("--parts", nargs="*", help="sub-sets to take (default: all of the dataset)")
    ap.add_argument("--out", default=None, help="default /data/datasets/EEG-public/<DATASET>/raw")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--list", action="store_true", help="only write listing.jsonl and print totals")
    args = ap.parse_args()
    parts = args.parts or list(DATASETS[args.dataset])
    out = args.out or f"/data/datasets/EEG-public/{args.dataset.upper()}/raw"
    os.makedirs(out, exist_ok=True)
    syn = login()
    print(f"logged in as {syn.getUserProfile()['userName']}; {args.dataset} {parts} -> {out}", flush=True)

    rows = build_listing(syn, args.dataset, parts, args.jobs)
    with open(os.path.join(out, "listing.jsonl"), "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    for part in parts:
        sel = [r for r in rows if r["part"] == part]
        exts = {}
        for r in sel:
            ext = os.path.splitext(r["path"])[1].lower() or "(none)"
            exts[ext] = exts.get(ext, 0) + 1
        print(f"  {part}: {len(sel)} files, {sum(r['size'] or 0 for r in sel) / 1e9:.2f} GB, {exts}")
    if args.list:
        return 0

    done = {"ok": 0, "skip": 0}
    errors = []
    with cf.ThreadPoolExecutor(args.jobs) as pool:
        futs = {pool.submit(download_one, syn, r, out): r for r in rows}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            res = fut.result()
            if res in done:
                done[res] += 1
            else:
                errors.append({"path": futs[fut]["path"], "error": res})
            if i % 100 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}: {done['ok']} downloaded, {done['skip']} present, "
                      f"{len(errors)} errors", flush=True)
    with open(os.path.join(out, "download_errors.json"), "w") as fh:
        json.dump(errors, fh, indent=1)
    for e in errors[:10]:
        print("  ", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
