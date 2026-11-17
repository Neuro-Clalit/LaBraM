#!/usr/bin/env python
"""Convert external resting-state EEG corpora to an evaluation cohort in the TUH npy format.

The signal pipeline is ``make_TUH_npy.preprocess_raw``, the one every training
cohort went through (0.1-75 Hz band-pass, spectrum-interpolation line removal at
50 and 60 Hz, 200 Hz, µV, float16, the 23 TUH channels in model order with
missing ones zero and flagged in ``channel_mask``, whole 10 s windows). Only the
reader and the channel naming differ per corpus:

* ``lemon`` -- MPI-LEMON (Babayan et al. 2019), BrainVision, 61 EEG channels at
  2500 Hz, FCz reference (kept: the models apply a common average reference at
  their input), 16 min of alternating 1-min eyes-closed / eyes-open blocks. Ages
  are only published as 5-year bins ("20-25"); ``age`` is the bin midpoint and
  ``raw_age`` the bin.
* ``mipdb`` -- CMI MIPDB (Langer et al. 2017), EGI HydroCel 128 at 500 Hz, the
  resting-state file (trigger 90, alternating eyes open / closed). The
  recording reference is the vertex (Cz = E129/VREF, all zero), so the 19
  mapped channels are re-referenced to their mean on reading
  (``reference="average"``); the models' input CAR makes this equivalent to any
  other reference. Ages are whole years from ``MIPDB_PublicFile.csv``.
* ``chbp`` -- Cuban Human Brain Mapping Project (Valdes-Sosa et al. 2021,
  Synapse ``syn22324937``, ``download_synapse.py``), BIDS EDF at 200 Hz,
  recorded 0.5-50 Hz with a 60 Hz notch, 120 EEG channels (10-5). Only the
  resting part (eyes closed, then eyes open / closed) up to the first
  hyperventilation event is kept. Ages are whole years from
  ``Demographic_data.csv``.

* ``tdbrain`` -- TDBRAIN (van Dijk et al. 2022, brainclinics.com/resources,
  ORCID login + DUA), BIDS BrainVision at 500 Hz, DC-coupled, 26 EEG channels
  (10-10) referenced to averaged mastoids, 2 min eyes closed + 2 min eyes open
  per session, each converted as its own recording. Psychiatric patients
  (MDD, ADHD, SMC, OCD, ...) and healthy controls, 5-88 years, decimal ages and
  clinical flags from ``participants.tsv``; ``clinical.parquet`` keeps every
  participants.tsv column. See ``docs/tdbrain_onboarding.md``.

* ``dortmund`` -- Dortmund Vital Study (Wascher et al., OpenNeuro ds005385,
  CC0), BIDS EDF at 1000 Hz, DC-coupled (low-pass 250 Hz), 64 EEG channels
  (actiCAP, 10-10) referenced to FCz, healthy adults of 20-70 years. Four
  3-min runs per session (eyes closed and eyes open, before and after a 2-h
  block of cognitive tasks), each converted as its own recording; 208 of the
  608 participants have a second session about 5 years later. Whole-year ages
  from ``participants.tsv`` (the age at session 1; session 2 adds the
  difference of the recording years). See ``docs/dortmund_onboarding.md``.

* ``ahepa`` -- AHEPA dementia dataset (Miltiadous et al. 2023, OpenNeuro
  ds004504, CC0), EEGLAB ``.set`` at 500 Hz from a Nihon Kohden EEG-2100, the
  19 10-20 channels, eyes closed, 5-21 min; Alzheimer's disease,
  frontotemporal dementia and healthy controls, with MMSE.
* ``iowapd`` -- Iowa Parkinson's rest dataset (Singh et al., OpenNeuro
  ds004584, CC0), EEGLAB ``.set`` at 500 Hz, 63-66 channels (BrainVision,
  10-10) referenced to Pz, eyes open, 2-6 min; Parkinson's disease and
  controls, with MoCA and UPDRS. Pz is the reference and so not recorded: it is
  added as a zero channel and the 19 channels are re-referenced to their mean.
* ``telavivpd`` -- Tel Aviv PD-EEG (OpenNeuro ds007526, CC0), EEGLAB ``.set``
  at 250 Hz, 64-channel EGI net with 10-10 labels, the resting run (eyes open,
  about 4 min; the walking run is not converted); Parkinson's disease and
  controls. The vertex reference ``VREF`` is Cz (all zero): renamed, and the 19
  channels are re-referenced to their mean. A run with a break in continuity
  is cut to its longest continuous piece.
* ``srm`` -- SRM resting-state EEG (Hatlestad-Hall et al., OpenNeuro ds003775,
  CC0), EDF at 1024 Hz, 64 channels (BioSemi, 10-10), released average-
  referenced, 4 min eyes closed; healthy, 17-71 years, a second session for 42
  of the 111 participants. The EDF headers carry no physical unit (the values
  are microvolts).

* ``nmt`` -- NMT Scalp EEG (Khan et al. 2022, Zenodo 10.5281/zenodo.21629135,
  CC BY 4.0), EDF at 200 Hz from a Contec KT88-2400, the 19 10-20 channels +
  A1 / A2, released average-referenced; routine clinical EEG of 2,417 patients
  of all ages (1,374 adults) with a normal / abnormal label. Two undocumented
  properties of the released files, found by the audit
  (``docs/open_cohorts_onboarding.md``): the signals are band-limited to about
  1-35 Hz, and the stored values are about a hundredth of real EEG amplitude
  although the header says µV. They are multiplied by ``NMT_SCALE`` = 100,
  which puts the cohort's median amplitude between Temple's and Harvard's; this
  is an assumption, and a cohort-level, label-free one. ``--meta`` is
  ``Labels.csv``.

The four OpenNeuro corpora take the BIDS ``participants.tsv`` as ``--meta``. In the three clinical
corpora the controls are labelled ``normal`` and ``healthy``; patients carry
``dx_*`` flags and stay unlabelled.

The healthy-volunteer corpora are labelled ``normal`` (MIPDB's ``DX_Status``
is kept in the manifest); in TDBRAIN only the healthy controls are, the
patients stay unlabelled. Every recording is placed in the **test** split, so a
model trained elsewhere scores the whole cohort
(``scripts/eval_age_by_cohort.py --splits test --pred-stats-from-log``).

    python -m dataset_maker.make_external_npy --dataset lemon \\
        --raw /data/datasets/EEG-public/LEMON/raw \\
        --meta /data/datasets/EEG-public/LEMON/meta/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv \\
        --out /data/datasets/EEG-public/LEMON/cohorts/lemon_v1 --jobs 6
"""
import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import sys

import numpy as np

try:
    from dataset_maker.make_TUAB_npy import npy_bytes
    from dataset_maker.make_TUH_npy import (FLOAT16_MAX, MIN_DURATION_SEC, SAMPLE_RATE,
                                            WINDOW_SAMPLES, preprocess_raw)
    from labram.data.tuh_metadata import TUH_CHANNELS, bids_canonical_channel, channel_map
except ImportError:      # run as a plain script from dataset_maker/
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dataset_maker.make_TUAB_npy import npy_bytes
    from dataset_maker.make_TUH_npy import (FLOAT16_MAX, MIN_DURATION_SEC, SAMPLE_RATE,
                                            WINDOW_SAMPLES, preprocess_raw)
    from labram.data.tuh_metadata import TUH_CHANNELS, bids_canonical_channel, channel_map

NOTCH = (50.0, 60.0)
LINE = "interp"
# EGI HydroCel GSN 128 electrodes closest to the 10-20 positions (Luu & Ferree
# 2005; the vertex reference E129 is Cz). A1/A2/T1/T2 have no counterpart here.
EGI_HYDROCEL_1020 = {
    "FP1": "E22", "FP2": "E9", "F7": "E33", "F3": "E24", "FZ": "E11", "F4": "E124", "F8": "E122",
    "T3": "E45", "C3": "E36", "CZ": "E129", "C4": "E104", "T4": "E108",
    "T5": "E58", "P3": "E52", "PZ": "E62", "P4": "E92", "T6": "E96", "O1": "E70", "O2": "E83",
}


# ------------------------------------------------------------------ readers
def lemon_recordings(raw_dir, meta_csv):
    import pandas as pd
    meta = pd.read_csv(meta_csv).rename(columns={"Unnamed: 0": "ID"})
    meta = meta.set_index(meta.columns[0])
    out = []
    for vhdr in sorted(glob.glob(os.path.join(raw_dir, "sub-*", "RSEEG", "*.vhdr"))):
        sub = os.path.basename(vhdr)[:-5]
        if sub not in meta.index:
            continue
        m = meta.loc[sub]
        lo, hi = (float(v) for v in str(m["Age"]).split("-"))
        sex = {1: "F", 2: "M"}.get(int(m["Gender_ 1=female_2=male"]), None)
        out.append({"stem": f"{sub}_lemon-rs", "path": os.path.relpath(vhdr, raw_dir), "corpus": "lemon",
                    "subject": sub, "session": "1", "age": (lo + hi) / 2, "raw_age": str(m["Age"]),
                    "sex": sex, "label": "normal", "_file": vhdr})
    return out


def mipdb_recordings(raw_dir, meta_csv):
    import pandas as pd
    meta = pd.read_csv(meta_csv).set_index("ID")
    out = []
    for f in sorted(glob.glob(os.path.join(raw_dir, "A*", "*.raw"))):
        sub = os.path.basename(os.path.dirname(f))
        if sub not in meta.index or pd.isna(meta.loc[sub, "Age"]):
            continue
        m = meta.loc[sub]
        dx = m.get("DX_Status")
        out.append({"stem": f"{sub}_mipdb-rs", "path": os.path.relpath(f, raw_dir), "corpus": "mipdb",
                    "subject": sub, "session": "1", "age": float(m["Age"]), "raw_age": float(m["Age"]),
                    "sex": {1: "M", 2: "F"}.get(int(m["Sex"]), None) if not pd.isna(m["Sex"]) else None,
                    "label": "normal", "dx_status": None if pd.isna(dx) else int(dx), "_file": f})
    return out


def open_lemon(path):
    """The published .vhdr files still name their data / marker files by the
    pre-BIDS ids (``sub-010002.eeg``); they are read through a temporary folder
    of symlinks carrying those names (the downloaded files stay untouched)."""
    import re
    import shutil
    import tempfile
    import mne
    text = open(path, encoding="utf-8", errors="replace").read()
    names = dict(re.findall(r"^(DataFile|MarkerFile)=(.+?)\s*$", text, re.M))
    base = path[:-5]
    tmp = tempfile.mkdtemp(prefix="lemon_")
    try:
        os.symlink(base + ".eeg", os.path.join(tmp, names.get("DataFile", os.path.basename(base) + ".eeg")))
        os.symlink(base + ".vmrk", os.path.join(tmp, names.get("MarkerFile", os.path.basename(base) + ".vmrk")))
        shutil.copy(path, os.path.join(tmp, "header.vhdr"))
        raw = mne.io.read_raw_brainvision(os.path.join(tmp, "header.vhdr"), preload=True, verbose="error")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return raw, channel_map(raw.ch_names, bids_canonical_channel), None


def open_mipdb(path):
    import mne
    raw = mne.io.read_raw_egi(path, preload=False, verbose="error")
    names = set(raw.ch_names)
    present = {}
    for tuh, egi in EGI_HYDROCEL_1020.items():
        for label in (egi, "VREF", "Vertex Reference") if egi == "E129" else (egi,):
            if label in names:
                present[tuh] = label
                break
    return raw, present, "average"


# CHBP protocol events (Spanish in release 1.0, English in 1.1) that end the resting part.
CHBP_NON_REST = ("hiperventilacion", "hyperventilation", "fotoestimulacion", "photic", "recuperacion", "recovery")


def chbp_recordings(raw_dir, meta_csv):
    import pandas as pd
    meta = pd.read_csv(meta_csv, skiprows=1, encoding="utf-8-sig").dropna(axis=1, how="all")
    meta = meta.dropna(subset=["Code"]).set_index("Code")
    out = []
    for edf in sorted(glob.glob(os.path.join(raw_dir, "sub-*", "eeg", "*_eeg.edf"))):
        code = os.path.basename(edf).split("_")[0][len("sub-"):]
        if code not in meta.index or pd.isna(meta.loc[code, "Age"]):
            continue
        m = meta.loc[code]
        out.append({"stem": f"sub-{code}_chbp-rs", "path": os.path.relpath(edf, raw_dir), "corpus": "chbp",
                    "subject": code, "session": "1", "age": float(m["Age"]), "raw_age": float(m["Age"]),
                    "sex": m["Gender"] if m["Gender"] in ("M", "F") else None, "label": "normal", "_file": edf})
    return out


def chbp_resting_end(edf):
    """Onset (s) of the first hyperventilation / recovery / photic event in the
    BIDS ``events.tsv`` next to ``edf`` (None if there is none)."""
    import csv
    events = edf[:-len("_eeg.edf")] + "_events.tsv"
    if not os.path.exists(events):
        return None
    with open(events, encoding="utf-8", errors="replace") as fh:
        onsets = [float(r["onset"]) for r in csv.DictReader(fh, delimiter="\t")
                  if any(k in r["trial_type"].lower() for k in CHBP_NON_REST)]
    return min(onsets) if onsets else None


def open_chbp(path):
    """EDF at 200 Hz, recorded 0.5-50 Hz with a 60 Hz notch, 120 EEG channels
    (10-5). Labels in release 1.0 carry a space and a reference suffix that
    varies (``'F3 -REF'``, ``'F3 -A12'`` for linked ears, a truncated ``'F4 -'``),
    so only the part before the hyphen is matched. Only the resting part is
    kept: eyes closed (~10 min) then alternating eyes open / closed, cut where
    hyperventilation starts."""
    import mne
    raw = mne.io.read_raw_edf(path, preload=False, verbose="error")
    end = chbp_resting_end(path)
    if end is not None and end < raw.times[-1]:
        raw.crop(0, end, include_tmax=False)
    present = channel_map(raw.ch_names, lambda label: bids_canonical_channel(label.split("-")[0].strip()))
    return raw, present, None


# ------------------------------------------------------------------ TDBRAIN
TDBRAIN_TASKS = ("restEC", "restEO")
# Indication keyword (upper case, substring) -> clinical flag. The first match of
# each token of the indication wins; a token matching none sets ``dx_other``.
TDBRAIN_INDICATIONS = (("HEALTHY", "healthy"), ("CONTROL", "healthy"), ("REPLICATION", "blinded"),
                       ("UNKNOWN", "unknown"), ("MDD", "dx_mdd"), ("DEPRESS", "dx_mdd"),
                       ("ADHD", "dx_adhd"), ("ADD", "dx_adhd"), ("OCD", "dx_ocd"), ("SMC", "dx_smc"))
TDBRAIN_DX = ("dx_mdd", "dx_adhd", "dx_ocd", "dx_smc", "dx_other")
TDBRAIN_PLACEHOLDER_AGE = 50.0          # signals-only conversion, see tdbrain_recordings
# gender 0 / 1 without a sidecar (release 3.1 table): 0 = female. In that table, adults coded
# 0 have a median height / weight of 168 cm / 65 kg against 183 cm / 79 kg for 1, and 0 is
# 47 % of the subjects (the paper: 620 female of 1,274 = 49 %).
TDBRAIN_GENDER_LEVELS = {"0": "female", "1": "male"}


def _column(df, *names, required=False):
    """The column of ``df`` matching one of ``names`` (case, spaces, ``_`` and
    ``/`` ignored), else None; ``required`` raises listing the columns found."""
    def key(s):
        return "".join(ch for ch in str(s).lower() if ch.isalnum())
    cols = {key(c): c for c in df.columns}
    for name in names:
        if key(name) in cols:
            return cols[key(name)]
    if required:
        raise KeyError(f"none of {names} in participants.tsv; columns: {list(df.columns)}")
    return None


def _number(value):
    """``'35,53'`` / ``'35.53'`` / 35.53 -> 35.53; blank or text -> NaN."""
    try:
        return float(str(value).strip().replace(",", "."))
    except ValueError:
        return float("nan")


def _bare_id(value):
    s = str(value).strip()
    return s[len("sub-"):] if s.startswith("sub-") else s


def tdbrain_indication_flags(indication):
    """``'MDD/ADHD'`` -> ``{'healthy': False, 'dx_mdd': True, 'dx_adhd': True, ...,
    'clinical_group': 'dx_mdd'}``. The group is ``healthy``, ``blinded`` (the
    REPLICATION set, diagnosis withheld by the authors), ``unknown`` (no
    indication; *not* healthy, per van Dijk et al. 2022) or the first diagnosis."""
    import re
    flags = {"healthy": False, "blinded": False, "unknown": False, **{d: False for d in TDBRAIN_DX}}
    text = "" if indication is None or str(indication).strip().lower() in ("", "nan", "n/a") else str(indication)
    tokens = [t.strip() for t in re.split(r"[/,;+&]|\bAND\b", text.upper()) if t.strip()]
    first = None
    for tok in tokens:
        flag = next((f for kw, f in TDBRAIN_INDICATIONS if kw in tok), "dx_other")
        flags[flag] = True
        first = first or flag
    if not tokens:
        flags["unknown"] = True
    group = ("healthy" if flags["healthy"] else "blinded" if flags["blinded"]
             else first if first in TDBRAIN_DX else "unknown")
    return {**flags, "clinical_group": group}


def tdbrain_clinical_table(participants_tsv):
    """One row per participants.tsv row with every original column kept and
    harmonised ones added: ``subject`` (id without ``sub-``), ``session``
    (str, None if the file has no session column), ``age`` (years, decimal
    comma accepted), ``sex`` (M / F from the ``participants.json`` levels
    next to the tsv, else ``TDBRAIN_GENDER_LEVELS`` for a 0 / 1 code; None
    when they cannot be decoded), ``indication``,
    ``formal_status`` (DSM diagnosis vs referral indication), ``disc_rep``
    (DISCOVERY / REPLICATION) and the flags of :func:`tdbrain_indication_flags`.
    TDBRAIN has no EEG report and no ICD / medication record (medication was
    allowed but not tracked); the clinical information is the indication."""
    import pandas as pd
    ext = os.path.splitext(participants_tsv)[1].lower()
    if ext in (".xlsx", ".xls"):          # the participants file may ship separately as a sheet
        df = pd.read_excel(participants_tsv, dtype=str).fillna("")
    else:                                 # .tsv, or .csv with , or ; (sniffed)
        df = pd.read_csv(participants_tsv, sep="\t" if ext == ".tsv" else None, engine="python",
                         dtype=str, keep_default_na=False)
    pid = _column(df, "participants_ID", "participant_id", "TDBRAIN_ID", required=True)
    age = _column(df, "age", required=True)
    ses = _column(df, "sessID", "session_id", "session", "ses")
    sex = _column(df, "gender", "sex")
    ind = _column(df, "indication")
    fstat = _column(df, "formal_status", "formal_dx", "formalstatus")
    drep = _column(df, "DISC/REP", "disc_rep", "discrep")
    levels = {}
    if sex and set(df[sex].str.strip()) <= {"0", "1", ""}:
        levels = dict(TDBRAIN_GENDER_LEVELS)       # overridden by a participants.json below
    sidecar = os.path.splitext(participants_tsv)[0] + ".json"
    if sex and os.path.exists(sidecar):
        desc = json.load(open(sidecar, encoding="utf-8", errors="replace")).get(sex, {})
        levels = {str(k).strip(): str(v) for k, v in (desc.get("Levels") or {}).items()}

    def decode_sex(v):
        word = levels.get(str(v).strip(), str(v)).strip().lower()
        return {"m": "M", "male": "M", "man": "M", "f": "F", "female": "F", "woman": "F"}.get(word)

    out = df.copy()
    out["subject"] = df[pid].map(_bare_id)
    out["session"] = df[ses].map(lambda v: str(v).strip().removeprefix("ses-") or None) if ses else None
    out["age"] = df[age].map(_number)
    out["raw_age"] = df[age]
    out["sex"] = df[sex].map(decode_sex) if sex else None
    out["indication"] = df[ind].str.strip() if ind else None
    out["formal_status"] = df[fstat].str.strip().str.upper() if fstat else None
    out["disc_rep"] = df[drep].str.strip().str.upper() if drep else None
    flags = pd.DataFrame([tdbrain_indication_flags(v) for v in (out["indication"] if ind else [None] * len(df))],
                         index=df.index)
    # sub-19* is the blinded replication set even where the indication column says otherwise
    blinded = out["subject"].str.startswith("19") | (out["disc_rep"] == "REPLICATION")
    flags.loc[blinded & ~flags["healthy"], ["blinded", "clinical_group"]] = [True, "blinded"]
    return pd.concat([out, flags], axis=1)


def tdbrain_eeg_files(raw_dir):
    """``(subject, session, task, path)`` of every BIDS resting file under
    ``raw_dir`` (``sub-*/ses-*/eeg/*_task-rest{EC,EO}_eeg.{bdf,vhdr}``: BDF in
    release 3.1, BrainVision in the first release); the ``derivatives`` copies
    (csv) and the oddball task are skipped."""
    import re
    pat = re.compile(r"^sub-([^_]+)_ses-([^_]+)_task-(rest(?:EC|EO))_eeg\.(?:bdf|vhdr)$", re.I)
    out = []
    for dirpath, dirnames, filenames in os.walk(raw_dir):
        dirnames[:] = sorted(d for d in dirnames if d.lower() != "derivatives")
        for name in sorted(filenames):
            m = pat.match(name)
            if m:
                task = "restEC" if m[3].upper().endswith("EC") else "restEO"
                out.append((m[1], m[2], task, os.path.join(dirpath, name)))
    return out


def tdbrain_recordings(raw_dir, meta_tsv):
    """One recording per (subject, session, task): the 2-min eyes-closed and
    eyes-open files are converted separately (stem
    ``sub-<id>_ses-<n>_<task>_tdbrain``) and scored per file or pooled per
    session. Age, sex and the clinical flags come from the participants.tsv row
    of that session (or of the subject when the file lists no session).
    ``label`` is ``normal`` for healthy participants only; patients stay
    unlabelled (no EEG report exists, a psychiatric indication is not an EEG
    finding).

    Without a participants table (``meta_tsv`` None: release 3.1 ships the EEG
    without one) every file is listed with the placeholder age
    ``TDBRAIN_PLACEHOLDER_AGE`` and ``age_placeholder: True``, so the signals can
    be converted and scored before the table arrives; ``--relabel`` then writes
    the real ages without converting again. Metrics on placeholder ages mean
    nothing."""
    if meta_tsv is None:
        return [{"stem": f"sub-{sub}_ses-{ses}_{task}_tdbrain", "path": os.path.relpath(path, raw_dir),
                 "corpus": "tdbrain", "subject": sub, "session": ses, "task": task,
                 "age": TDBRAIN_PLACEHOLDER_AGE, "raw_age": None, "sex": None, "label": None,
                 "age_placeholder": True, "_file": path}
                for sub, ses, task, path in tdbrain_eeg_files(raw_dir)]
    table = tdbrain_clinical_table(meta_tsv)
    by_session = {(r.subject, r.session): r for r in table.itertuples()} if table["session"].notna().any() else {}
    by_subject = {}
    for r in table.itertuples():
        by_subject.setdefault(r.subject, r)
    keep = ("indication", "formal_status", "disc_rep", "healthy", "blinded", "unknown", *TDBRAIN_DX,
            "clinical_group")
    out = []
    for sub, ses, task, path in tdbrain_eeg_files(raw_dir):
        row = by_session.get((sub, ses)) or by_session.get((sub, ses.lstrip("0") or "0"))
        matched = row is not None
        row = row if matched else by_subject.get(sub)
        if row is None or not np.isfinite(row.age):
            continue
        out.append({"stem": f"sub-{sub}_ses-{ses}_{task}_tdbrain", "path": os.path.relpath(path, raw_dir),
                    "corpus": "tdbrain", "subject": sub, "session": ses, "task": task,
                    "age": float(row.age), "raw_age": row.raw_age, "sex": row.sex,
                    "label": "normal" if row.healthy else None, "age_session_matched": matched,
                    "age_placeholder": False,
                    **{k: (bool(getattr(row, k)) if isinstance(getattr(row, k), (bool, np.bool_))
                           else getattr(row, k)) for k in keep},
                    "_file": path})
    return out


def crop_constant_ends(raw, picks):
    """Cut leading / trailing samples where none of the ``picks`` channels changes
    (constant padding up to whole data records), in place."""
    moving = np.flatnonzero(np.any(np.diff(raw.get_data(picks=picks), axis=1) != 0, axis=0))
    # moving[i] = j means sample j+1 differs from j: the signal runs from the first change
    # (sample 0 when nothing leads) to the last sample before the final change into the pad
    if moving.size:
        first = moving[0] + 1 if moving[0] > 0 else 0
        last = moving[-1] if moving[-1] < raw.n_times - 2 else raw.n_times - 1
        if first > 0 or last < raw.n_times - 1:
            raw.crop(first / raw.info["sfreq"], last / raw.info["sfreq"])


def open_tdbrain(path):
    """BDF (release 3.1) or BrainVision at 500 Hz, DC-coupled (low-pass
    100 Hz), 26 EEG channels (10-10) + EOG / ECG / EMG (+ the BDF ``Status``),
    referenced to linked mastoids, so A1/A2 (and T1/T2) are absent. The recording is DC-coupled: each channel's
    mean is removed before the pipeline's 0.1 Hz high-pass, which otherwise
    leaves a transient over the first seconds of a 2-min file. The original
    files are never written to (DUA)."""
    import mne
    if path.lower().endswith(".bdf"):
        raw = mne.io.read_raw_bdf(path, preload=True, verbose="error")
    else:
        raw = mne.io.read_raw_brainvision(path, preload=True, verbose="error")
    # Release 3.1 pads each file with a constant (digital zero) up to whole 1-s BDF
    # records, ~1 s; against a DC offset of mV that step would ring through the filters
    # into the last windows. Samples where no 10-20 channel changes are cut at both ends.
    present = channel_map(raw.ch_names, bids_canonical_channel)
    crop_constant_ends(raw, list(present.values()))
    raw._data -= raw._data.mean(axis=1, keepdims=True)
    return raw, present, None


def dortmund_events(events_tsv):
    """``(boundaries, n_triggers, usb_note)`` of a run's ``*_events.tsv``: the
    1-based sample latencies of the ``boundary`` events (the first one, at
    sample 1, is the start of the file; any further one is a discontinuity of
    the recording), the number of stimulus triggers (``S  n``: a trigger line
    that kept firing in some runs, the ``late_ses*`` count of participants.tsv)
    and whether the recorder noted a lost cap connection."""
    import pandas as pd
    if not os.path.exists(events_tsv):
        return [1], 0, False
    ev = pd.read_csv(events_tsv, sep="\t")
    kind = ev["type"].astype(str)
    bounds = sorted(int(v) for v in ev.loc[kind == "boundary", "onset"])
    return bounds or [1], int(kind.str.match(r"S\s*\d+").sum()), bool(kind.str.contains("USB").any())


def dortmund_recordings(raw_dir, meta_tsv):
    """One recording per run ``sub-<id>_ses-<n>_task-Eyes{Closed,Open}_acq-{pre,post}_eeg.edf``
    (stem ``sub-<id>_ses-<n>_<EC|EO>-<pre|post>_dortmund``), scored per run or pooled per session.
    ``age`` is the participants.tsv age for session 1 and that age plus the difference of the
    sessions' recording years (``sub-<id>_sessions.tsv``) for session 2, so a follow-up age is
    known to about +-1 year (``age_basis``). Every participant is healthy by the study's
    exclusion criteria: ``label`` is ``normal``."""
    import re
    import pandas as pd
    meta = pd.read_csv(meta_tsv, sep="\t").set_index("participant_id")
    pat = re.compile(r"^(sub-[^_]+)_ses-([^_]+)_task-Eyes(Closed|Open)_acq-(pre|post)_eeg\.edf$")
    out = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "sub-*", "ses-*", "eeg", "*_eeg.edf"))):
        m = pat.match(os.path.basename(path))
        if not m or m[1] not in meta.index or pd.isna(meta.loc[m[1], "age"]):
            continue
        sub, ses, task, acq = m[1], m[2], "EC" if m[3] == "Closed" else "EO", m[4]
        row = meta.loc[sub]
        years = {}
        ses_tsv = os.path.join(raw_dir, sub, f"{sub}_sessions.tsv")
        if os.path.exists(ses_tsv):
            t = pd.read_csv(ses_tsv, sep="\t")
            years = {str(a).replace("ses-", ""): float(b) for a, b in zip(t["session_id"], t["recording_year"])
                     if not pd.isna(b)}
        age, basis, matched = float(row["age"]), "participants.tsv (session 1)", True
        if ses != "1":
            if ses in years and "1" in years:
                age, basis = age + years[ses] - years["1"], "session 1 age + difference of recording years"
            else:                       # no recording year: the follow-up age is unknown
                basis, matched = "session 1 age (no recording year)", False
        bounds, n_trig, usb = dortmund_events(path[:-len("_eeg.edf")] + "_events.tsv")
        late = row.get(f"late_ses{ses}")
        out.append({"stem": f"{sub}_ses-{ses}_{task}-{acq}_dortmund", "path": os.path.relpath(path, raw_dir),
                    "corpus": "dortmund", "subject": sub, "session": ses, "task": task, "acq": acq,
                    "age": age, "raw_age": float(row["age"]), "age_basis": basis, "age_session_matched": matched,
                    "sex": row["sex"] if row["sex"] in ("M", "F") else None, "label": "normal", "healthy": True,
                    "handedness": None if pd.isna(row.get("handedness")) else str(row["handedness"]),
                    "recording_year": years.get(ses), "n_boundaries": len(bounds), "n_triggers": n_trig,
                    "usb_note": usb, "late_triggers": None if late is None or pd.isna(late) else int(late),
                    "_file": path})
    return out


def open_dortmund(path):
    """EDF at 1000 Hz, DC-coupled (hardware low-pass 250 Hz), 64 EEG channels
    (10-10 names) + ``Status``, FCz reference (kept: the models apply a common
    average reference over the 19 channels at their input). Three things are
    done before the shared pipeline:

    * a run with discontinuities (more than one ``boundary`` in its events.tsv)
      is cut to its longest continuous segment, so no filter runs across a jump;
    * constant padding at either end is cut;
    * each channel's mean is removed: the recording is DC-coupled with offsets
      of millivolts, which the pipeline's 0.1 Hz high-pass would turn into a
      transient over the first seconds of a 3-min run.

    The EDF physical range differs from file to file (the exporter scales every
    file to its own range); MNE applies it, and the audit of
    ``docs/dortmund_onboarding.md`` checks the result on every file."""
    import mne
    raw = mne.io.read_raw_edf(path, preload=True, verbose="error")
    raw.drop_channels([c for c in raw.ch_names if c.lower() == "status"])
    bounds, _, _ = dortmund_events(path[:-len("_eeg.edf")] + "_events.tsv")
    if len(bounds) > 1:
        edges = [b - 1 for b in bounds] + [raw.n_times]          # 0-based segment starts, then the end
        edges = sorted({min(max(e, 0), raw.n_times) for e in edges})
        k = int(np.argmax(np.diff(edges)))
        raw.crop(edges[k] / raw.info["sfreq"], (edges[k + 1] - 1) / raw.info["sfreq"])
    present = channel_map(raw.ch_names, bids_canonical_channel)
    crop_constant_ends(raw, list(present.values()))
    raw._data -= raw._data.mean(axis=1, keepdims=True)
    return raw, present, None


def keep_longest_segment(raw, breaks_sec):
    """Cut ``raw`` (preloaded) to its longest piece between the discontinuities at ``breaks_sec``."""
    sf = raw.info["sfreq"]
    edges = sorted({0, raw.n_times, *(int(round(b * sf)) for b in breaks_sec if 0 < b * sf < raw.n_times)})
    if len(edges) > 2:
        k = int(np.argmax(np.diff(edges)))
        raw.crop(edges[k] / sf, (edges[k + 1] - 1) / sf)


def _participants(meta_tsv):
    """BIDS participants.tsv (with or without a byte-order mark), indexed by participant_id."""
    import pandas as pd
    return pd.read_csv(meta_tsv, sep="\t", encoding="utf-8-sig").set_index("participant_id")


def _value(row, col):
    """A participants.tsv cell as a float, or None when it is missing."""
    import pandas as pd
    v = pd.to_numeric(row.get(col), errors="coerce")
    return None if pd.isna(v) else float(v)


def _sex(value):
    v = str(value).strip().upper()[:1]
    return v if v in ("M", "F") else None


def ahepa_recordings(raw_dir, meta_tsv):
    """One eyes-closed recording per participant; ``Group`` A / F / C = Alzheimer's disease /
    frontotemporal dementia / healthy control."""
    meta = _participants(meta_tsv)
    out = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "sub-*", "eeg", "*_task-eyesclosed_eeg.set"))):
        sub = os.path.basename(path).split("_")[0]
        if sub not in meta.index or _value(meta.loc[sub], "Age") is None:
            continue
        m = meta.loc[sub]
        group = str(m["Group"]).strip().upper()
        out.append({"stem": f"{sub}_ahepa", "path": os.path.relpath(path, raw_dir), "corpus": "ahepa",
                    "subject": sub, "session": "1", "age": _value(m, "Age"), "raw_age": _value(m, "Age"),
                    "sex": _sex(m["Gender"]), "label": "normal" if group == "C" else None,
                    "healthy": group == "C", "dx_ad": group == "A", "dx_ftd": group == "F",
                    "clinical_group": {"A": "alzheimer", "F": "ftd", "C": "control"}.get(group, "unknown"),
                    "mmse": _value(m, "MMSE"), "_file": path})
    return out


def open_ahepa(path):
    """EEGLAB at 500 Hz, the 19 10-20 channels as exported by the clinical system (time constant
    0.3 s, 70 Hz high cut); kept in the exported reference."""
    import mne
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose="error")
    raw._data -= raw._data.mean(axis=1, keepdims=True)
    return raw, channel_map(raw.ch_names, bids_canonical_channel), None


def iowapd_recordings(raw_dir, meta_tsv):
    meta = _participants(meta_tsv)
    out = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "sub-*", "eeg", "*_task-Rest_eeg.set"))):
        sub = os.path.basename(path).split("_")[0]
        if sub not in meta.index or _value(meta.loc[sub], "AGE") is None:
            continue
        m = meta.loc[sub]
        control = str(m["GROUP"]).strip().lower() == "control"
        out.append({"stem": f"{sub}_iowapd", "path": os.path.relpath(path, raw_dir), "corpus": "iowapd",
                    "subject": sub, "session": "1", "age": _value(m, "AGE"), "raw_age": _value(m, "AGE"),
                    "sex": _sex(m["GENDER"]), "label": "normal" if control else None, "healthy": control,
                    "dx_pd": not control, "clinical_group": "control" if control else "parkinson",
                    "moca": _value(m, "MOCA"), "updrs": _value(m, "UPDRS"), "_file": path})
    return out


def open_iowapd(path):
    """EEGLAB at 500 Hz, 63-66 channels referenced to Pz. Pz itself is not in the file: it is added
    as the (zero) reference channel, and ``reference="average"`` then re-references the 19 picked
    channels to their mean, which gives Pz its signal back."""
    import mne
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose="error")
    if "Pz" not in raw.ch_names:
        raw = mne.add_reference_channels(raw, ["Pz"])
    keep_longest_segment(raw, [a["onset"] for a in raw.annotations if a["description"] == "boundary"])
    raw._data -= raw._data.mean(axis=1, keepdims=True)
    return raw, channel_map(raw.ch_names, bids_canonical_channel), "average"


def telavivpd_recordings(raw_dir, meta_tsv):
    meta = _participants(meta_tsv)
    out = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "sub-*", "eeg", "*_task-rest_eeg.set"))):
        sub = os.path.basename(path).split("_")[0]
        if sub not in meta.index or _value(meta.loc[sub], "age") is None:
            continue
        m = meta.loc[sub]
        control = str(m["group"]).strip().upper() == "HC"
        out.append({"stem": f"{sub}_rest_telavivpd", "path": os.path.relpath(path, raw_dir), "corpus": "telavivpd",
                    "subject": sub, "session": "1", "age": _value(m, "age"), "raw_age": _value(m, "age"),
                    "sex": _sex(m["sex"]), "label": "normal" if control else None, "healthy": control,
                    "dx_pd": not control, "clinical_group": "control" if control else "parkinson",
                    "moca": _value(m, "moca"), "updrs": _value(m, "updrs_total"), "ledd": _value(m, "ledd"),
                    "disease_duration": _value(m, "disease_duration"), "_file": path})
    return out


def open_telavivpd(path):
    """EEGLAB at 250 Hz, 60 EEG channels with 10-10 labels + 4 EOG + ``VREF``. ``VREF`` is the
    vertex reference, i.e. Cz, and all zero: it is renamed and ``reference="average"`` gives it its
    signal back. The recorder marks breaks in continuity (``break cnt``, ``boundary``): the longest
    continuous piece is kept. The DC offset (tens of millivolts) is removed."""
    import mne
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose="error")
    if "VREF" in raw.ch_names and "Cz" not in raw.ch_names:
        raw.rename_channels({"VREF": "Cz"})
    keep_longest_segment(raw, [a["onset"] for a in raw.annotations if a["description"] in ("break cnt", "boundary")])
    raw._data -= raw._data.mean(axis=1, keepdims=True)
    return raw, channel_map(raw.ch_names, bids_canonical_channel), "average"


def srm_recordings(raw_dir, meta_tsv):
    """One recording per session (``t1`` for all 111 participants, ``t2`` for 42). The listed age is
    that of ``t1``; a ``t2`` age adds the time between the sessions' ``acq_time`` (``*_scans.tsv``)."""
    import pandas as pd
    meta = _participants(meta_tsv)
    out = []
    when = {}
    for scans in glob.glob(os.path.join(raw_dir, "sub-*", "ses-*", "*_scans.tsv")):
        t = pd.read_csv(scans, sep="\t")
        parts = os.path.normpath(scans).split(os.sep)
        when[(parts[-3], parts[-2])] = pd.to_datetime(t["acq_time"].iloc[0], errors="coerce")
    for path in sorted(glob.glob(os.path.join(raw_dir, "sub-*", "ses-*", "eeg", "*_task-resteyesc_eeg.edf"))):
        sub, ses = os.path.basename(path).split("_")[:2]
        if sub not in meta.index or _value(meta.loc[sub], "age") is None:
            continue
        m = meta.loc[sub]
        age, matched = _value(m, "age"), True
        if ses != "ses-t1":
            t1, t = when.get((sub, "ses-t1")), when.get((sub, ses))
            if t1 is None or t is None or pd.isna(t1) or pd.isna(t):
                matched = False
            else:
                age += (t - t1).days / 365.25
        out.append({"stem": f"{sub}_{ses}_srm", "path": os.path.relpath(path, raw_dir), "corpus": "srm",
                    "subject": sub, "session": ses.replace("ses-", ""), "age": age, "raw_age": _value(m, "age"),
                    "age_session_matched": matched, "sex": _sex(m["sex"]), "label": "normal", "healthy": True,
                    "_file": path})
    return out


def _edf_dimensions(path):
    """The physical-dimension strings of an EDF file's signals."""
    with open(path, "rb") as fh:
        ns = int(fh.read(256)[252:256])
        fh.seek(256 + ns * (16 + 80))
        return [fh.read(8).decode("latin1").strip() for _ in range(ns)]


def open_srm(path):
    """EDF at 1024 Hz, 64 channels, already re-referenced to the average of the 64 (kept). The
    headers have an empty physical dimension, which MNE reads as volts; the values are microvolts
    (the alpha peak is then a few µV²/Hz), so they are scaled. The recording is DC-coupled: each
    channel's mean is removed."""
    import mne
    raw = mne.io.read_raw_edf(path, preload=True, verbose="error")
    if not any(_edf_dimensions(path)):
        raw._data *= 1e-6
    crop_constant_ends(raw, list(channel_map(raw.ch_names, bids_canonical_channel).values()))
    raw._data -= raw._data.mean(axis=1, keepdims=True)
    return raw, channel_map(raw.ch_names, bids_canonical_channel), None


NMT_SCALE = 100.0        # released values x this = microvolts (assumption, see the module docstring)


def nmt_recordings(raw_dir, meta_csv):
    """One recording per patient (``Labels.csv``: recordname, label, age, gender, loc), found
    under ``<label>/<loc>/<recordname>``. ``label`` is the neurologists' normal / abnormal verdict."""
    import pandas as pd
    meta = pd.read_csv(meta_csv)
    out = []
    for r in meta.itertuples():
        path = os.path.join(raw_dir, str(r.label), str(r.loc), str(r.recordname))
        age = pd.to_numeric(r.age, errors="coerce")
        if not os.path.exists(path) or pd.isna(age):
            continue
        rec = os.path.splitext(str(r.recordname))[0]
        out.append({"stem": f"nmt_{rec}", "path": os.path.relpath(path, raw_dir), "corpus": "nmt",
                    "subject": f"nmt_{rec}", "session": "1", "age": float(age), "raw_age": float(age),
                    "sex": _sex(r.gender), "label": str(r.label) if str(r.label) in ("normal", "abnormal") else None,
                    "source_set": str(r.loc), "amplitude_scale": NMT_SCALE, "_file": path})
    return out


def open_nmt(path):
    """EDF at 200 Hz, 19 channels + A1 / A2, average-referenced as released; values scaled by
    ``NMT_SCALE`` to microvolts."""
    import mne
    raw = mne.io.read_raw_edf(path, preload=True, verbose="error")
    raw._data *= NMT_SCALE
    raw._data -= raw._data.mean(axis=1, keepdims=True)
    return raw, channel_map(raw.ch_names, bids_canonical_channel), None


DATASETS = {"lemon": (lemon_recordings, open_lemon), "mipdb": (mipdb_recordings, open_mipdb),
            "chbp": (chbp_recordings, open_chbp), "tdbrain": (tdbrain_recordings, open_tdbrain),
            "dortmund": (dortmund_recordings, open_dortmund), "ahepa": (ahepa_recordings, open_ahepa),
            "iowapd": (iowapd_recordings, open_iowapd), "telavivpd": (telavivpd_recordings, open_telavivpd),
            "srm": (srm_recordings, open_srm), "nmt": (nmt_recordings, open_nmt)}


# ------------------------------------------------------------------ convert
def convert_one(rec, opener, out_dir):
    raw, present, reference = opener(rec["_file"])
    # recorder markers (segment boundaries, events): a mid-file boundary means a discontinuity
    annotations = {}
    for desc in raw.annotations.description:
        annotations[str(desc)] = annotations.get(str(desc), 0) + 1
    sfreq = float(raw.info["sfreq"])
    duration = raw.n_times / sfreq
    if duration < MIN_DURATION_SEC:
        return {"stem": rec["stem"], "error": f"too short ({duration:.0f} s)"}
    if len(present) < 19:
        return {"stem": rec["stem"], "error": f"only {len(present)} of the 19 10-20 channels found"}
    x, mask = preprocess_raw(raw, present, NOTCH, LINE, reference=reference)
    x = np.clip(x, -FLOAT16_MAX, FLOAT16_MAX).astype(np.float16)
    data = npy_bytes(np.ascontiguousarray(x))
    with open(os.path.join(out_dir, "recordings", rec["stem"] + ".npy"), "wb") as fh:
        fh.write(data)
    return {"stem": rec["stem"], "n_windows": x.shape[0] // WINDOW_SAMPLES, "n_samples": x.shape[0],
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "channel_mask": mask,
            "dtype": "float16", "notch": "50,60", "line": "interp:50,60",
            "reference": reference or "as recorded", "sfreq_native": sfreq, "duration_sec": duration,
            "annotations": annotations}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--meta", default=None,
                    help="participants / demographics table (optional for tdbrain: placeholder ages)")
    ap.add_argument("--relabel", action="store_true",
                    help="rewrite manifest / age_metadata / age_split of an existing --out from --meta "
                         "without converting the signals again")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.meta is None and args.dataset != "tdbrain":
        ap.error(f"--meta is required for {args.dataset}")
    lister, opener = DATASETS[args.dataset]
    recs = lister(args.raw, args.meta)[:args.limit or None]
    os.makedirs(os.path.join(args.out, "recordings"), exist_ok=True)
    print(f"{args.dataset}: {len(recs)} recordings", flush=True)
    results, errors = {}, []
    if args.relabel:
        # keep each converted file's signal fields; only the metadata is rebuilt
        old = json.load(open(os.path.join(args.out, "manifest.json")))
        fields = ("stem", "n_windows", "n_samples", "sha256", "bytes", "channel_mask", "dtype", "notch",
                  "line", "reference", "sfreq_native", "duration_sec", "annotations")
        results = {r["stem"]: {k: r.get(k) for k in fields} for r in old["recordings"]}
        errors = old["cohort"]["errors"]
        missing = [r["stem"] for r in recs if r["stem"] not in results]
        print(f"relabel: {len(recs) - len(missing)} converted recordings get metadata, "
              f"{len(missing)} listed but never converted", flush=True)
        recs = [r for r in recs if r["stem"] in results]
    with cf.ProcessPoolExecutor(args.jobs) as pool:
        futs = {} if args.relabel else {pool.submit(convert_one, r, opener, args.out): r for r in recs}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            r = futs[fut]
            try:
                res = fut.result()
            except Exception as exc:          # record, never abort the run
                res = {"stem": r["stem"], "error": f"{type(exc).__name__}: {exc}"[:300]}
            if "error" in res:
                errors.append(res)
            else:
                results[r["stem"]] = res
            if i % 100 == 0 or i == len(recs):
                print(f"  {i}/{len(recs)} done, {len(errors)} errors", flush=True)
    rows = []
    for r in recs:
        if r["stem"] not in results:
            continue
        row = {**results[r["stem"]], **{k: v for k, v in r.items() if not k.startswith("_")},
               "token": None, "year": None, "montage": None, "session_date": None,
               "abnormal": None, "epilepsy": None, "source_split": "test", "site": args.dataset}
        rows.append(row)
    manifest = {
        "format_version": 2, "layout": "recordings/<stem>.npy, [n_samples, n_channels], time-major",
        "dtype": "float16", "sample_rate": SAMPLE_RATE, "window_samples": WINDOW_SAMPLES,
        "channels": list(TUH_CHANNELS), "units": "uV", "notch_hz": "50,60", "line_removal": "interp:50,60",
        "provenance": (f"dataset_maker/make_external_npy.py --dataset {args.dataset}: "
                       "make_TUH_npy.preprocess_raw (0.1-75.0 Hz band-pass, line removal interp:50,60 Hz, "
                       "resampled to 200 Hz); missing channels zero-filled, see channel_mask"),
        "n_recordings": len(rows), "n_windows": int(sum(r["n_windows"] for r in rows)),
        "cohort": {"experiment": "external_eval", "corpus": args.dataset, "split": "all test",
                   "errors": errors, "age_placeholder": any(r.get("age_placeholder") for r in rows)},
        "recordings": rows,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh)
    if args.dataset == "tdbrain" and args.meta:
        # every participants.tsv column (questionnaires, demographics) + the harmonised clinical flags
        tdbrain_clinical_table(args.meta).to_parquet(os.path.join(args.out, "clinical.parquet"), index=False)
    fields = ("stem", "subject", "session", "token", "age", "sex", "year", "raw_age", "label")
    with open(os.path.join(args.out, "age_metadata.json"), "w") as fh:
        json.dump({"version": 1, "recordings": {r["stem"]: {k: r.get(k) for k in fields} for r in rows}}, fh)
    # a TDBRAIN session without its own participants row carries the subject's first-session
    # age: kept in the manifest, left out of the scored split
    scored = [r for r in rows if r.get("age_session_matched", True) is not False]
    files = [f"test/{r['stem']}_{k}.pkl" for r in scored for k in range(r["n_windows"])]
    with open(os.path.join(args.out, "age_split.json"), "w") as fh:
        json.dump({"version": 1, "seed": None, "counts": {"train": 0, "val": 0, "test": len(scored)},
                   "files": {"train": [], "val": [], "test": files}}, fh)
    print(f"{len(rows)} recordings, {manifest['n_windows']} windows -> {args.out}; {len(errors)} errors")
    for e in errors[:10]:
        print("  ", e)


if __name__ == "__main__":
    main()
