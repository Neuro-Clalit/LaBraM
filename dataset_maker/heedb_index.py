#!/usr/bin/env python
"""Harvard EEG Database (HEEDB v4.1, BDSP) -> a recording index aligned with TUH.

HEEDB is read through the BDSP credentialed S3 access point (DUA: never
redistribute; our copy lives in the private ``s3://eeg-data-public/HEEDB/v4.1/``,
see ``download_heedb.py``). It ships as BIDS
(``EEG/bids/<site>/sub-<site><pid>/ses-<N>/eeg/*_eeg.edf``) plus per-site CSVs:

* ``eeg-metadata/<site>_eeg_metadata_*.csv`` -- one row per recording session:
  start time, duration, service (Routine / LTM / EMU / OR), age at visit, sex;
* ``HEEDB_Metadata/<site>_EEG_reports_findings.csv`` -- per session, the report's
  normal / abnormal call and ~45 findings, each tagged with where it was found
  (``report``, ``annotation`` = Persyst/XLTEK machine annotation, ``verified``);
* ``HEEDB_ICD10_for_Neurology.csv`` / ``HEEDB_Medication_ATC.csv`` -- per
  patient, the ICD-10 codes and ATC level-4 drug groups of the health record,
  grouped into categories. **Per patient, not dated.**

This module turns them into one row per EDF recording with the **same column
names as the TUH index** (``dataset_maker.tuh_index``) and the **same clinical
flags as the NEDC TUEG sheet** (``dataset_maker.tuh_clinical``: ``dx_*``,
``f_*``, ``med_*``, ``abnormal_report``, ``any_neuro_dx``, ``sedated``), so
``tuh_clinical.normative_mask`` / ``report`` run unchanged on HEEDB. The EDF
headers are read with S3 range requests (no recording is downloaded).

    python -m dataset_maker.heedb_index metadata            # CSVs -> <root>/metadata
    python -m dataset_maker.heedb_index list                # BIDS listing -> scan/listing.parquet
    python -m dataset_maker.heedb_index headers --select adult_routine
    python -m dataset_maker.heedb_index build               # -> index/heedb_index.parquet
"""
import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from labram.data.tuh_metadata import (  # noqa: E402
    EDF_HEADER_BYTES, TUH_CHANNELS, bids_canonical_channel, channel_map, edf_header_size,
    parse_edf_signal_header)

ACCESS_POINT = "arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-credentialed-access-point"
EEG_PREFIX = "EEG"
DEFAULT_ROOT = "/data/datasets/HEEDB/v4.1"
DEFAULT_PROFILE = os.environ.get("HEEDB_AWS_PROFILE", "mfa")
# Sites with reports + health records; the others (I0004, I0008, I0009, ELUCID,
# Neurotech, norse, teleEEG) have recordings but no report/ICD tables.
MAIN_SITES = ("S0001", "S0002", "I0002", "I0003")
ADULT_AGE = (18.0, 89.0)            # same adult range as tuh_clinical.normative_mask
ROUTINE_MINUTES = (10.0, 240.0)     # a routine EEG's length, minutes
# Sites without a service label: a routine study is taken to be 15 min - 2 h.
ROUTINE_PROXY_MINUTES = (15.0, 120.0)

# --------------------------------------------------------------- TUH alignment
# ICD-10 prefixes -> the NEDC sheet's diagnosis flags (tuh_clinical.DIAGNOSES).
# dx_electrographic_seizure has no ICD code: it comes from the EEG report.
ICD_TO_TUH = {
    "dx_clinical_seizure": ("R56", "G40", "G41"),
    "dx_epilepsy": ("G40", "G41"),
    "dx_stroke": ("I60", "I61", "I62", "I63", "I64", "I69"),
    "dx_parkinsons": ("G20", "G21"),
    "dx_alzheimers": ("G30",),
    "dx_dementia": ("F01", "F02", "F03", "G30", "G31.0", "G31.1", "G31.83"),
    "dx_tbi": ("S06",),
    "dx_schizophrenia": ("F20", "F25"),
    "dx_substance": tuple(f"F1{i}" for i in range(10)),
    "dx_anxiety": ("F41",),
    "dx_depression": ("F32", "F33"),
    "dx_bipolar": ("F31",),
    "dx_adhd": ("F90",),
    "dx_ms": ("G35",),
}
# ATC level-4 prefixes -> the sheet's drug classes (tuh_clinical.DRUG_CLASSES).
ATC_TO_TUH = {
    "med_anesthetic": ("N01A", "N05CM"),
    "med_benzodiazepine": ("N05BA", "N05CD", "N03AE"),
    "med_barbiturate": ("N03AA", "N05CA", "N01AF"),
    "med_opioid": ("N02A", "N01AH", "N07BC"),
    "med_antiseizure": ("N03A",),
    "med_antipsychotic": ("N05A",),
}
# Report findings -> the sheet's f_* flags. FIRDA/TIRDA are frontal/temporal
# rhythmic delta; HEEDB only says generalized (GRDA) / lateralized (LRDA).
FINDING_TO_TUH = {"f_gpd": ("gpd",), "f_lpd": ("lpd",), "f_firda": ("grda",), "f_tirda": ("lrda",)}
FINDING_TO_TUH_EXACT = {"f_gpd": True, "f_lpd": True, "f_firda": False, "f_tirda": False}
NEURO = ("dx_electrographic_seizure", "dx_clinical_seizure", "dx_stroke", "dx_epilepsy",
         "dx_parkinsons", "dx_alzheimers", "dx_tbi", "dx_dementia", "dx_ms")
SEDATING = ("med_anesthetic", "med_benzodiazepine", "med_barbiturate", "med_opioid")

# Report findings grouped into candidate targets. Normal variants and sleep
# graphoelements are kept apart: they say the patient slept, not that the EEG
# is abnormal.
FINDING_GROUPS = {
    "epileptiform": ("spikes", "lpd", "gpd", "bipd", "bects", "eses", "ppr"),
    "seizure": ("seizure", "status"),
    "generalized_slowing": ("gen_slowing", "grda"),
    "focal_slowing": ("foc_slowing", "lrda"),
    "periodic_discharges": ("lpd", "gpd", "bipd"),
    "burst_suppression_low_voltage": ("bs", "low_voltage"),
    "normal_variants": ("bets", "wicket", "posts", "breach"),
    "sleep": ("n1", "n2", "spindles", "vertex_wave", "k_complexes"),
}
REPORT_SOURCES = ("report", "verified")      # what a report-based label means

# Health-record (ICD category) groups for candidate weak-label heads.
ICD_GROUPS = ("seizure_disorders", "cerebral_degeneration", "cerebrovascular_diseases",
              "cerebral_lobe_dysfunction", "behavioral_cognitive_syndromes",
              "developmental_delay", "intracranial_and_spinal_tumors", "headache_disorders",
              "movement_and_cerebellar_disorders", "sleep_disorders")

AUX_PATTERNS = {
    "has_ecg": r"ECG|EKG", "has_eog": r"EOG|LOC|ROC", "has_emg": r"EMG|CHIN",
    "has_resp": r"CHEST|ABD|FLOW|SNORE|SPO2|SAO2|RESP|THOR", "has_photic": r"PHOTIC|^DC\d",
}


def slug(name: str) -> str:
    """'Seizure Disorders' -> 'seizure_disorders'; 'Behavioral/Cognitive ...' too."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def decade(age):
    """Age decade 0, 10, ..., 90 (90 = 90+)."""
    return (np.floor(np.asarray(age, dtype=float) / 10) * 10).clip(max=90)


# ------------------------------------------------------------------ S3 access
def s3_client(profile: Optional[str] = DEFAULT_PROFILE, pool: int = 64):
    import boto3
    from botocore.config import Config
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    return session.client("s3", config=Config(max_pool_connections=pool,
                                              retries={"max_attempts": 8, "mode": "adaptive"}))


def fetch_metadata(client, root: str) -> List[str]:
    """Copy ``HEEDB_Metadata/`` and ``eeg-metadata/`` (~160 MB) to ``<root>/metadata``."""
    out = []
    for sub in ("HEEDB_Metadata", "eeg-metadata"):
        pages = client.get_paginator("list_objects_v2").paginate(
            Bucket=ACCESS_POINT, Prefix=f"{EEG_PREFIX}/{sub}/")
        for page in pages:
            for obj in page.get("Contents", []):
                if obj["Key"].endswith("/"):
                    continue
                dst = os.path.join(root, "metadata", sub, os.path.basename(obj["Key"]))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not (os.path.exists(dst) and os.path.getsize(dst) == obj["Size"]):
                    client.download_file(ACCESS_POINT, obj["Key"], dst)
                out.append(dst)
    return out


# ------------------------------------------------------------------- metadata
def _col(df: pd.DataFrame, *names):
    for name in names:
        if name in df:
            return df[name]
    return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")


def _session_str(values: pd.Series) -> pd.Series:
    """Session ids as strings ('7', not '7.0'); non-numeric ids kept as they are."""
    num = pd.to_numeric(values, errors="coerce")
    return num.map(lambda v: str(int(v)) if pd.notna(v) else None).where(
        num.notna(), values.astype("string"))


def _sex(values: pd.Series) -> pd.Series:
    first = values.astype("string").str.strip().str.upper().str[0]
    return first.where(first.isin(["M", "F"]))


def load_recordings(meta_dir: str) -> pd.DataFrame:
    """``eeg-metadata/*.csv`` of every site -> one row per recording session.

    Columns: ``site, subject, patient_id, session, start_time, year, duration_sec,
    service, folder, age, sex, deceased, has_xltek, has_persyst``. ``subject`` is
    the BIDS label (``<site><patient id>``); the age of I0008/I0009 is computed from
    the date of birth. Duplicate (subject, session) rows keep the first."""
    frames = []
    for path in sorted(glob.glob(os.path.join(meta_dir, "eeg-metadata", "*.csv"))):
        d = pd.read_csv(path, low_memory=False)
        site = _col(d, "SiteID", "InstituteID").astype("string")
        subject = _col(d, "BidsFolder").astype("string").str.replace("sub-", "", regex=False)
        pid = pd.to_numeric(_col(d, "BDSPPatientID"), errors="coerce")
        from_folder = pd.to_numeric(subject.str[len("S0001"):], errors="coerce")
        start = pd.to_datetime(_col(d, "StartTime", "StartDateTime"), errors="coerce")
        age = pd.to_numeric(_col(d, "AgeAtVisit"), errors="coerce")
        dob = pd.to_datetime(_col(d, "DateOfBirth"), errors="coerce")
        age = age.fillna((start - dob).dt.days / 365.25)
        frames.append(pd.DataFrame({
            "site": site, "subject": subject,
            "patient_id": pid.fillna(from_folder).astype("Int64"),
            "session": _session_str(_col(d, "SessionID")),
            "start_time": start, "year": start.dt.year.astype("Int64"),
            "duration_sec": pd.to_numeric(_col(d, "DurationInSeconds", "RecordingDuration"),
                                          errors="coerce"),
            "service": _col(d, "ServiceName").astype("string"),
            "folder": _col(d, "EEGFolder").astype("string"),
            "age": age, "sex": _sex(_col(d, "SexDSC")),
            "deceased": _col(d, "DateOfDeath").notna() if "DateOfDeath" in d else pd.NA,
            "has_xltek": _col(d, "HasXLTEKAnnotations").astype("string").str.upper().str[0].eq("Y"),
            "has_persyst": _col(d, "HasPersystAnnotations").astype("string").str.upper().str[0].eq("Y"),
        }))
    rec = pd.concat(frames, ignore_index=True)
    rec = rec[rec["subject"].notna() & rec["session"].notna()]
    rec = rec.drop_duplicates(["subject", "session"], keep="first").reset_index(drop=True)
    rec["deceased"] = rec["deceased"].astype("boolean")
    return add_routine_flag(rec)


def add_routine_flag(rec: pd.DataFrame) -> pd.DataFrame:
    """``routine`` (bool) and ``routine_source``: the service label where the
    site has one (Routine / rEEG folder), else the 15 min - 2 h duration proxy;
    every routine study must also be 10 min - 4 h long."""
    minutes = rec["duration_sec"] / 60
    labelled = rec.groupby("site")["service"].transform(lambda s: s.notna().any())
    by_label = rec["service"].eq("Routine").fillna(False) | rec["folder"].eq("rEEG").fillna(False)
    by_proxy = minutes.between(*ROUTINE_PROXY_MINUTES)
    rec = rec.copy()
    rec["routine_source"] = np.where(labelled, "service", "duration")
    rec["routine"] = np.where(labelled, by_label, by_proxy) & minutes.between(*ROUTINE_MINUTES)
    rec["adult"] = rec["age"].between(*ADULT_AGE)
    return rec


def _finding_name(col: str) -> str:
    return slug(col)


def load_reports(meta_dir: str) -> pd.DataFrame:
    """``*_EEG_*reports_findings.csv`` -> one row per (site, patient, session).

    ``report_normal`` / ``abnormal_report`` come from the report's normal /
    abnormal call (nullable bool, NA where the report says neither).
    ``rf_<finding>`` holds the provenance string of each finding (``'report
    annotation'``, NA = not mentioned); ``N1`` and ``n1`` are merged."""
    frames = []
    for path in sorted(glob.glob(os.path.join(meta_dir, "HEEDB_Metadata", "*reports_findings*.csv"))):
        site = os.path.basename(path)[:5]
        d = pd.read_csv(path, low_memory=False)
        first = list(d.columns).index("normal")
        out = pd.DataFrame({
            "site": site,
            "patient_id": pd.to_numeric(d["BDSPPatientID"], errors="coerce").astype("Int64"),
            "session": _session_str(d["SessionID"]),
            "report_age": pd.to_numeric(d["AgeAtVisit"], errors="coerce"),
            "report_service": _col(d, "ServiceName(EEG)").astype("string"),
        })
        for col in d.columns[first:]:
            name = "rf_" + _finding_name(col)
            vals = d[col].astype("string").str.strip().str.lower()
            vals = vals.where(vals.notna() & vals.ne(""))
            if name in out:                     # 'N1' and 'n1' -> one column
                both = out[name].fillna("") + " " + vals.fillna("")
                merged = both.str.split().map(lambda t: " ".join(sorted(set(t))) or None)
                out[name] = merged.astype("string")
            else:
                out[name] = vals
        frames.append(out)
    rep = pd.concat(frames, ignore_index=True)
    rep = rep.drop_duplicates(["site", "patient_id", "session"], keep="first")
    normal, abnormal = rep["rf_normal"].notna(), rep["rf_abnormal"].notna()
    rep["report_normal"] = normal.astype("boolean")
    rep["abnormal_report"] = pd.array(np.where(abnormal, True, np.where(normal, False, None)),
                                      dtype="boolean")
    return rep.reset_index(drop=True)


def finding_flags(reports: pd.DataFrame, sources: Sequence[str] = REPORT_SOURCES) -> pd.DataFrame:
    """``find_<x>`` booleans: the finding is present from one of ``sources``.
    Sessions without a report get NA."""
    pattern = "|".join(rf"\b{re.escape(s)}\b" for s in sources)
    has_report = reports["abnormal_report"].notna()
    out = {}
    for col in [c for c in reports.columns if c.startswith("rf_")]:
        hit = reports[col].fillna("").str.contains(pattern, regex=True)
        out["find_" + col[3:]] = pd.array(np.where(has_report, hit, None), dtype="boolean")
    flags = pd.DataFrame(out, index=reports.index)
    for group, members in FINDING_GROUPS.items():
        cols = [f"find_{m}" for m in members if f"find_{m}" in flags]
        flags[f"fg_{group}"] = flags[cols].astype("boolean").any(axis=1).astype("boolean").where(
            has_report, pd.NA) if cols else pd.NA
    return flags


def _per_patient_codes(path: str, prefix: str) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    cats = [c for c in d.columns if c not in
            ("BDSPPatientID", "SiteID", "SexDSC", "VisitCount", "AgeAtVisitAvg")]
    out = pd.DataFrame({
        "site": d["SiteID"].astype("string"),
        "patient_id": pd.to_numeric(d["BDSPPatientID"], errors="coerce").astype("Int64"),
    })
    for cat in cats:
        out[f"{prefix}_{slug(cat)}"] = d[cat].notna() & d[cat].astype("string").str.strip().ne("")
    codes = d[cats].astype("string").fillna("").agg(" ".join, axis=1)
    out[f"{prefix}_codes"] = codes.str.split().map(lambda t: " ".join(sorted(set(t))))
    return out.drop_duplicates(["site", "patient_id"]).reset_index(drop=True)


def load_icd(meta_dir: str) -> pd.DataFrame:
    """Per patient: ``icd_<category>`` flags + ``icd_codes`` (space-joined)."""
    return _per_patient_codes(os.path.join(meta_dir, "HEEDB_Metadata",
                                           "HEEDB_ICD10_for_Neurology.csv"), "icd")


def load_atc(meta_dir: str) -> pd.DataFrame:
    """Per patient: ``atc_<anatomical group>`` flags + ``atc_codes``."""
    return _per_patient_codes(os.path.join(meta_dir, "HEEDB_Metadata",
                                           "HEEDB_Medication_ATC.csv"), "atc")


def load_patients(meta_dir: str) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(meta_dir, "HEEDB_Metadata", "HEEDB_patients.csv"), low_memory=False)
    return pd.DataFrame({
        "site": d["SiteID"].astype("string"),
        "patient_id": pd.to_numeric(d["BDSPPatientID"], errors="coerce").astype("Int64"),
        "patient_age_avg": pd.to_numeric(d["AgeAtVisitAvg"], errors="coerce"),
        "race": d["RaceAndEthnicity"].astype("string"),
        "visit_count": pd.to_numeric(d["VisitCount"], errors="coerce"),
        "icd_count": pd.to_numeric(d["ICD10Count"], errors="coerce"),
        "medication_count": pd.to_numeric(d["MedicationCount"], errors="coerce"),
    }).drop_duplicates(["site", "patient_id"])


def code_flags(codes: pd.Series, mapping: Dict[str, Sequence[str]]) -> pd.DataFrame:
    """Boolean column per mapping key: any code starts with one of its prefixes
    (codes are space-joined strings; NA codes give NA flags)."""
    tokens = codes.fillna("").str.split()
    out = {}
    for name, prefixes in mapping.items():
        pre = tuple(prefixes)
        hit = tokens.map(lambda ts, p=pre: any(t.startswith(p) for t in ts))
        out[name] = pd.array(np.where(codes.notna(), hit, None), dtype="boolean")
    return pd.DataFrame(out, index=codes.index)


def tuh_aligned_flags(index: pd.DataFrame) -> pd.DataFrame:
    """The NEDC TUEG sheet's columns (``tuh_clinical.FLAGS``) built from HEEDB.

    * ``dx_*`` from ICD-10 (``ICD_TO_TUH``), NA for patients without any ICD
      record -- as in ``tuh_clinical.join_index``, no record is not evidence of
      health; ``dx_electrographic_seizure`` from the report's seizure finding;
    * ``med_*`` from ATC groups (``ATC_TO_TUH``), NA without medication record;
    * ``f_*`` from report findings (``FINDING_TO_TUH``);
    * ``abnormal_report``, ``any_neuro_dx``, ``sedated``, ``in_sheet``,
      ``label`` exactly as the TUH pipeline defines them."""
    has_icd = index["icd_count"].fillna(0).gt(0) & index["icd_codes"].notna()
    has_atc = index["medication_count"].fillna(0).gt(0) & index["atc_codes"].notna()
    dx = code_flags(index["icd_codes"].where(has_icd), ICD_TO_TUH)
    med = code_flags(index["atc_codes"].where(has_atc), ATC_TO_TUH)
    out = pd.concat([dx, med], axis=1)
    out["dx_electrographic_seizure"] = index.get("find_seizure", pd.Series(pd.NA, index=index.index))
    for flag, members in FINDING_TO_TUH.items():
        cols = [f"find_{m}" for m in members if f"find_{m}" in index]
        out[flag] = index[cols].astype("boolean").any(axis=1).where(
            index["abnormal_report"].notna(), pd.NA).astype("boolean")
    out["f_oirda"] = pd.array([pd.NA] * len(index), dtype="boolean")
    neuro = out[list(NEURO)].astype("boolean")
    # any_neuro_dx: True if any flag is True; False only with an ICD record.
    out["any_neuro_dx"] = pd.array(np.where(neuro.fillna(False).any(axis=1), True,
                                            np.where(has_icd, False, None)), dtype="boolean")
    sed = out[list(SEDATING)].astype("boolean")
    out["sedated"] = pd.array(np.where(sed.fillna(False).any(axis=1), True,
                                       np.where(has_atc, False, None)), dtype="boolean")
    out["in_sheet"] = index["abnormal_report"].notna()
    out["label"] = index["abnormal_report"].map({True: "abnormal", False: "normal"}).astype("string")
    return out


# -------------------------------------------------------------------- listing
_KEY_RE = re.compile(r"^EEG/bids/(?P<site>[^/]+)/sub-(?P<subject>[^/]+)/ses-(?P<session>[^/]+)/"
                     r"(?:eeg/)?(?P<file>[^/]+)$")
_EDF_RE = re.compile(r"^sub-[^_]+_ses-[^_]+_task-(?P<task>[^_]*)(?:_run-(?P<run>[^_]+))?_eeg\.edf$",
                     re.IGNORECASE)


def parse_key(key: str) -> Optional[dict]:
    m = _KEY_RE.match(key)
    return m.groupdict() if m else None


def _list_prefix(client, prefix: str, delimiter: Optional[str] = None):
    kw = {"Bucket": ACCESS_POINT, "Prefix": prefix}
    if delimiter:
        kw["Delimiter"] = delimiter
    objs, prefixes = [], []
    for page in client.get_paginator("list_objects_v2").paginate(**kw):
        objs += [(o["Key"], o["Size"]) for o in page.get("Contents", [])]
        prefixes += [p["Prefix"] for p in page.get("CommonPrefixes", [])]
    return objs, prefixes


def list_site(client, site: str, workers: int = 48) -> pd.DataFrame:
    """Every object under ``bids/<site>/``: subject prefixes first (delimiter
    listing), then one listing per subject, in parallel."""
    _, subjects = _list_prefix(client, f"{EEG_PREFIX}/bids/{site}/", "/")
    rows = []
    with cf.ThreadPoolExecutor(workers) as pool:
        for objs, _ in pool.map(lambda p: _list_prefix(client, p), subjects):
            rows += objs
    return pd.DataFrame(rows, columns=["key", "bytes"])


def edf_table(listing: pd.DataFrame) -> pd.DataFrame:
    """Object listing -> one row per EDF with BIDS fields and the sidecars of
    its session (``has_channels_tsv``, ``has_annotations``, ``has_xltek_csv``,
    ``has_persyst_csv``) and ``stem`` = file name without ``_eeg.edf``."""
    parsed = pd.DataFrame([parse_key(k) or {} for k in listing["key"]], index=listing.index)
    df = pd.concat([listing, parsed], axis=1).dropna(subset=["file"])
    side = df.assign(
        has_channels_tsv=df["file"].str.endswith("_channels.tsv"),
        has_annotations=df["file"].str.endswith("_annotations.csv"),
        has_xltek_csv=df["file"].str.endswith("_Xltek.csv"),
        has_persyst_csv=df["file"].str.contains("Persyst", case=False),
    ).groupby(["site", "subject", "session"])[
        ["has_channels_tsv", "has_annotations", "has_xltek_csv", "has_persyst_csv"]].any()
    edf = df[df["file"].str.lower().str.endswith("_eeg.edf")].copy()
    m = edf["file"].str.extract(_EDF_RE)
    edf["task"] = m["task"].astype("string")
    edf["run"] = m["run"].astype("string")
    edf["stem"] = edf["file"].str.replace(r"_eeg\.edf$", "", regex=True, flags=re.IGNORECASE)
    edf = edf.join(side, on=["site", "subject", "session"])
    return edf.drop(columns=["file"]).reset_index(drop=True)


# -------------------------------------------------------------------- headers
def fetch_header(client, key: str):
    """(main header, signal block) of an EDF on the access point: one range GET
    covers up to 64 signals, a second one fetches the rest of a larger header."""
    first = 65 * EDF_HEADER_BYTES
    data = client.get_object(Bucket=ACCESS_POINT, Key=key, Range=f"bytes=0-{first - 1}")["Body"].read()
    size = edf_header_size(data[:EDF_HEADER_BYTES])
    if size > len(data):
        rest = client.get_object(Bucket=ACCESS_POINT, Key=key,
                                 Range=f"bytes={len(data)}-{size - 1}")["Body"].read()
        data += rest
    return data[:EDF_HEADER_BYTES], data[EDF_HEADER_BYTES:size]


def header_row(head: bytes, block: bytes) -> dict:
    """Header fields aligned with ``tuh_metadata.recording_row`` (``sfreq,
    mixed_sfreq, n_signals, duration_sec, missing_channels``) plus the signal
    facts the report needs: EDF type, units, prefilter, physical range and
    resolution of the 10-20 channels, auxiliary channel types, all labels."""
    hdr = parse_edf_signal_header(head, block)
    present = channel_map(hdr.labels, bids_canonical_channel)
    idx = [hdr.labels.index(present[c]) for c in TUH_CHANNELS if c in present]
    sf = sorted({hdr.sfreqs[i] for i in idx})
    labels_up = [lab.upper() for lab in hdr.labels]
    lsb = [(hdr.physical_max[i] - hdr.physical_min[i]) / max(hdr.digital_max[i] - hdr.digital_min[i], 1)
           for i in idx]
    row = {
        "edf_type": head[192:236].decode("latin-1").strip() or "EDF",
        "n_records": hdr.n_records, "record_duration": hdr.record_duration,
        "header_duration_sec": hdr.duration_sec,
        "n_signals": len(hdr.labels),
        "n_annotation_signals": sum(lab.startswith("EDF ANNOTATIONS") for lab in labels_up),
        "n_tuh_channels": len(idx),
        "missing_channels": [c for c in TUH_CHANNELS if c not in present],
        "sfreq": float(sf[-1]) if sf else float("nan"),
        "mixed_sfreq": len(sf) > 1,
        "units": ",".join(sorted({hdr.units[i] for i in idx})),
        "prefilter": max((hdr.prefilters[i] for i in idx), key=len, default=""),
        "phys_min": float(np.nanmax([hdr.physical_min[i] for i in idx])) if idx else float("nan"),
        "phys_max": float(np.nanmin([hdr.physical_max[i] for i in idx])) if idx else float("nan"),
        "lsb_uv": float(np.nanmedian(lsb)) if lsb else float("nan"),
        "labels": "|".join(hdr.labels),
    }
    for name, pattern in AUX_PATTERNS.items():
        row[name] = any(re.search(pattern, lab) for lab in labels_up)
    return row


def _header_job(args):
    client, key = args
    try:
        return key, header_row(*fetch_header(client, key))
    except Exception as exc:                 # one bad file must not stop the scan
        return key, {"error": f"{type(exc).__name__}: {exc}"}


def scan_headers(client, keys: Iterable[str], out: str, *, workers: int = 64,
                 flush_every: int = 5000) -> pd.DataFrame:
    """Header rows for ``keys``, appended to the parquet ``out`` in parts so an
    interrupted scan resumes where it stopped."""
    done = pd.read_parquet(out) if os.path.exists(out) else pd.DataFrame(columns=["key"])
    todo = [k for k in keys if k not in set(done["key"])]
    print(f"headers: {len(done)} cached, {len(todo)} to read", flush=True)
    rows = []

    def flush():
        nonlocal done, rows
        if rows:
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_parquet(out, index=False)
            rows = []

    with cf.ThreadPoolExecutor(workers) as pool:
        for i, (key, row) in enumerate(pool.map(_header_job, [(client, k) for k in todo]), 1):
            rows.append({"key": key, **row})
            if i % flush_every == 0:
                flush()
                print(f"  {i}/{len(todo)}", flush=True)
    flush()
    return done


# ---------------------------------------------------------------------- build
def build_index(meta_dir: str, edfs: Optional[pd.DataFrame] = None,
                headers: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Recording sessions (+ their EDFs and headers when given) joined with the
    report findings, the per-patient ICD-10 / ATC records and the TUH-aligned
    clinical flags. One row per EDF (per session when ``edfs`` is None)."""
    rec = load_recordings(meta_dir)
    rep = load_reports(meta_dir)
    rep = pd.concat([rep, finding_flags(rep)], axis=1)
    idx = rec.merge(rep, on=["site", "patient_id", "session"], how="left")
    pats = load_patients(meta_dir)
    idx = idx.merge(pats, on=["site", "patient_id"], how="left")
    idx = idx.merge(load_icd(meta_dir), on=["site", "patient_id"], how="left")
    idx = idx.merge(load_atc(meta_dir), on=["site", "patient_id"], how="left")
    if edfs is not None:
        idx = idx.merge(edfs, on=["site", "subject", "session"], how="left")
    else:
        idx["stem"] = "sub-" + idx["subject"] + "_ses-" + idx["session"]
    if headers is not None and len(headers):
        idx = idx.merge(headers, on="key", how="left")
    idx = pd.concat([idx, tuh_aligned_flags(idx)], axis=1)
    idx["corpus"] = "heedb"
    return idx.reset_index(drop=True)


def download_list(listing: pd.DataFrame, meta_dir: str) -> pd.DataFrame:
    """Every object (EDF + BIDS sidecars + annotation CSVs) of the adult routine
    sessions of the main sites: the one-pass copy set of the onboarding plan."""
    rec = load_recordings(meta_dir)
    want = rec[rec["routine"] & rec["adult"] & rec["site"].isin(MAIN_SITES)][["site", "subject", "session"]]
    parsed = pd.DataFrame([parse_key(k) or {} for k in listing["key"]], index=listing.index)
    objs = pd.concat([listing, parsed[["site", "subject", "session"]]], axis=1)
    return objs.merge(want, on=["site", "subject", "session"], how="inner").reset_index(drop=True)


def convert_scan(index: pd.DataFrame, sample: Optional[int] = None, seed: int = 0) -> pd.DataFrame:
    """The adult routine EDFs of the main sites with a scanned header, as a
    ``make_TUH_npy`` work list (TUH scan columns): ``path`` is relative to the
    raw mirror root (``EEG/`` dropped), ``duration_sec`` from the EDF header.
    ``sample``: a stratified pilot of that many (site x report label, one EDF
    per patient)."""
    sel = index[index["site"].isin(MAIN_SITES) & index["routine"].fillna(False)
                & index["adult"].fillna(False) & index["key"].notna() & index["n_signals"].notna()]
    out = pd.DataFrame({
        "stem": sel["stem"], "path": sel["key"].str.slice(len(EEG_PREFIX) + 1),
        "corpus": "heedb", "site": sel["site"], "subject": sel["subject"],
        "session": sel["session"], "token": sel["run"],
        "age": sel["age"], "raw_age": sel["age"], "sex": sel["sex"], "year": sel["year"],
        "label": sel["label"], "abnormal": sel["abnormal_report"].astype("Int8"),
        "duration_sec": sel["header_duration_sec"], "sfreq": sel["sfreq"],
        "n_tuh_channels": sel["n_tuh_channels"],
    }).drop_duplicates("stem").reset_index(drop=True)
    if sample:
        cells = out.assign(lab=out["label"].fillna("unlabelled")).drop_duplicates("subject")
        groups = cells.groupby(["site", "lab"])
        per = int(np.ceil(sample / max(groups.ngroups, 1)))
        out = (groups.apply(lambda g: g.sample(n=min(per, len(g)), random_state=seed))
               .reset_index(drop=True).drop(columns="lab").head(sample))
    return out


def split_of(patient_id, seed: int = 12345, val_fraction: float = 0.1, test_fraction: float = 0.1) -> str:
    """train / val / test of one patient from ``sha256("heedb:<seed>:<patient id>")``
    -- the same rule as ``tueg_patient_split.hash_split`` plus a held-out test
    band, stable under adding or removing other patients. Keyed by the BDSP
    patient id, not the BIDS subject, because a patient can have sessions at two
    sites (S0001 and S0002 share ids)."""
    u = int.from_bytes(hashlib.sha256(f"heedb:{seed}:{int(patient_id)}".encode()).digest(), "big") / 2 ** 256
    return "test" if u < test_fraction else ("val" if u < test_fraction + val_fraction else "train")


def patient_split(index: pd.DataFrame, seed: int = 12345, val_fraction: float = 0.1,
                  test_fraction: float = 0.1) -> dict:
    """``{"patients": {pid: split}, "subjects": {BIDS subject: split}, counts}``
    for every patient in ``index``; asserts that no patient spans two splits."""
    pids = index["patient_id"].dropna().astype(int).unique()
    patients = {str(p): split_of(p, seed, val_fraction, test_fraction) for p in sorted(pids)}
    subj = index.dropna(subset=["patient_id"]).drop_duplicates("subject")
    subjects = {s: patients[str(int(p))] for s, p in zip(subj["subject"], subj["patient_id"])}
    counts = {k: sum(v == k for v in patients.values()) for k in ("train", "val", "test")}
    return {"version": 1, "rule": "sha256('heedb:<seed>:<patient_id>') / 2^256: "
            "[0, test) test, [test, test+val) val, else train",
            "seed": seed, "val_fraction": val_fraction, "test_fraction": test_fraction,
            "counts": counts, "patients": patients, "subjects": subjects}


# ------------------------------------------------------------------------ CLI
def _paths(root):
    return {"meta": os.path.join(root, "metadata"),
            "listing": os.path.join(root, "scan", "listing.parquet"),
            "edfs": os.path.join(root, "scan", "edfs.parquet"),
            "headers": os.path.join(root, "scan", "headers.parquet"),
            "index": os.path.join(root, "index", "heedb_index.parquet"),
            "download": os.path.join(root, "scan", "download_list.parquet")}


def select_keys(edfs: pd.DataFrame, meta_dir: str, which: str, limit: Optional[int] = None,
                seed: int = 0) -> List[str]:
    """EDF keys of a selection: ``adult_routine`` (main sites), ``all``, or
    ``sample`` (``limit`` random EDFs of every service, for the comparison)."""
    rec = load_recordings(meta_dir)
    joined = edfs.merge(rec, on=["site", "subject", "session"], how="left")
    if which == "adult_routine":
        sel = joined[joined["routine"].fillna(False) & joined["adult"].fillna(False)
                     & joined["site"].isin(MAIN_SITES)]
    elif which == "all":
        sel = joined
    elif which == "sample":
        sel = joined.sample(n=min(limit or 2000, len(joined)), random_state=seed)
    else:
        raise ValueError(which)
    keys = sel["key"].tolist()
    return keys[:limit] if limit and which != "sample" else keys


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--profile", default=DEFAULT_PROFILE)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("metadata", help="copy the metadata CSVs to <root>/metadata")
    ls = sub.add_parser("list", help="list the BIDS tree -> scan/listing.parquet + edfs.parquet")
    ls.add_argument("--sites", nargs="+", default=None)
    hd = sub.add_parser("headers", help="EDF headers by range GET -> scan/headers.parquet")
    hd.add_argument("--select", default="adult_routine", choices=["adult_routine", "all", "sample"])
    hd.add_argument("--limit", type=int, default=None)
    hd.add_argument("--workers", type=int, default=64)
    sub.add_parser("build", help="join everything -> index/heedb_index.parquet")
    sub.add_parser("download-list", help="adult routine objects -> scan/download_list.parquet")
    cs = sub.add_parser("convert-scan", help="make_TUH_npy work list -> scan/convert_scan[_pilot<N>].parquet")
    cs.add_argument("--sample", type=int, default=None, help="stratified pilot of N EDFs")
    ps = sub.add_parser("patient-split", help="train/val/test per patient -> heedb_patient_split.json")
    ps.add_argument("--seed", type=int, default=12345)
    ps.add_argument("--val-fraction", type=float, default=0.1)
    ps.add_argument("--test-fraction", type=float, default=0.1)
    args = p.parse_args(argv)
    paths = _paths(args.root)
    for path in paths.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)

    if args.command == "metadata":
        files = fetch_metadata(s3_client(args.profile), args.root)
        print(f"{len(files)} metadata files in {paths['meta']}")
    elif args.command == "list":
        client = s3_client(args.profile)
        _, sites = _list_prefix(client, f"{EEG_PREFIX}/bids/", "/")
        sites = args.sites or [s.rstrip("/").split("/")[-1] for s in sites]
        parts = []
        for site in sites:
            part = list_site(client, site)
            print(f"{site}: {len(part)} objects, {part['bytes'].sum() / 1e12:.2f} TB", flush=True)
            parts.append(part)
        listing = pd.concat(parts, ignore_index=True)
        listing.to_parquet(paths["listing"], index=False)
        edf_table(listing).to_parquet(paths["edfs"], index=False)
    elif args.command == "headers":
        edfs = pd.read_parquet(paths["edfs"])
        keys = select_keys(edfs, paths["meta"], args.select, args.limit)
        scan_headers(s3_client(args.profile), keys, paths["headers"], workers=args.workers)
    elif args.command == "build":
        edfs = pd.read_parquet(paths["edfs"]) if os.path.exists(paths["edfs"]) else None
        headers = pd.read_parquet(paths["headers"]) if os.path.exists(paths["headers"]) else None
        index = build_index(paths["meta"], edfs, headers)
        index.to_parquet(paths["index"], index=False)
        print(json.dumps({"rows": len(index), "sessions": int(index[["subject", "session"]].drop_duplicates().shape[0]),
                          "patients": int(index["subject"].nunique())}))
    elif args.command == "patient-split":
        sp = patient_split(pd.read_parquet(paths["index"]), args.seed, args.val_fraction, args.test_fraction)
        out = os.path.join(args.root, "index", "heedb_patient_split.json")
        with open(out, "w") as fh:
            json.dump(sp, fh)
        print(json.dumps({"out": out, **sp["counts"]}))
    elif args.command == "convert-scan":
        cs = convert_scan(pd.read_parquet(paths["index"]), args.sample)
        name = f"convert_scan_pilot{args.sample}.parquet" if args.sample else "convert_scan.parquet"
        cs.to_parquet(os.path.join(args.root, "scan", name), index=False)
        print(json.dumps({"edfs": len(cs), "hours": round(cs["duration_sec"].sum() / 3600),
                          "by_site": cs["site"].value_counts().to_dict()}))
    elif args.command == "download-list":
        dl = download_list(pd.read_parquet(paths["listing"]), paths["meta"])
        dl.to_parquet(paths["download"], index=False)
        print(json.dumps({"objects": len(dl), "sessions": int(dl[["subject", "session"]].drop_duplicates().shape[0]),
                          "TB": round(dl["bytes"].sum() / 1e12, 3)}))


if __name__ == "__main__":
    sys.exit(main())
