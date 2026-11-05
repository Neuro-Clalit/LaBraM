# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Health records and EEG reports of TUAB, the TUH training cohort and HEEDB.
# ---------------------------------------------------------
"""One harmonised per-recording clinical table per corpus, and the statistics
behind ``notebooks/clinical_records_analytics.ipynb``.

Each row is a recording: ``corpus``, ``stem``, ``subject``, ``age``, ``sex``,
``status`` (normal / abnormal / unknown, the EEG report), one nullable boolean
column per **pathology** (``px_<name>``: True / False / <NA> = not known) and
per **abnormality type** of the report (``ab_<name>``, abnormal recordings with
a report text or findings only).

Sources:

* TUH (TUAB, the TUEG cohort ``all_both_v3_anymiss``): the NEDC TUEG clinical
  sheet (``dataset_maker/tuh_clinical.py``) joined on subject + session. Its
  diagnosis columns are Y/N per session (unknown when the session is not in the
  sheet). The free-text "Other diagnosis" adds categories by keyword
  (:data:`TUH_KEYWORDS`), as **mentions**: True when mentioned, False when a
  sheet session does not mention it -- not evidence of absence, so these rates
  (:data:`KEYWORD_PATHOLOGIES`) are lower bounds. Abnormality types come
  from the "abnormal due to" text (``meta_analysis.REPORT_REASONS``).
* HEEDB: per-patient ICD-10 codes (``heedb_index``: TUH-aligned ``dx_*`` and
  the neurology ``icd_*`` groups; unknown without a health record), the report
  findings (``fg_*`` groups) for abnormality types.

A pathology a corpus does not code at all is <NA> for every recording there
(never 0). HEEDB tables suppress cells under :data:`SMALL_CELL` recordings
(DUA).
"""
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

TUH_ROOT = "/data/datasets/EEG-public/TUH_EEG"
TUH_INDEX = f"{TUH_ROOT}/index/tuh_index.parquet"
TUH_SHEET = f"{TUH_ROOT}/index/nedc/metadata_v12.xlsm.xlsx"
TUAB_META = f"{TUH_ROOT}/tuab/v3.0.1/edf/sidecars/age_metadata.json"
TUH_COHORT = f"{TUH_ROOT}/cohorts/all_both_v3_anymiss"
HEEDB_COHORT = "/data/datasets/HEEDB/v4.1/cohorts/heedb_routine_v1"
SMALL_CELL = 10
AGE_BINS = [18, 30, 40, 50, 60, 70, 80, 120]
AGE_LABELS = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]

# Pathology categories: name -> (TUH dx columns, HEEDB columns). Columns are
# OR-ed; an empty tuple means the corpus does not code the category.
PATHOLOGIES: Dict[str, tuple] = {
    "seizure_epilepsy": (("dx_clinical_seizure", "dx_epilepsy", "dx_electrographic_seizure"),
                         ("dx_clinical_seizure", "dx_epilepsy")),
    "stroke_cerebrovascular": (("dx_stroke",), ("dx_stroke", "icd_cerebrovascular_diseases")),
    "dementia": (("dx_dementia", "dx_alzheimers"), ("dx_dementia", "dx_alzheimers")),
    "parkinson_movement": (("dx_parkinsons",), ("dx_parkinsons", "icd_movement_and_cerebellar_disorders")),
    "multiple_sclerosis": (("dx_ms",), ("dx_ms",)),
    "tbi": (("dx_tbi",), ()),
    "depression": (("dx_depression",), ("dx_depression",)),
    "anxiety": (("dx_anxiety",), ()),
    "bipolar": (("dx_bipolar",), ()),
    "schizophrenia": (("dx_schizophrenia",), ()),
    "substance_use": (("dx_substance",), ()),
    "adhd": (("dx_adhd",), ("dx_adhd",)),
    "headache_migraine": (("kw_headache_migraine",), ("icd_headache_disorders",)),
    "sleep_disorder": (("kw_sleep_disorder",), ("icd_sleep_disorders",)),
    "brain_tumor": (("kw_brain_tumor",), ("icd_intracranial_and_spinal_tumors",)),
    "developmental": (("kw_developmental",), ("icd_developmental_delay",)),
    "cns_infection": (("kw_cns_infection",), ("icd_infections",)),
    "syncope": (("kw_syncope",), ()),
    "altered_mental_status": (("kw_altered_mental_status",), ()),
    "anoxic_injury": (("kw_anoxic_injury",), ()),
    "vascular_risk": (("kw_vascular_risk",), ()),
}
# TUH "Other diagnosis" keywords (lower-case regex), recorded as mentions.
TUH_KEYWORDS = {
    "kw_headache_migraine": r"migr|headache",
    "kw_sleep_disorder": r"sleep apn|\bosa\b|insomnia|narcolep|sleep disorder|parasomnia",
    "kw_brain_tumor": r"tumou?r|glioma|glioblastoma|\bgbm\b|meningioma|astrocytoma|brain mass|brain met",
    "kw_developmental": r"cerebral palsy|developmental|mental retardation|autis|intellectual disab",
    "kw_cns_infection": r"mening|encephalitis|brain abscess",
    "kw_syncope": r"syncop|loss of consciousness",
    "kw_altered_mental_status": r"mental status|confusion|encephalopath|delirium",
    "kw_anoxic_injury": r"anoxi|hypoxic|cardiac arrest",
    "kw_vascular_risk": (r"hypertension|\bhtn\b|diabet|\bdm\b|hyperlipid|\bhld\b|coronary|\bcad\b|"
                         r"atrial fib|\bafib\b|heart failure|\bchf\b"),
}
# Categories TUH knows only from the free text (mention rates, lower bounds).
KEYWORD_PATHOLOGIES = tuple(k for k, (tuh, _) in PATHOLOGIES.items() if tuh and tuh[0].startswith("kw_"))
# Abnormality types: name -> (TUH report-reason category, HEEDB finding groups).
ABNORMAL_TYPES: Dict[str, tuple] = {
    "diffuse_slowing": ("diffuse_slowing", ("fg_generalized_slowing",)),
    "focal_slowing": ("focal_slowing", ("fg_focal_slowing",)),
    "epileptiform": ("epileptiform", ("fg_epileptiform", "fg_periodic_discharges")),
    "seizure": ("seizure", ("fg_seizure",)),
    "suppression_low_voltage": ("suppression", ("fg_burst_suppression_low_voltage",)),
    "pdr_abnormal": ("pdr_abnormal", ()),
    "excess_beta": ("excess_beta", ()),
    "asymmetry": ("asymmetry", ()),
}


def _bool(s: pd.Series) -> pd.Series:
    return s.astype("boolean")


def _any_of(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    """OR of nullable flags: True if any is True, False if all known ones are
    False and at least one is known, <NA> if none is known (or no column)."""
    cols = [c for c in cols if c in df]
    if not cols:
        return pd.Series(pd.NA, index=df.index, dtype="boolean")
    x = df[cols].apply(_bool)
    any_true = x.fillna(False).any(axis=1)
    any_known = x.notna().any(axis=1)
    return pd.Series(np.where(any_true, True, np.where(any_known, False, None)),
                     index=df.index, dtype="boolean")


def age_group(age: pd.Series) -> pd.Series:
    return pd.cut(age, AGE_BINS, right=False, labels=AGE_LABELS)


def _status(label: pd.Series) -> pd.Series:
    return label.where(label.isin(["normal", "abnormal"]), "unknown").astype(str)


# ------------------------------------------------------------------ loading

def tuh_clinical(index_path: str = TUH_INDEX, sheet_path: str = TUH_SHEET) -> pd.DataFrame:
    """Per TUEG stem: the sheet's flags, keyword mentions and report reasons."""
    from dataset_maker import tuh_clinical as tc
    from labram.eval.meta_analysis import report_reason_flags
    c = tc.join_index(pd.read_parquet(index_path), tc.load_sheet(sheet_path))
    c = c.drop_duplicates("stem").reset_index(drop=True)
    in_sheet = c["in_sheet"].fillna(False).astype(bool)
    text = c["other_diagnosis"].where(in_sheet).fillna("").astype(str).str.lower()
    for col, pattern in TUH_KEYWORDS.items():
        hit = text.str.contains(pattern, regex=True)
        # mentioned / not mentioned in a sheet session / unknown outside the sheet
        c[col] = pd.array(np.where(hit, True, np.where(in_sheet, False, None)), dtype="boolean")
    abnormal = c["abnormal_report"].astype("boolean").fillna(False).astype(bool)
    reasons = report_reason_flags(c["abnormal_due_to"].where(abnormal))
    has_reason = abnormal & c["abnormal_due_to"].notna() & c["abnormal_due_to"].astype(str).str.strip().ne("")
    for col in reasons:
        c[f"reason_{col}"] = pd.array(np.where(has_reason, reasons[col], None), dtype="boolean")
    return c


def _tuh_table(meta: Dict[str, dict], corpus: str, clinical: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame([{"stem": s, "subject": m.get("subject"), "age": m.get("age"),
                          "sex": m.get("sex"), "label": m.get("label")} for s, m in meta.items()])
    c = clinical.set_index("stem")
    j = rows.join(c, on="stem", rsuffix="_sheet")
    out = pd.DataFrame({"corpus": corpus, "stem": j["stem"], "subject": j["subject"],
                        "age": pd.to_numeric(j["age"], errors="coerce"),
                        "sex": j["sex"].map({"F": "F", "M": "M"}).fillna("unknown"),
                        "status": _status(j["label"])})
    in_sheet = j["in_sheet"].fillna(False).astype(bool)
    for name, (tuh_cols, _) in PATHOLOGIES.items():
        dx = [k for k in tuh_cols if k.startswith("dx_")]
        kw = [k for k in tuh_cols if k.startswith("kw_")]
        if dx:
            v = _any_of(j, dx)
            out[f"px_{name}"] = v.where(in_sheet, pd.NA)
        elif kw:
            out[f"px_{name}"] = _any_of(j, kw)
        else:
            out[f"px_{name}"] = pd.Series(pd.NA, index=j.index, dtype="boolean")
    abnormal = out["status"].eq("abnormal")
    for name, (reason, _) in ABNORMAL_TYPES.items():
        col = f"reason_{reason}"
        out[f"ab_{name}"] = (j[col] if col in j else pd.Series(pd.NA, index=j.index)).astype("boolean") \
            .where(abnormal, pd.NA)
    out["has_health_record"] = in_sheet
    out["has_abnormal_detail"] = j.get("reason_diffuse_slowing", pd.Series(pd.NA, index=j.index)).notna() & abnormal
    return out.reset_index(drop=True)


def load_tuab(clinical: Optional[pd.DataFrame] = None, meta_path: str = TUAB_META) -> pd.DataFrame:
    meta = json.load(open(meta_path))["recordings"]
    return _tuh_table(meta, "TUAB", clinical if clinical is not None else tuh_clinical())


def load_tuh_cohort(clinical: Optional[pd.DataFrame] = None, root: str = TUH_COHORT) -> pd.DataFrame:
    meta = json.load(open(os.path.join(root, "age_metadata.json")))["recordings"]
    return _tuh_table(meta, "TUH cohort", clinical if clinical is not None else tuh_clinical())


def load_heedb(root: str = HEEDB_COHORT) -> pd.DataFrame:
    lab = pd.read_parquet(os.path.join(root, "labels.parquet"))
    meta = json.load(open(os.path.join(root, "age_metadata.json")))["recordings"]
    out = pd.DataFrame({"corpus": "HEEDB", "stem": lab["stem"],
                        "subject": lab["stem"].map(lambda s: (meta.get(s) or {}).get("subject")),
                        "age": pd.to_numeric(lab["age"], errors="coerce"),
                        "sex": lab["sex"].map({"F": "F", "M": "M"}).fillna("unknown"),
                        "status": _status(lab["label"]), "site": lab["site"]})
    record = lab[[c for c in lab if c.startswith("dx_") and c != "dx_electrographic_seizure"]].notna().any(axis=1)
    for name, (_, heedb_cols) in PATHOLOGIES.items():
        v = _any_of(lab, heedb_cols) if heedb_cols else pd.Series(pd.NA, index=lab.index, dtype="boolean")
        # icd_* is unknown for patients with a record but no neurology code: negative
        out[f"px_{name}"] = v.where(~(record & v.isna()), False).where(record | v.notna(), pd.NA) \
            if heedb_cols else v
    abnormal = out["status"].eq("abnormal")
    has_findings = lab[[c for c in lab if c.startswith("fg_")]].notna().any(axis=1)
    for name, (_, groups) in ABNORMAL_TYPES.items():
        v = _any_of(lab, groups) if groups else pd.Series(pd.NA, index=lab.index, dtype="boolean")
        out[f"ab_{name}"] = v.where(abnormal & has_findings, pd.NA)
    out["has_health_record"] = record
    out["has_abnormal_detail"] = abnormal & has_findings
    return out


def load_all() -> pd.DataFrame:
    clinical = tuh_clinical()
    return pd.concat([load_tuab(clinical), load_tuh_cohort(clinical), load_heedb()], ignore_index=True)


# ------------------------------------------------------------------ statistics

def pathology_names(df: pd.DataFrame) -> List[str]:
    return [c[3:] for c in df if c.startswith("px_")]


def abnormal_type_names(df: pd.DataFrame) -> List[str]:
    return [c[3:] for c in df if c.startswith("ab_")]


def suppress(table: pd.DataFrame, counts: pd.DataFrame, k: int = SMALL_CELL) -> pd.DataFrame:
    """Blank (NaN) every cell whose count is below ``k`` (DUA small-cell rule)."""
    return table.where(counts.reindex_like(table).fillna(0) >= k)


def overview(df: pd.DataFrame) -> pd.DataFrame:
    """Recordings, patients, age, sex and status mix per corpus."""
    rows = []
    for corpus, g in df.groupby("corpus", sort=False):
        rows.append({"corpus": corpus, "recordings": len(g), "patients": g["subject"].nunique(),
                     "age mean": g["age"].mean(), "age SD": g["age"].std(),
                     "female %": 100 * g["sex"].eq("F").mean(),
                     "normal %": 100 * g["status"].eq("normal").mean(),
                     "abnormal %": 100 * g["status"].eq("abnormal").mean(),
                     "status unknown %": 100 * g["status"].eq("unknown").mean(),
                     "health record %": 100 * g["has_health_record"].mean()})
    return pd.DataFrame(rows).set_index("corpus")


def age_sex_status(df: pd.DataFrame, corpus: str) -> pd.DataFrame:
    """Recording counts by age group x sex x status (one corpus)."""
    g = df[(df["corpus"] == corpus) & df["sex"].isin(["F", "M"])]
    t = pd.crosstab(age_group(g["age"]), [g["sex"], g["status"]])
    return t


def prevalence(df: pd.DataFrame, by: Optional[str] = None, k: int = SMALL_CELL) -> pd.DataFrame:
    """Per corpus (and ``by`` group): % positive among the recordings whose
    pathology status is known, Eq. (3.1); blank when fewer than ``k`` known or
    (HEEDB) fewer than ``k`` positives."""
    rows = {}
    keys = ["corpus"] + ([by] if by else [])
    for key, g in df.groupby(keys, sort=bool(by), observed=True):
        key = key if isinstance(key, tuple) else (key,)
        r = {}
        for p in pathology_names(df):
            x = g[f"px_{p}"]
            n, pos = int(x.notna().sum()), int(x.fillna(False).sum())
            ok = n >= k and (key[0] != "HEEDB" or pos == 0 or pos >= k)
            r[p] = 100 * pos / n if ok else np.nan
        rows[key] = r
    out = pd.DataFrame(rows).T
    out.index = pd.MultiIndex.from_tuples(out.index, names=keys) if by else out.index.get_level_values(0)
    return out


def known_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Per corpus and pathology: recordings with the status known and positives."""
    rows = []
    for corpus, g in df.groupby("corpus", sort=False):
        for p in pathology_names(df):
            x = g[f"px_{p}"]
            rows.append({"corpus": corpus, "pathology": p, "known": int(x.notna().sum()),
                         "positive": int(x.fillna(False).sum())})
    return pd.DataFrame(rows)


def abnormal_rate_by_pathology(df: pd.DataFrame, k: int = SMALL_CELL) -> pd.DataFrame:
    """% abnormal EEG report among recordings with / without each pathology
    (status known), Eq. (3.2)."""
    rows = []
    for corpus, g in df.groupby("corpus", sort=False):
        g = g[g["status"].isin(["normal", "abnormal"])]
        for p in pathology_names(df):
            x = g[f"px_{p}"]
            pos, neg = g[x.fillna(False).astype(bool)], g[x.notna() & ~x.fillna(False).astype(bool)]
            if len(pos) < k:
                continue
            rows.append({"corpus": corpus, "pathology": p, "n with": len(pos),
                         "abnormal % with": 100 * pos["status"].eq("abnormal").mean(),
                         "abnormal % without": 100 * neg["status"].eq("abnormal").mean() if len(neg) else np.nan})
    return pd.DataFrame(rows)


def abnormal_types(df: pd.DataFrame, by: Optional[str] = None, k: int = SMALL_CELL) -> pd.DataFrame:
    """% of abnormal recordings (with a report reason / findings) showing each
    abnormality type, Eq. (4.1)."""
    g0 = df[df["has_abnormal_detail"].fillna(False).astype(bool)]
    rows = {}
    keys = ["corpus"] + ([by] if by else [])
    for key, g in g0.groupby(keys, sort=bool(by), observed=True):
        key = key if isinstance(key, tuple) else (key,)
        r = {"n": len(g)}
        for t in abnormal_type_names(df):
            x = g[f"ab_{t}"]
            n, pos = int(x.notna().sum()), int(x.fillna(False).sum())
            ok = n >= k and (key[0] != "HEEDB" or pos == 0 or pos >= k)
            r[t] = 100 * pos / n if ok else np.nan
        rows[key] = r
    out = pd.DataFrame(rows).T
    out.index = pd.MultiIndex.from_tuples(out.index, names=keys) if by else out.index.get_level_values(0)
    return out


def type_by_pathology_lift(df: pd.DataFrame, corpus: str, k: int = SMALL_CELL) -> pd.DataFrame:
    """Lift of each abnormality type in abnormal recordings with a pathology
    over all abnormal recordings of the corpus, Eq. (4.2)."""
    g = df[(df["corpus"] == corpus) & df["has_abnormal_detail"].fillna(False).astype(bool)]
    base = {t: g[f"ab_{t}"].fillna(False).mean() for t in abnormal_type_names(df)}
    rows = {}
    for p in pathology_names(df):
        sel = g[g[f"px_{p}"].fillna(False).astype(bool)]
        if len(sel) < k:
            continue
        rows[p] = {t: (sel[f"ab_{t}"].fillna(False).mean() / base[t]) if base[t] > 0 and
                   sel[f"ab_{t}"].fillna(False).sum() >= (k if corpus == "HEEDB" else 1) else np.nan
                   for t in abnormal_type_names(df)}
    return pd.DataFrame(rows).T


def co_occurrence(df: pd.DataFrame, corpus: str, names: Iterable[str], k: int = SMALL_CELL) -> pd.DataFrame:
    """P(column | row) in %: of the recordings with pathology ``row``, the share
    also having ``column`` (both known), Eq. (3.3)."""
    g = df[df["corpus"] == corpus]
    names = list(names)
    out = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        xa = g[f"px_{a}"].fillna(False).astype(bool)
        for b in names:
            xb = g[f"px_{b}"]
            m = xa & xb.notna()
            pos = int(xb[m].fillna(False).sum())
            out.loc[a, b] = 100 * pos / m.sum() if m.sum() >= k and (corpus != "HEEDB" or pos == 0 or pos >= k) else np.nan
    return out
