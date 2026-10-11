#!/usr/bin/env python
"""NEDC's TUEG clinical metadata sheet -> a clean per-session table.

TUH no longer distributes EEG reports; NEDC instead shares a spreadsheet
(``metadata_v12 ...xlsm``, sheet ``TUEG``) with one row per session
(``<subject>_s<NNN>``): 15 Y/N diagnosis flags, free-text "other diagnosis",
a medication list, an "abnormal due to" finding and FIRDA/OIRDA/TIRDA/GPD/LPD
flags. This module normalises it so it can be joined onto the recording index
(``dataset_maker.tuh_index``) on ``subject`` + ``session``:

* ids lower-cased (the pre-release sheet has ~1.3k auto-capitalised ids);
* Y/N -> bool, "abnormal due to" non-empty -> ``abnormal_report`` (agrees with
  TUAB's normal/abnormal folder on 94.6% of shared sessions);
* medications -> drug-class flags (sedatives/anaesthetics, benzodiazepines,
  antiseizure, antipsychotics, opioids), since several shift the EEG spectrum.

    python -m dataset_maker.tuh_clinical report --sheet metadata_v12.xlsm \\
        --index tuh_index.parquet --out clinical_report.json
"""
import argparse
import json
import re
import sys

import numpy as np
import pandas as pd

DIAGNOSES = {
    "Electrographic seizure": "dx_electrographic_seizure", "Clinical seizure": "dx_clinical_seizure",
    "Stroke": "dx_stroke", "Epilepsy": "dx_epilepsy", "Parkinson's": "dx_parkinsons",
    "Alzheimer's": "dx_alzheimers", "TBI": "dx_tbi", "Schizophrenia": "dx_schizophrenia",
    "Dementia": "dx_dementia", "Substance abuse / dependence": "dx_substance",
    "Anxiety": "dx_anxiety", "Depression": "dx_depression", "Bipolar disorder": "dx_bipolar",
    "ADHD": "dx_adhd", "Multiple sclerosis": "dx_ms",
}
FINDINGS = {"FIRDA": "f_firda", "OIRDA": "f_oirda", "TIRDA": "f_tirda", "GPD": "f_gpd", "LPD": "f_lpd"}
# Neurological diagnoses that make a session unsuitable for a normative
# brain-age model (psychiatric ones are kept as covariates).
NEURO = ("dx_electrographic_seizure", "dx_clinical_seizure", "dx_stroke", "dx_epilepsy",
         "dx_parkinsons", "dx_alzheimers", "dx_tbi", "dx_dementia", "dx_ms")

# Drug classes by brand/generic name (lower case; the sheet's spelling varies,
# e.g. 'Keppraa'). Matched as whole tokens of the comma-separated list.
DRUG_CLASSES = {
    "med_anesthetic": ["propofol", "diprivan", "ketamine", "precedex", "dexmedetomidine",
                       "pentobarbital", "etomidate"],
    "med_benzodiazepine": ["ativan", "lorazepam", "versed", "midazolam", "valium", "diazepam",
                           "klonopin", "clonazepam", "xanax", "alprazolam", "restoril"],
    "med_barbiturate": ["phenobarbital", "luminal", "pentobarbital", "fioricet"],
    "med_opioid": ["fentanyl", "morphine", "dilaudid", "hydromorphone", "oxycodone",
                   "percocet", "vicodin", "methadone", "tramadol", "norco"],
    "med_antiseizure": ["keppra", "keppraa", "levetiracetam", "dilantin", "phenytoin",
                        "fosphenytoin", "depakote", "depakene", "valproate", "valproic acid",
                        "tegretol", "carbamazepine", "lamictal", "lamotrigine", "topamax",
                        "topiramate", "vimpat", "lacosamide", "trileptal", "oxcarbazepine",
                        "zonegran", "zonisamide", "neurontin", "gabapentin", "lyrica",
                        "pregabalin", "onfi", "clobazam", "banzel", "felbatol"],
    "med_antipsychotic": ["haldol", "haloperidol", "seroquel", "quetiapine", "risperdal",
                          "risperidone", "zyprexa", "olanzapine", "abilify", "aripiprazole",
                          "geodon", "clozapine", "clozaril"],
}
SEDATING = ("med_anesthetic", "med_benzodiazepine", "med_barbiturate", "med_opioid")
_SESSION_RE = re.compile(r"^[a-z]{8}_s\d{3}$")


def _yn(col):
    return col.astype(str).str.strip().str.upper().eq("Y")


def _tokens(text):
    if not isinstance(text, str):
        return set()
    return {t.strip().lower() for t in re.split(r"[,;/]", text) if t.strip()}


def load_sheet(path, sheet="TUEG"):
    """The sheet as one clean row per session (see module doc)."""
    raw = pd.read_excel(path, sheet_name=sheet)
    raw.columns = [c.strip() for c in raw.columns]
    ids = raw["Subject ID"].astype(str).str.strip().str.lower()
    keep = ids.str.match(_SESSION_RE)               # drops the trailing totals row
    raw, ids = raw[keep], ids[keep]
    out = pd.DataFrame({"subject": ids.str.split("_").str[0], "session": ids.str.split("_").str[1]})
    for src, dst in {**DIAGNOSES, **FINDINGS}.items():
        out[dst] = _yn(raw[src]).values
    out["other_diagnosis"] = raw["Other diagnosis"].values
    meds = raw["Major Medications (Comma Separated List)"]
    out["medications"] = meds.values
    toks = meds.map(_tokens)
    for cls, names in DRUG_CLASSES.items():
        out[cls] = toks.map(lambda ts, n=set(names): bool(ts & n)).values
    out["abnormal_report"] = raw["abnormal due to"].notna().values
    out["abnormal_due_to"] = raw["abnormal due to"].values
    out["any_neuro_dx"] = out[list(NEURO)].any(axis=1)
    out["sedated"] = out[list(SEDATING)].any(axis=1)
    dup = out.duplicated(["subject", "session"], keep="first")
    return out[~dup].reset_index(drop=True)


FLAGS = (list(DIAGNOSES.values()) + list(FINDINGS.values()) + list(DRUG_CLASSES)
         + ["abnormal_report", "any_neuro_dx", "sedated"])


def join_index(index, sheet):
    """Recording index (one row per EDF) + session-level clinical columns.

    Flags of recordings whose session is not in the sheet are <NA> (nullable
    boolean), never False: absence from the sheet is not evidence of health."""
    out = index.merge(sheet, on=["subject", "session"], how="left", indicator="_in_sheet")
    out["in_sheet"] = out.pop("_in_sheet").eq("both")
    for col in FLAGS:
        out[col] = out[col].astype("boolean")
    return out


def normative_mask(df):
    """Sessions suitable for a normative (healthy-ageing) brain-age model:
    report-normal, no neurological diagnosis, no sedating drugs, adult age."""
    return (df["in_sheet"] & ~df["abnormal_report"].fillna(True).astype(bool)
            & ~df["any_neuro_dx"].fillna(True).astype(bool)
            & ~df["sedated"].fillna(True).astype(bool)
            & df["age"].between(18, 89))


# ---------------------------------------------------------------------- report
def _decade(age):
    return (age // 10 * 10).clip(upper=90)


def report(df):
    """Age, normal/abnormal and diagnosis structure of the joined table."""
    rec = df[df["in_sheet"] & df["age"].notna()].copy()
    rec[FLAGS] = rec[FLAGS].astype(bool)          # every row is in the sheet
    dec = _decade(rec["age"])
    out = {
        "recordings_in_index": int(len(df)), "recordings_in_sheet": int(df["in_sheet"].sum()),
        "recordings_with_age_and_sheet": int(len(rec)),
        "subjects": int(rec["subject"].nunique()),
        "age": {k: round(float(v), 1) for k, v in rec["age"].describe().items()},
        "abnormal_rate_by_decade": {f"{int(d)}s": round(float(g.mean()), 3)
                                    for d, g in rec.groupby(dec)["abnormal_report"]},
        "recordings_by_decade": {f"{int(d)}s": int(n) for d, n in dec.value_counts().sort_index().items()},
    }
    rows = {}
    for col in list(NEURO) + ["dx_substance", "dx_bipolar", "dx_schizophrenia", "dx_depression",
                              "dx_anxiety", "dx_adhd"] + list(DRUG_CLASSES) + ["sedated"]:
        pos = rec[rec[col]]
        if len(pos) < 5:
            rows[col] = {"n_recordings": int(len(pos))}
            continue
        rows[col] = {
            "n_recordings": int(len(pos)), "n_subjects": int(pos["subject"].nunique()),
            "mean_age": round(float(pos["age"].mean()), 1),
            "age_vs_rest": round(float(pos["age"].mean() - rec.loc[~rec[col], "age"].mean()), 1),
            "abnormal_rate": round(float(pos["abnormal_report"].mean()), 3),
            "prevalence_by_decade": {f"{int(d)}s": round(float(g.mean()), 3)
                                     for d, g in rec.groupby(dec)[col] if len(g) >= 50},
        }
    out["conditions"] = rows
    norm = normative_mask(df)
    n = df[norm]
    out["normative_candidate"] = {
        "recordings": int(norm.sum()), "subjects": int(n["subject"].nunique()),
        "age": {k: round(float(v), 1) for k, v in n["age"].describe().items()},
        "recordings_by_decade": {f"{int(d)}s": int(c) for d, c in _decade(n["age"]).value_counts().sort_index().items()},
    }
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("clean", help="sheet -> clean per-session parquet")
    c.add_argument("--sheet", required=True)
    c.add_argument("--out", required=True)
    r = sub.add_parser("report", help="join with a recording index and summarise")
    r.add_argument("--sheet", required=True)
    r.add_argument("--index", required=True)
    r.add_argument("--out", default=None)
    args = p.parse_args(argv)
    sheet = load_sheet(args.sheet)
    if args.command == "clean":
        sheet.to_parquet(args.out, index=False)
        print(f"wrote {len(sheet)} sessions -> {args.out}")
        return
    rep = report(join_index(pd.read_parquet(args.index), sheet))
    text = json.dumps(rep, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
    print(text)


if __name__ == "__main__":
    sys.exit(main())
