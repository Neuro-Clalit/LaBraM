#!/usr/bin/env python
"""Audit the calibration fields of a set of EDF files from their headers alone.

EDF stores 16-bit integers and scales them per signal with the header's
physical and digital minimum / maximum:

    x_uV = (d - d_min) * (p_max - p_min) / (d_max - d_min) + p_min

A wrong or unusual physical range reads as a gain difference, and a gain
difference reads as age to the brain-age models (``docs/chbp_domain_shift.md``).
Per file this script records, over its EEG signals (every signal except
``--skip`` labels): number of signals and records, duration, sampling rate,
physical dimension, the distinct physical and digital ranges, the quantisation
step ``(p_max - p_min) / (d_max - d_min)`` in physical units, the prefilter
text, and flags: a range that is not finite, zero or inverted; ranges that
differ between the EEG signals of one file; a step above ``--max-step``. No
signal is read and nothing here uses an age.

    python scripts/edf_header_audit.py --glob '/data/.../sub-*/ses-*/eeg/*_eeg.edf' \\
        --out docs/data/external_eval/dortmund/edf_headers.csv
"""
import argparse
import glob
import math
import os

import pandas as pd

FIELDS = [("label", 16), ("transducer", 80), ("dimension", 8), ("pmin", 8), ("pmax", 8), ("dmin", 8), ("dmax", 8),
          ("prefilter", 80), ("nsamp", 8), ("reserved", 32)]


def read_header(path):
    """Fixed header and the per-signal fields of an EDF file (strings, stripped)."""
    with open(path, "rb") as fh:
        h = fh.read(256)
        ns = int(h[252:256])
        sig = {name: [fh.read(width).decode("latin1").strip() for _ in range(ns)] for name, width in FIELDS}
    return {"patient": h[8:88].decode("latin1").strip(), "recording": h[88:168].decode("latin1").strip(),
            "n_records": int(h[236:244]), "record_sec": float(h[244:252]), "n_signals": ns, "signals": sig}


def _num(text):
    try:
        return float(text)
    except ValueError:
        return float("nan")


def audit(path, skip=("status",), max_step=1.0):
    h = read_header(path)
    s = h["signals"]
    eeg = [i for i, name in enumerate(s["label"]) if name.lower() not in skip]
    pmin, pmax = [_num(s["pmin"][i]) for i in eeg], [_num(s["pmax"][i]) for i in eeg]
    dmin, dmax = [_num(s["dmin"][i]) for i in eeg], [_num(s["dmax"][i]) for i in eeg]
    step = [(b - a) / (d - c) if d != c else float("nan") for a, b, c, d in zip(pmin, pmax, dmin, dmax)]
    bad = [not (math.isfinite(a) and math.isfinite(b)) or b <= a for a, b in zip(pmin, pmax)]
    rate = {int(s["nsamp"][i]) / h["record_sec"] for i in eeg} if h["record_sec"] else {float("nan")}
    return {"file": os.path.basename(path), "n_signals": h["n_signals"], "n_eeg": len(eeg),
            "duration_sec": h["n_records"] * h["record_sec"], "sfreq": max(rate), "n_sfreq": len(rate),
            "dimension": ",".join(sorted({s["dimension"][i] for i in eeg})),
            "pmax_min": min(pmax), "pmax_max": max(pmax), "pmin_min": min(pmin), "pmin_max": max(pmin),
            "dmin": min(dmin), "dmax": max(dmax), "step_min": min(step), "step_max": max(step),
            "prefilter": " | ".join(sorted({s["prefilter"][i] for i in eeg})),
            "patient": h["patient"], "recording": h["recording"],
            "flag_invalid_range": any(bad), "flag_ranges_differ": len(set(zip(pmin, pmax))) > 1,
            "flag_coarse_step": max(step) > max_step if all(math.isfinite(v) for v in step) else True}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--glob", required=True)
    ap.add_argument("--skip", nargs="*", default=["status"], help="signal labels that are not EEG (lower case)")
    ap.add_argument("--max-step", type=float, default=1.0, help="flag a quantisation step above this (physical units)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    files = sorted(glob.glob(args.glob))
    df = pd.DataFrame([audit(f, tuple(args.skip), args.max_step) for f in files])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} files")
    for col in ("n_signals", "n_eeg", "sfreq", "dimension", "dmin", "dmax", "prefilter", "patient", "recording"):
        print(f"  {col}: {df[col].value_counts().head(4).to_dict()}")
    q = [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]
    print("  physical max (EEG):", df["pmax_max"].quantile(q).round(1).tolist())
    print("  quantisation step:", df["step_max"].quantile(q).round(4).tolist())
    print("  duration (s):", df["duration_sec"].quantile(q).round(0).tolist())
    for flag in ("flag_invalid_range", "flag_ranges_differ", "flag_coarse_step"):
        print(f"  {flag}: {int(df[flag].sum())}")
    print(args.out)


if __name__ == "__main__":
    main()
