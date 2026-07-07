#!/usr/bin/env python3
"""Deterministic design-review metrics — a thin wrapper over `radon` and `ruff`.

Self-contained (no repo-root dependencies). Reports McCabe cyclomatic
complexity and Maintainability Index (via radon) and lint errors (via ruff),
applies the canonical gates, and exits non-zero if any hard gate is breached.

Gates (single source of truth for the `architect` + `design_reviewer` skills):
    McCabe: warn > 7, reject > 12
    Maintainability Index: warn < 65

Usage:
    python design_review_metrics.py <path> [--format text|json] [--strict]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass, field

MCCABE_WARN = 7
MCCABE_REJECT = 12
MI_WARN = 65.0


@dataclass
class Report:
    max_mccabe: int = 0
    worst_function: str = ""
    min_mi: float = 100.0
    worst_module: str = ""
    lint_errors: int = 0
    findings: list[dict] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.max_mccabe > MCCABE_REJECT:
            return "reject"
        if self.max_mccabe > MCCABE_WARN or self.min_mi < MI_WARN or self.lint_errors:
            return "revise"
        return "approve"


def _run(cmd: list[str]) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None


def collect_radon_cc(path: str, rep: Report) -> None:
    if not shutil.which("radon"):
        rep.degraded.append("radon not installed (pip install radon) — McCabe skipped")
        return
    proc = _run(["radon", "cc", "-j", path])
    if proc is None or proc.returncode != 0:
        rep.degraded.append("radon cc failed")
        return
    for module, blocks in json.loads(proc.stdout or "{}").items():
        for b in blocks:
            cc = b.get("complexity", 0)
            if cc > rep.max_mccabe:
                rep.max_mccabe = cc
                rep.worst_function = f"{module}:{b.get('lineno')} {b.get('name')}"
            if cc > MCCABE_WARN:
                rep.findings.append({
                    "layer": "complexity",
                    "severity": "reject" if cc > MCCABE_REJECT else "warn",
                    "location": f"{module}:{b.get('lineno')}",
                    "issue": f"McCabe {cc} on {b.get('name')}",
                    "suggested_fix": "extract functions / flatten branching",
                })


def collect_radon_mi(path: str, rep: Report) -> None:
    if not shutil.which("radon"):
        rep.degraded.append("radon not installed — Maintainability Index skipped")
        return
    proc = _run(["radon", "mi", "-j", path])
    if proc is None or proc.returncode != 0:
        rep.degraded.append("radon mi failed")
        return
    for module, data in json.loads(proc.stdout or "{}").items():
        mi = data.get("mi", 100.0) if isinstance(data, dict) else 100.0
        if mi < rep.min_mi:
            rep.min_mi = mi
            rep.worst_module = module
        if mi < MI_WARN:
            rep.findings.append({
                "layer": "maintainability",
                "severity": "warn",
                "location": module,
                "issue": f"Maintainability Index {mi:.1f} < {MI_WARN}",
                "suggested_fix": "reduce size/complexity; split module",
            })


def collect_ruff(path: str, rep: Report) -> None:
    if not shutil.which("ruff"):
        rep.degraded.append("ruff not installed (pip install ruff) — lint skipped")
        return
    proc = _run(["ruff", "check", "--output-format", "json", path])
    if proc is None:
        rep.degraded.append("ruff failed")
        return
    try:
        issues = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        rep.degraded.append("ruff produced no parseable output")
        return
    rep.lint_errors = len(issues)
    for i in issues[:50]:
        rep.findings.append({
            "layer": "hygiene",
            "severity": "warn",
            "location": f"{i.get('filename')}:{(i.get('location') or {}).get('row')}",
            "issue": f"{i.get('code')} {i.get('message')}",
            "suggested_fix": "run `ruff check --fix`",
        })


def build_report(path: str) -> Report:
    rep = Report()
    collect_radon_cc(path, rep)
    collect_radon_mi(path, rep)
    collect_ruff(path, rep)
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="file or directory to analyze")
    ap.add_argument("--format", choices=["text", "json"], default="text")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero on any degraded metric too")
    args = ap.parse_args(argv)

    rep = build_report(args.path)
    payload = {
        "verdict": rep.verdict,
        "metrics": {
            "max_mccabe": rep.max_mccabe,
            "worst_function": rep.worst_function,
            "min_mi": round(rep.min_mi, 1),
            "worst_module": rep.worst_module,
            "lint_errors": rep.lint_errors,
        },
        "findings": rep.findings,
        "degraded": rep.degraded,
    }

    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(f"verdict: {rep.verdict}")
        print(f"  max McCabe : {rep.max_mccabe}  ({rep.worst_function or 'n/a'})")
        print(f"  min MI     : {rep.min_mi:.1f}  ({rep.worst_module or 'n/a'})")
        print(f"  lint errors: {rep.lint_errors}")
        for f in rep.findings:
            print(f"  [{f['severity']}] {f['layer']}: {f['issue']} @ {f['location']}")
        for d in rep.degraded:
            print(f"  (degraded) {d}")

    if rep.max_mccabe > MCCABE_REJECT:
        return 2
    if args.strict and rep.degraded:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
