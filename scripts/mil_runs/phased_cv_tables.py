#!/usr/bin/env python
"""Rows of Tables 13.2-13.4 of docs/mil_stage1_results.md (phased one-run CV).

Inputs: <logs>/cw4/*.log (CloudWatch logs of the labram-mil-p-* jobs; a resumed
job's log overrides its failed original's epochs) and the downloaded models under
/data/runs/heedb/mil_phased_cv/<model>/fold_<k>/.

    PYTHONPATH=. python scripts/mil_runs/phased_cv_tables.py <logs dir>
"""
import collections, glob, json, re, sys
import numpy as np, pandas as pd
S = sys.argv[1]
sys.path.insert(0, "scripts")
import mil_cohort_summary as mcs

def parse(path):
    out, ep = {}, -1
    for line in open(path).read().splitlines():
        m = re.search(r"Epoch: \[(\d+)\]  \[\s*\d+/", line) or re.search(r"MIL epoch (\d+):", line)
        if m: ep = int(m.group(1))
        m = re.search(r"\] (Val|Test|Test\[temple\]|Test\[lemon\]): recording MAE ([\d.]+)", line)
        if m: out.setdefault(ep, {})[m.group(1)] = float(m.group(2))
    return out

runs = collections.defaultdict(dict)
for f in sorted(glob.glob(f"{S}/cw4/*.log"), key=lambda p: ("2026-11-12" in p, p)):
    if "22-10-59-546" in f: continue
    m = re.search(r"labram-mil-p-(l1-k4|l1|w4|w7)-f(\d)-fold", f)
    for e, d in parse(f).items():
        runs[(m.group(1).upper().replace("-", "_"), int(m.group(2)))].setdefault(e, {}).update(d)
E2L = [10.18, 8.56, 8.47, 8.58, 7.47, 8.44, 7.73, 7.81, 7.02, 7.29, 7.30, 7.00, 7.09, 7.21, 7.08, 7.13]
PH = ["window"] * 10 + ["attention only"] * 2 + ["joint"] * 4
fmt = lambda v: (f"{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f} ({len(v)})" if len(v) > 1 else (f"{v[0]:.2f} (1)" if v else "—"))
print("### 13.2")
for ep in range(16):
    cells, allv = [], []
    for mdl in ("L1", "W4", "W7"):
        v = [d[ep]["Test"] for (m, k), d in runs.items() if m == mdl and ep in d and "Test" in d[ep]]
        cells.append(fmt(v)); allv += v
    b = "**" if ep == 15 else ""
    l0, k4 = runs[("L1", 0)].get(ep, {}).get("Test"), runs[("L1_K4", 0)].get(ep, {}).get("Test")
    c = [f"{b}{x.split(' (')[0]}{b} ({x.split(' (')[1]}" if b and x != "—" else x for x in cells]
    print(f"| {ep} | {PH[ep]} | {' | '.join(c)} | {fmt(allv)} | {E2L[ep]:.2f} | {l0:.2f} / {k4:.2f} |")
print("### 13.3")
final = {}
for (m, k) in sorted(runs, key=lambda x: (["L1", "W4", "W7", "L1_K4"].index(x[0]), x[1])):
    d = runs[(m, k)]
    if 15 not in d or "Test" not in d[15]:
        print(f"| {m} | {k} | — | running | | | |"); continue
    val = {e: x["Val"] for e, x in d.items() if "Val" in x}
    b = min(val, key=val.get); x = d[b]; final[(m, k)] = x
    name = "L1, 4-window bags" if m == "L1_K4" else m
    print(f"| {name} | {k} | {b} | {x['Val']:.2f} | {x['Test']:.2f} | {x.get('Test[temple]', float('nan')):.2f} | {x.get('Test[lemon]', float('nan')):.2f} |")
for mdl in ("L1", "W4", "W7"):
    t = [v["Test"] for (m, k), v in final.items() if m == mdl]; le = [v["Test[lemon]"] for (m, k), v in final.items() if m == mdl]
    print(mdl, "test", fmt(t), "lemon", f"{min(le):.2f}-{max(le):.2f}", "temple", fmt([v["Test[temple]"] for (m, k), v in final.items() if m == mdl]))
print("### 13.4")
meta = mcs.recording_table()
def rec_pred(p):
    w = pd.read_parquet(p, columns=["recording", "alpha", "window_pred", "age"])
    g = w.groupby("recording")
    return (w["alpha"] * w["window_pred"]).groupby(w["recording"]).sum() / g["alpha"].sum(), g["age"].first()
res = {}
for mdl in ("L1", "W4", "W7"):
    P = {}
    for p in sorted(glob.glob(f"/data/runs/heedb/mil_phased_cv/{mdl}/fold_*/cv/fold_*/analysis/test_bag_weights_epoch15.parquet")):
        P[p.split("/fold_")[1][0]], age = rec_pred(p)
    P = pd.DataFrame(P); ens = P.mean(1); m = meta.reindex(P.index)
    groups = {"all (7,612)": np.ones(len(P), bool)}
    for src, lab in (("HEEDB", "Harvard"), ("TUAB", "TUAB")):
        for c in ("normal", "abnormal", "unlabelled"):
            mk = ((m.source == src) & (m.cohort == c)).to_numpy()
            if mk.sum() >= 10: groups[f"{lab} {c} ({mk.sum():,})"] = mk
    for g, mk in groups.items():
        f = [np.abs(P[k][mk] - age[mk]).mean() for k in P]
        res[(mdl, g)] = (np.mean(f), np.std(f, ddof=1), np.abs(ens[mk] - age[mk]).mean(), P.shape[1])
two = {"L1": [6.09, 5.71, 6.79, 5.80, 4.41, 8.13], "W4": [6.10, 5.72, 6.77, 5.83, 4.45, 8.20], "W7": [6.12, 5.83, 6.73, 5.87, 4.36, 8.07]}
gs = list(dict.fromkeys(g for (_, g) in res))
for i, g in enumerate(gs):
    cells = []
    for mdl in ("L1", "W4", "W7"):
        fm, fs, en, n = res[(mdl, g)]; tw = two[mdl][i]
        cells += [f"{fm:.2f} ± {fs:.2f}", f"**{en:.2f}**" if en <= tw else f"{en:.2f}", f"**{tw:.2f}**" if tw < en else f"{tw:.2f}"]
    print(f"| {g} | " + " | ".join(cells) + " |")
print({m: res[(m, gs[0])][3] for m in ("L1", "W4", "W7")}, "folds")
