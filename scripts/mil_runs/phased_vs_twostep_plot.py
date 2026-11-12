#!/usr/bin/env python
"""Figure 13.1 of docs/mil_stage1_results.md + its CSV: test MAE per cohort over epochs, one run (fold mean ± SD) vs two-step."""
import collections, glob, json, re, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from labram.eval.age_plots import assert_max_columns, panel_labels
# usage: phased_vs_twostep_plot.py <logs dir with cw4/ and twostep/> <figure.png> <rows.csv>
S, FIG, CSV = sys.argv[1:4]
COH = ["all", "harvard_normal", "harvard_abnormal", "harvard_unlabelled", "temple_normal", "temple_abnormal"]
def parse(path):
    out, ep = {}, -1
    for line in open(path).read().splitlines():
        m = re.search(r"Epoch: \[(\d+)\]  \[\s*\d+/", line) or re.search(r"MIL epoch (\d+):", line)
        if m: ep = int(m.group(1))
        m = re.search(r"\] Test: recording MAE ([\d.]+)", line)
        if m: out.setdefault(ep, {})["all"] = float(m.group(1))
        m = re.search(r"\] Test EEG mae by dataset/cohort: (.*)", line)
        if m:
            for k, v in re.findall(r"(\w+)=([\d.]+)", m.group(1)): out.setdefault(ep, {})[k] = float(v)
    return out
rows = []; runs = collections.defaultdict(dict)
for f in sorted(glob.glob(f"{S}/cw4/*.log"), key=lambda p: ("2026-11-12" in p, p)):
    if "22-10-59-546" in f or "l1-k4" in f: continue
    m = re.search(r"labram-mil-p-(l1|w4|w7)-f(\d)-fold", f)
    for e, d in parse(f).items(): runs[(m.group(1).upper(), int(m.group(2)))].setdefault(e, {}).update(d)
for (model, fold), eps in runs.items():
    for e, d in eps.items():
        rows += [{"recipe": "one run", "model": model, "fold": fold, "epoch": e, "cohort": c, "mae": d[c]} for c in COH if c in d]
e2l = {}
for l in open("/data/runs/heedb/e2l/finetune_20261104_215141_883/log.txt"):
    d = json.loads(l)
    if "test_mae" in d and d["epoch"] <= 11:
        e2l[d["epoch"]] = {"all": d["test_mae"], **{c: d.get(f"test_{c}_mae") for c in COH[1:]}}
L1 = glob.glob("/data/runs/heedb/mil_local/L1_abmil_2*/log/run.log")[0]
for model, path in (("L1", L1), ("W4", f"{S}/twostep/labram-mil-w4.log"), ("W7", f"{S}/twostep/labram-mil-w7.log")):
    for e, d in e2l.items():
        rows += [{"recipe": "two-step", "model": model, "fold": 0, "epoch": e, "cohort": c, "mae": d[c]} for c in COH]
    for e, d in parse(path).items():
        rows += [{"recipe": "two-step", "model": model, "fold": 0, "epoch": 12 + e, "cohort": c, "mae": d[c]} for c in COH if c in d]
g = (pd.DataFrame(rows).groupby(["recipe", "model", "cohort", "epoch"])["mae"]
     .agg(["mean", "std", "count"]).reset_index())
g.to_csv(CSV, index=False)
TITLES = {"all": "all test (7,612)", "harvard_normal": "Harvard normal (1,518)", "harvard_abnormal": "Harvard abnormal (2,253)",
          "harvard_unlabelled": "Harvard unlabelled (3,567)", "temple_normal": "TUAB normal (149)", "temple_abnormal": "TUAB abnormal (125)"}
C = {"L1": "C0", "W4": "C1", "W7": "C2"}
fig, axes = plt.subplots(3, 2, figsize=(14, 13.5))
for ax, c in zip(axes.flat, COH):
    for m, col in C.items():
        o = g[(g.recipe == "one run") & (g.model == m) & (g.cohort == c)].sort_values("epoch")
        t = g[(g.recipe == "two-step") & (g.model == m) & (g.cohort == c)].sort_values("epoch")
        ax.plot(o.epoch, o["mean"], color=col, lw=1.8, marker="o", ms=3, label=f"{m} one run (fold mean)")
        ax.fill_between(o.epoch, o["mean"] - o["std"].fillna(0), o["mean"] + o["std"].fillna(0), color=col, alpha=0.15, lw=0)
        ax.plot(t.epoch, t["mean"], color=col, lw=1.4, ls="--", label=f"{m} two-step")
    ax.axvspan(9.5, 11.5, color="0.85", alpha=0.5, lw=0); ax.axvline(11.5, color="0.5", lw=0.8, ls=":")
    lo = g[(g.cohort == c) & (g.epoch >= 8)]["mean"].min(); hi = g[(g.cohort == c) & (g.epoch >= 4)]["mean"].max()
    ax.set_ylim(lo - 0.3, hi + 0.3); ax.set_xlim(3.5, 15.5)
    ax.set_xlabel("epoch (one run: 0-9 window, shaded 10-11 attention only, 12-15 joint;\ntwo-step: 0-11 E2L, 12-15 bag warm start)", fontsize=8)
    ax.set_ylabel("test recording MAE (y)"); ax.set_title(TITLES[c])
axes[0][0].legend(fontsize=7, ncol=2)
fig.tight_layout(); panel_labels(list(axes.flat)); assert_max_columns(fig); fig.savefig(FIG, dpi=110)
print(g[g.epoch == 15].pivot_table(index="cohort", columns=["recipe", "model"], values="count"))
