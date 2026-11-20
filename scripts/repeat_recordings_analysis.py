#!/usr/bin/env python
"""Patients with more than one recording: how many, how far apart in age, and whether
the predicted age follows the age difference (technical report Section 3.6). Aggregates
only (TUH and HEEDB are under data use agreements): no per-patient row is written.

* Training cohort ``tuh_heedb_v1`` per corpus (Temple, Harvard; patient = the cross-site
  HEEDB patient id or the TUH subject): patients, patients with >= 2 recordings, their
  recordings, recordings per patient, age span (last minus first recording, whole years),
  sessions, and for Temple the consistency of the age span with the span of recording
  years and of the sex (header errors or reused subject codes).
* Longitudinal check: for every patient with recordings of different ages in a scored
  set, the change of the predicted age between the youngest and the oldest recording,
  Delta_pred = pred(last) - pred(first), against Delta_age; OLS slope of Delta_pred on
  Delta_age through the origin and Pearson r, mean change per Delta_age band. Sets: the
  Temple and Harvard test patients (first 5 minutes) and the external cohorts with repeat
  sessions (TDBRAIN, Dortmund, SRM; every window), models E2L and P4ens.

    python scripts/repeat_recordings_analysis.py --out-dir docs/data/external_eval/tech_report \\
        --figure docs/tech_report/fig_repeat_recordings.png
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from external_cohort_table import load_meta  # noqa: E402

TRAIN = "/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1"
OUT = "/data/runs/heedb/external_eval"
E = "/data/datasets/EEG-public"
# every report test cohort, for the EEGs-per-person figure (scored adults: recordings in the P4 ensemble's predictions)
TEST_COHORTS = [("Temple test", TRAIN, f"{OUT}/own_5min_models.json", "temple", "recordings"),
                ("Harvard test", TRAIN, f"{OUT}/own_5min_models.json", "harvard", "recordings"),
                ("TDBRAIN", f"{E}/TDBRAIN/cohorts/tdbrain_v1", f"{OUT}/tdbrain_all_models.json", None, "sessions of 2 files"),
                ("LEMON", f"{E}/LEMON/cohorts/lemon_v1", f"{OUT}/lemon_all_models.json", None, "recordings"),
                ("Dortmund", f"{E}/DORTMUND/cohorts/dortmund_v1", f"{OUT}/dortmund_all_models.json", None, "sessions of 4 runs"),
                ("SRM", f"{E}/SRM/cohorts/srm_v1", f"{OUT}/srm_all_models.json", None, "recordings"),
                ("CHBP", f"{E}/CHBP/cohorts/chbp_v1", f"{OUT}/chbp_all_models.json", None, "recordings")]
SETS = [("Temple test", TRAIN, f"{OUT}/own_5min_models.json", "temple"),
        ("Harvard test", TRAIN, f"{OUT}/own_5min_models.json", "harvard"),
        ("TDBRAIN", f"{E}/TDBRAIN/cohorts/tdbrain_v1", f"{OUT}/tdbrain_all_models.json", None),
        ("Dortmund", f"{E}/DORTMUND/cohorts/dortmund_v1", f"{OUT}/dortmund_all_models.json", None),
        ("SRM", f"{E}/SRM/cohorts/srm_v1", f"{OUT}/srm_all_models.json", None)]
BANDS = [(1, 2, "1"), (2, 4, "2–3"), (4, 7, "4–6"), (7, 11, "7–10"), (11, 200, "11+")]
# grid of patients with >= 2 recordings: EEGs per patient x age span (Figure 3(c), (d))
N_BINS = [(2, 3, "2"), (3, 4, "3"), (4, 5, "4"), (5, 6, "5"), (6, 10, "6–9"), (10, 1000, "10+")]
SPAN_BINS = [(0, 1, "0"), (1, 2, "1"), (2, 4, "2–3"), (4, 6, "4–5"), (6, 10, "6–9"), (10, 15, "10–14"), (15, 200, "15+")]
SMALL_CELL = 5                                               # cells with 1-4 patients are shown as "<5"


def training_summary():
    p = pd.read_parquet(os.path.join(TRAIN, "cohort.parquet")).set_index("stem")
    man = pd.DataFrame(json.load(open(os.path.join(TRAIN, "manifest.json")))["recordings"]).set_index("stem")
    p["session"], p["year"], p["sex"] = man["session"], man["year"], man["sex"]
    p["pat"] = [f"h{a}" if c == "heedb" else str(s) for a, s, c in zip(p.patient_id, p.subject, p.corpus)]
    rows, spans, counts, grids = [], {}, {}, {}
    for corpus, name in (("tuh", "Temple"), ("heedb", "Harvard")):
        g = p[p.corpus == corpus]
        per = g.groupby("pat").agg(n=("age", "size"), span=("age", lambda a: a.max() - a.min()),
                                   sessions=("session", "nunique"), yspan=("year", lambda y: y.max() - y.min()),
                                   sexes=("sex", "nunique"))
        rep = per[per.n >= 2]
        s = rep["span"]
        row = {"dataset": name, "recordings": len(g), "patients": len(per), "patients_2plus": len(rep),
               "share_2plus": len(rep) / len(per), "recordings_of_2plus": int(rep.n.sum()),
               "share_recordings_of_2plus": rep.n.sum() / len(g), "rec_per_repeater_median": rep.n.median(),
               "rec_per_repeater_p90": rep.n.quantile(0.9), "rec_per_patient_max": int(per.n.max()),
               "span_mean": s.mean(), "span_median": s.median(), "span_p75": s.quantile(0.75),
               "span_p90": s.quantile(0.9), "span_max": s.max(), "span_0": (s == 0).mean(), "span_le1": (s <= 1).mean(),
               "span_ge5": (s >= 5).mean(), "span_ge10": (s >= 10).mean(),
               "repeaters_one_session": (rep.sessions == 1).mean()}
        if corpus == "tuh":
            row["age_vs_year_span_off_gt2"] = int(((rep.span - rep.yspan).abs() > 2).sum())
            row["two_sexes"] = int((rep.sexes > 1).sum())
        for split in ("train", "val", "test"):
            gs = g[g.split == split].groupby("pat").size()
            row[f"share_2plus_{split}"] = (gs >= 2).mean()
        rows.append(row)
        spans[name], counts[name] = s.to_numpy(), per.n.to_numpy()
        grids[name] = grid(rep)
    return pd.DataFrame(rows), spans, counts, grids


def grid(rep):
    """Patients with >= 2 recordings per (EEGs per patient, age span) cell: rows = span bins, columns = count bins."""
    g = np.zeros((len(SPAN_BINS), len(N_BINS)), int)
    for i, (slo, shi, _) in enumerate(SPAN_BINS):
        for j, (nlo, nhi, _) in enumerate(N_BINS):
            g[i, j] = int(((rep.span >= slo) & (rep.span < shi) & (rep.n >= nlo) & (rep.n < nhi)).sum())
    return g


def longitudinal():
    rows, curves = [], {}
    for name, cdir, ev, family in SETS:
        meta = load_meta(cdir, family, 18.0)
        if family:                                            # the test split of the training cohort
            test = {f.split("/")[-1].rsplit("_", 1)[0] for f in json.load(open(os.path.join(cdir, "age_split.json")))["files"]["test"]}
            meta = meta[meta.index.isin(test)]
            if family == "harvard":                           # cross-site patient id
                pid = pd.read_parquet(os.path.join(cdir, "cohort.parquet")).set_index("stem")["patient_id"]
                meta = meta.assign(subject=pid.reindex(meta.index).astype(str))
        res = json.load(open(ev))
        for model in ("E2L", "P4ens"):
            pred = pd.Series(res[model]["splits"]["test"]["predictions"], dtype=float)
            d = meta[["subject", "age"] + (["session"] if "task" in meta else [])].join(pred.rename("pred"), how="inner")
            if "task" in meta:                                # sessions = mean of their files
                d = d.groupby(["subject", "session"]).agg(age=("age", "first"), pred=("pred", "mean")).reset_index()
            d = d.sort_values("age")
            g = d.groupby("subject").agg(n=("age", "size"), a0=("age", "first"), a1=("age", "last"),
                                         p0=("pred", "first"), p1=("pred", "last"))
            g = g[(g.n >= 2) & (g.a1 > g.a0)]
            if len(g) < 5:
                continue
            da, dp = (g.a1 - g.a0).to_numpy(float), (g.p1 - g.p0).to_numpy(float)
            slope = (da * dp).sum() / (da * da).sum()
            rows.append({"set": name, "model": model, "patients": len(g), "delta_age_mean": da.mean(),
                         "delta_age_median": np.median(da), "delta_pred_mean": dp.mean(), "slope_through_origin": slope,
                         "r": np.corrcoef(da, dp)[0, 1] if len(g) > 2 else np.nan, "share_predicted_older": (dp > 0).mean()})
            curves[(name, model)] = [(lbl, int(((da >= lo) & (da < hi)).sum()), dp[(da >= lo) & (da < hi)].mean(),
                                      dp[(da >= lo) & (da < hi)].std(ddof=1) / np.sqrt(max(1, ((da >= lo) & (da < hi)).sum())),
                                      da[(da >= lo) & (da < hi)].mean())
                                     for lo, hi, lbl in BANDS if ((da >= lo) & (da < hi)).sum() >= 10]
    return pd.DataFrame(rows), curves


def scored_meta(cdir, ev, family):
    """Adults of a test cohort restricted to the scored recordings; Harvard keyed by the cross-site patient id."""
    meta = load_meta(cdir, family, 18.0)
    if family:
        test = {f.split("/")[-1].rsplit("_", 1)[0] for f in json.load(open(os.path.join(cdir, "age_split.json")))["files"]["test"]}
        meta = meta[meta.index.isin(test)]
        if family == "harvard":
            pid = pd.read_parquet(os.path.join(cdir, "cohort.parquet")).set_index("stem")["patient_id"]
            meta = meta.assign(subject=pid.reindex(meta.index).astype(str))
    scored = json.load(open(ev))["P4ens"]["splits"]["test"]["predictions"]
    return meta[meta.index.isin(list(scored))]


def test_cohorts():
    """Per test cohort: people, EEG units per person (sessions where a session has several files) and,
    for people with >= 2 units, the time between the first and the last (age difference, years)."""
    rows, spans = [], {}
    for name, cdir, ev, family, unit in TEST_COHORTS:
        m = scored_meta(cdir, ev, family)
        if "task" in m:                                       # TDBRAIN / Dortmund: one unit = one session
            u = m.groupby(["subject", "session"]).agg(age=("age", "first")).reset_index()
        else:
            u = m.reset_index()[["subject", "age"]]
        per = u.groupby("subject").agg(n=("age", "size"), span=("age", lambda a: a.max() - a.min()))
        rep = per[per.n >= 2]
        row = {"cohort": name, "unit": unit, "files": len(m), "units": len(u), "people": len(per),
               "people_1": int((per.n == 1).sum()), "people_2": int((per.n == 2).sum()), "people_3plus": int((per.n >= 3).sum()),
               "max_units": int(per.n.max()), "span_median": rep.span.median() if len(rep) else np.nan,
               "span_p10": rep.span.quantile(0.1) if len(rep) else np.nan, "span_p90": rep.span.quantile(0.9) if len(rep) else np.nan,
               "span_gt0": int((rep.span > 0).sum())}
        rows.append(row)
        spans[name] = rep.span.to_numpy(float)
    return pd.DataFrame(rows), spans


def test_figure(path, tab, spans):
    """Figure 4 of the report: EEGs per person and time between them in the seven test cohorts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from labram.eval.age_plots import assert_max_columns, panel_labels
    plt.rcParams.update({"font.size": 16, "axes.titlesize": 16})
    fig, axes = plt.subplots(1, 2, figsize=(15, 8.2), gridspec_kw={"width_ratios": [1.3, 1]})
    a, b = axes
    cols = [("people_1", "1", "#cfd9e8"), ("people_2", "2", "#2a78d6"), ("people_3plus", "3 or more", "#0b3c7a")]
    tab = tab.iloc[::-1].reset_index(drop=True)
    def c(v):
        return f"{v:,}" if v == 0 or v >= SMALL_CELL else f"<{SMALL_CELL}"
    ylab = [f"{r.cohort} ({r.unit})\n{r.people:,} people: {c(r.people_1)} / {c(r.people_2)} / {c(r.people_3plus)}"
            for r in tab.itertuples()]
    for k, r in enumerate(tab.itertuples()):
        left = 0.0
        for col, lbl, c in cols:
            v = getattr(r, col)
            share = 100 * v / r.people
            a.barh(k, share, left=left, color=c, edgecolor="white", label=lbl if k == 0 else None)
            if share >= 8:
                a.text(left + share / 2, k, f"{share:.0f} %", ha="center", va="center", fontsize=13.5,
                       color="white" if col != "people_1" else "#1f1f1e")
            left += share
    a.set_yticks(range(len(tab)), ylab, fontsize=14)
    a.set_xlim(0, 100)
    a.set_xlabel("share of people (%)")
    a.set_title("EEGs per person (people: 1 / 2 / 3+)")
    a.legend(title="EEGs per person", frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.07), fontsize=14)
    a.grid(axis="x", color="#e4e3df", lw=0.8)
    a.set_axisbelow(True)
    order = [n for n in tab.cohort if len(spans[n]) >= 5]
    data = [spans[n] for n in order]
    pos = [list(tab.cohort).index(n) for n in order]
    bp = b.boxplot(data, positions=pos, vert=False, whis=(10, 90), showfliers=False, widths=0.55, patch_artist=True,
                   medianprops=dict(color="#1f1f1e", lw=2))
    for patch in bp["boxes"]:
        patch.set(facecolor="#fce4d6", edgecolor="#eb6834", lw=1.4)
    for k, n in enumerate(tab.cohort):
        if n not in order:
            b.text(0.2, k, "one EEG per person", va="center", fontsize=14, color="#6b6a64", style="italic")
        else:
            s = spans[n]
            b.text(b.get_xlim()[1] if False else 15.3, k, f"n = {len(s):,}, median {np.median(s):.1f} y", va="center", ha="right", fontsize=13.5)
    b.set_yticks(range(len(tab)), [""] * len(tab))
    b.set_ylim(a.get_ylim())
    b.set_xlim(0, 15.5)
    b.set_xlabel("first to last EEG of a person (years)")
    b.set_title("time between EEGs (people with ≥ 2)")
    b.grid(axis="x", color="#e4e3df", lw=0.8)
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    panel_labels(list(axes))
    assert_max_columns(fig)
    fig.savefig(path, dpi=150)
    print(path)


def figure(path, spans, counts, curves, grids):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from labram.eval.age_plots import assert_max_columns, panel_labels
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 16})
    col = {"Temple": "#2a78d6", "Harvard": "#eb6834"}
    fig, axes = plt.subplots(3, 2, figsize=(15, 15), gridspec_kw={"height_ratios": [1, 1.2, 1]})
    a = list(axes.flat)
    width = 0.4
    for k, (name, n) in enumerate(counts.items()):
        vals = np.clip(n, 1, 10)
        cnt = np.array([(vals == v).mean() for v in range(1, 11)]) * 100
        a[0].bar(np.arange(1, 11) + (k - 0.5) * width, cnt, width, color=col[name], label=name)
    a[0].set_yscale("log")
    a[0].set_xticks(range(1, 11), [str(v) for v in range(1, 10)] + ["10+"])
    a[0].set_xlabel("recordings per patient")
    a[0].set_ylabel("share of patients (%)")
    a[0].set_title("recordings per patient")
    a[0].legend(frameon=False)
    edges = np.arange(-0.5, 21.5, 1)
    for name, s in spans.items():
        a[1].hist(np.clip(s, 0, 20), bins=edges, density=True, histtype="step", lw=2, color=col[name],
                  label=f"{name} (median {np.median(s):.0f} y, ≥ 5 y: {100 * (s >= 5).mean():.0f} %)")
    a[1].set_xlabel("age span, last minus first (years; 20 = 20+)")
    a[1].set_ylabel("share of patients with ≥ 2")
    a[1].set_title("age span between a patient's recordings")
    a[1].legend(frameon=False, fontsize=13)
    for ax, (name, g) in zip(a[2:4], grids.items()):
        im = ax.imshow(np.log10(np.maximum(g, 1)), cmap="Blues", vmin=0, vmax=np.log10(max(gg.max() for gg in grids.values())),
                       aspect="auto", origin="lower")
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                v = g[i, j]
                txt = "" if v == 0 else ("<5" if v < SMALL_CELL else f"{v:,}")
                ax.text(j, i, txt, ha="center", va="center", fontsize=14,
                        color="white" if v >= 10 ** (0.6 * np.log10(max(gg.max() for gg in grids.values()))) else "#1f1f1e")
        cols, rows_ = g.sum(axis=0), g.sum(axis=1)
        ax.set_xticks(range(len(N_BINS)), [f"{lbl}\n({c:,})" if c >= SMALL_CELL else f"{lbl}\n(<5)" if c else f"{lbl}\n(0)"
                                           for (_, _, lbl), c in zip(N_BINS, cols)])
        ax.set_yticks(range(len(SPAN_BINS)), [f"{lbl} ({r:,})" if r >= SMALL_CELL else f"{lbl} (<5)" if r else f"{lbl} (0)"
                                              for (_, _, lbl), r in zip(SPAN_BINS, rows_)])
        ax.set_xlabel("EEGs per patient (patients)")
        ax.set_ylabel("age span, years (patients)")
        ax.set_title(f"{name}: {g.sum():,} patients with ≥ 2 EEGs")
        ax.grid(False)
    for ax, model in ((a[4], "P4ens"), (a[5], "E2L")):
        for (name, m), pts in curves.items():
            if m != model or not pts:
                continue
            x = [p[4] for p in pts]
            y = [p[2] for p in pts]
            e = [1.96 * p[3] for p in pts]
            ax.errorbar(x, y, yerr=e, marker="o", ms=5, capsize=3, lw=1.6, label=name)
        lim = 14
        ax.plot([0, lim], [0, lim], ls=":", color="#1f1f1e", lw=1.2, label="identity")
        ax.set_xlim(0, lim)
        ax.set_xlabel("age difference, first to last recording (years)")
        ax.set_ylabel("change of predicted age (y)")
        ax.set_title(f"{model}: predicted change against true change")
        ax.legend(frameon=False, fontsize=13)
    for ax in a[:2] + a[4:]:
        ax.grid(color="#e4e3df", lw=0.8)
    for ax in a:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    panel_labels(a)
    assert_max_columns(fig)
    fig.savefig(path, dpi=150)
    print(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--figure", required=True)
    ap.add_argument("--test-figure", default=None, help="EEGs per person in the test cohorts (report Figure 4)")
    args = ap.parse_args()
    summ, spans, counts, grids = training_summary()
    lon, curves = longitudinal()
    os.makedirs(args.out_dir, exist_ok=True)
    summ.to_csv(os.path.join(args.out_dir, "repeat_recordings_summary.csv"), index=False)
    lon.to_csv(os.path.join(args.out_dir, "repeat_recordings_longitudinal.csv"), index=False)
    pd.DataFrame([{"set": k[0], "model": k[1], "band": p[0], "patients": p[1], "delta_pred_mean": p[2], "se": p[3],
                   "delta_age_mean": p[4]} for k, v in curves.items() for p in v]).to_csv(
        os.path.join(args.out_dir, "repeat_recordings_by_band.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(summ.T.round(3).to_string())
        print(lon.round(3).to_string(index=False))
    pd.DataFrame([{"dataset": name, "age_span": sl, "eegs_per_patient": nl,
                   "patients": (int(g[i, j]) if g[i, j] == 0 or g[i, j] >= SMALL_CELL else f"<{SMALL_CELL}")}
                  for name, g in grids.items() for i, (_, _, sl) in enumerate(SPAN_BINS)
                  for j, (_, _, nl) in enumerate(N_BINS)]).to_csv(os.path.join(args.out_dir, "repeat_recordings_grid.csv"), index=False)
    for name, g in grids.items():
        print(name, "patients by EEGs per patient:", dict(zip([b[2] for b in N_BINS], g.sum(axis=0))))
    figure(args.figure, spans, counts, curves, grids)
    if args.test_figure:
        tab, tspans = test_cohorts()
        out = tab.copy()
        for c in ("people_1", "people_2", "people_3plus"):                 # small cells (data use agreements)
            out[c] = [v if v == 0 or v >= SMALL_CELL else f"<{SMALL_CELL}" for v in out[c]]
        out.to_csv(os.path.join(args.out_dir, "repeat_recordings_test_cohorts.csv"), index=False)
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(tab.round(2).to_string(index=False))
        test_figure(args.test_figure, tab, tspans)


if __name__ == "__main__":
    main()
