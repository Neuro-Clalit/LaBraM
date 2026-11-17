#!/usr/bin/env python
"""One table of every test cohort of the brain-age work, with each model's result.

For each cohort (the in-domain Temple / Harvard test split and every external
cohort) and each model (E2L, the device-robust E2L, the P3 and P4 five-fold
ensembles) it recomputes, from the saved predictions, the metrics of the
adults (18+): MAE with a 95 % subject-bootstrap interval, bias, Pearson r and
R^2 about the cohort's own mean age, for all scored units and for the healthy
/ normal-report ones. The scoring unit is the recording, or the session (mean
of its runs) where a cohort has several runs per session (TDBRAIN, Dortmund),
as in ``external_cohort_table.py``.

With ``--bias-by-group`` (the ``--by-group`` CSV of ``bias_correction_cohorts.py``
run on the same cohorts) the page gets a section with every age-bias correction
per cohort and subset (normal / abnormal / no report, or controls / patients).

Outputs: ``--out`` (long CSV, one row per cohort, model and group) and
``--doc`` (a Markdown page with the cohort details of ``COHORTS`` below, one
accuracy table per metric and the verdict on each cohort). The cohort details
are maintained here, next to the paths, so that the page is regenerated in one
step when a model or a cohort is added.

    python scripts/all_cohorts_summary.py --out docs/data/external_eval/all_cohorts_summary.csv \\
        --doc docs/all_cohorts_summary.md
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from external_cohort_table import load_meta, units  # noqa: E402
from tdbrain_brain_age import metrics  # noqa: E402

E = "/data/datasets/EEG-public"
O = "/data/runs/heedb/external_eval"
OWN = "/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1"
MODELS = [("E2L", "E2L"), ("E2LrobF", "device-robust E2L"), ("P3ens", "P3 ensemble"), ("P4ens", "P4 ensemble")]

# name, cohort dir, prediction files, family, and the short details of the page
COHORTS = [
    dict(key="temple", name="Temple (TUH)", dir=OWN, evals=["own_5min_models.json", "own_5min_e2lrobf.json"],
         family="temple", role="in-domain test split", who="hospital patients, routine clinical EEG; normal and abnormal reports",
         chain="clinical systems, 0.1–75 Hz after our pipeline", protocol="first 5 min after a 60 s trim",
         doc="docs/tech_report", use="yes", note="training corpus; held-out patients"),
    dict(key="harvard", name="Harvard (HEEDB)", dir=OWN, evals=["own_5min_models.json", "own_5min_e2lrobf.json"],
         family="harvard", role="in-domain test split", who="hospital patients, routine clinical EEG; normal, abnormal and unlabelled",
         chain="clinical systems, 0.1–75 Hz after our pipeline", protocol="first 5 min after a 60 s trim",
         doc="docs/tech_report", use="yes", note="training corpus; held-out patients"),
    dict(key="tdbrain", name="TDBRAIN", dir=f"{E}/TDBRAIN/cohorts/tdbrain_v1", evals=["tdbrain_all_models.json"],
         role="external, zero-shot", who="psychiatric outpatients (depression, ADHD, OCD, memory complaints) and healthy controls",
         chain="26 channels, 500 Hz, linked mastoids", protocol="2 min eyes closed + 2 min eyes open per session, every window",
         doc="docs/tdbrain_onboarding.md", use="yes", note="data use agreement: aggregates only"),
    dict(key="lemon", name="LEMON", dir=f"{E}/LEMON/cohorts/lemon_v1", evals=["lemon_all_models.json"],
         role="external, zero-shot", who="healthy volunteers, a young and an old group",
         chain="62 channels, 2500 Hz, FCz reference", protocol="16 min alternating eyes closed / open, every window",
         doc="docs/external_eval_results.md", use="yes", note="ages public only as 5-year bins"),
    dict(key="dortmund", name="Dortmund Vital Study", dir=f"{E}/DORTMUND/cohorts/dortmund_v1",
         evals=["dortmund_all_models.json"], role="external, zero-shot", who="healthy adults, even over 20–70; 208 with a 5-year follow-up",
         chain="64 channels, 1000 Hz, FCz reference", protocol="four 3-min runs per session (eyes closed / open, before / after tasks), every window",
         doc="docs/dortmund_onboarding.md", use="yes", note="post-task runs look 1.5–1.7 y older"),
    dict(key="srm", name="SRM", dir=f"{E}/SRM/cohorts/srm_v1", evals=["srm_all_models.json"],
         role="external, zero-shot", who="healthy adults",
         chain="64 channels, 1024 Hz, average reference", protocol="4 min eyes closed, every window",
         doc="docs/open_cohorts_onboarding.md", use="yes", note="small"),
    dict(key="chbp", name="CHBP", dir=f"{E}/CHBP/cohorts/chbp_v1", evals=["chbp_all_models.json"],
         role="external, zero-shot", who="healthy volunteers, mostly under 45",
         chain="recorded 0.5–50 Hz at about 0.6× the training amplitude; two caps", protocol="resting part, every window",
         doc="docs/chbp_domain_shift.md", use="only with the device-robust model", note="recording-chain shift: band and gain"),
    dict(key="nmt", name="NMT", dir=f"{E}/NMT/cohorts/nmt_v1", evals=["nmt_all_models.json"],
         role="external, zero-shot", who="hospital patients in Pakistan, routine clinical EEG; normal and abnormal reports",
         chain="released band-limited to about 1–35 Hz, amplitude scale undocumented", protocol="about 12 min, every window after a 60 s trim",
         doc="docs/open_cohorts_onboarding.md", use="no", note="band limit breaks the raw-input models; normal / abnormal contrast only"),
    dict(key="ahepa", name="AHEPA", dir=f"{E}/AHEPA/cohorts/ahepa_v1", evals=["ahepa_all_models.json"],
         role="external, zero-shot", who="Alzheimer's disease, frontotemporal dementia and elderly controls",
         chain="clinical system, 19 channels, 500 Hz", protocol="eyes closed, 5–21 min, every window after a 60 s trim",
         doc="docs/open_cohorts_onboarding.md", use="no", note="controls predicted about 10 y too young, cause unknown; patient / control contrast only"),
    dict(key="iowapd", name="Iowa Parkinson's", dir=f"{E}/IOWAPD/cohorts/iowapd_v1", evals=["iowapd_all_models.json"],
         role="external, zero-shot", who="Parkinson's disease and elderly controls",
         chain="63–66 channels, 500 Hz, Pz reference", protocol="eyes open, 2–6 min, every window",
         doc="docs/open_cohorts_onboarding.md", use="no", note="controls predicted about 11 y too young, cause unknown; patient / control contrast only"),
    dict(key="telavivpd", name="Tel Aviv Parkinson's", dir=f"{E}/TELAVIVPD/cohorts/telavivpd_v1",
         evals=["telavivpd_all_models.json"], role="external, zero-shot", who="Parkinson's disease and elderly controls",
         chain="EGI net, 250 Hz, vertex reference", protocol="eyes open, 4 min, every window",
         doc="docs/open_cohorts_onboarding.md", use="no", note="controls predicted about 7 y too young, cause unknown; patient / control contrast only"),
    dict(key="mipdb", name="MIPDB adults", dir=f"{E}/MIPDB/cohorts/mipdb_v1", evals=["mipdb_all_models.json"],
         role="external, zero-shot", who="young adults of a developmental cohort",
         chain="EGI net, 500 Hz, vertex reference", protocol="about 6 min, every window",
         doc="docs/external_eval_results.md", use="no", note="31 recordings: too few; left out of final reports"),
]


def cohort_rows(c):
    meta = load_meta(c["dir"], c.get("family"), 18.0)
    res = {}
    for f in c["evals"]:
        res.update(json.load(open(os.path.join(O, f))))
    rows = []
    for key, _ in MODELS:
        if key not in res:
            continue
        d = units(meta, pd.Series(res[key]["splits"]["test"]["predictions"], dtype=float))
        groups = [("all", d)]
        if d["pathological"].any() or (~(d["healthy"] | d["pathological"])).any():
            groups.append(("healthy", d[d["healthy"]]))
        for group, g in groups:
            if len(g) >= 10:
                rows.append({"cohort": c["key"], "model": key, "group": group, **metrics(g),
                             "slope": np.polyfit(g["age"], g["pred"], 1)[0]})
    return rows


def signed(x):
    """One signed decimal with a real minus sign and no negative zero."""
    return f"{round(float(x), 1) + 0.0:+.1f}".replace("-", "−")


def cell(t, cohort, model, group, what):
    r = t[(t.cohort == cohort) & (t.model == model) & (t.group == group)]
    if r.empty:
        return ""
    r = r.iloc[0]
    if what == "mae":
        return f"{r.mae:.2f} [{r.mae_lo:.2f}, {r.mae_hi:.2f}]"
    if what == "r2":
        return f"{r.r2:.2f} / {r.r:.2f}".replace("-", "−")
    return signed(r.bias)


REPORT_COHORTS = {"temple", "harvard", "nmt"}                 # subsets are EEG-report verdicts
SUBSET = {True: {"all": "all", "normal": "normal report", "abnormal": "abnormal report", "unlabelled": "no report"},
          False: {"all": "all", "normal": "healthy controls", "abnormal": "patients", "unlabelled": "unlabelled"}}
METHODS = [("none", "none"), ("cole", "Cole inversion"), ("recalibrate", "recalibration"),
           ("age_level", "linear gap (uses age)"), ("quadratic_gap", "quadratic gap (uses age)")]


def bias_section(path):
    """Markdown lines of the correction-by-subset section."""
    g = pd.read_csv(path)
    out = ["## 4. Age-bias correction per cohort and subset", "",
           "The corrections of `docs/tdbrain_onboarding.md` Section 19 (`scripts/bias_correction_cohorts.py --by-group`), fitted",
           "once per model on validation recordings of the training cohort with a normal EEG report and applied unchanged to",
           "every cohort and subset; nothing is fitted on a scored cohort. With prediction $\\hat y$ and age $y$, and the",
           "validation fits $\\hat y \\approx a\\,y + b$, $y \\approx c\\,\\hat y + d$ and $\\hat y - y \\approx c_0 + c_1 y + c_2 y^2$:", "",
           "$$ \\hat y^{cole} = \\frac{\\hat y - b}{a}, \\qquad \\hat y^{recal} = c\\,\\hat y + d, \\qquad "
           "\\hat y^{lin} = \\hat y - \\big((a-1)\\,y + b\\big), \\qquad \\hat y^{quad} = \\hat y - \\big(c_0 + c_1 y + c_2 y^2\\big) \\tag{4.1} $$", "",
           "Cole inversion and recalibration need no true age and give a new age estimate. The two gap corrections use the",
           "true age: they give a corrected brain-age gap for group comparisons, and their \"MAE\" is the mean absolute",
           "corrected gap, not an accuracy. Each cell is MAE in years / slope of the (corrected) prediction on age inside",
           "the subset (1 = no compression toward the mean age). Subsets with fewer than 20 units are left out; a slope",
           "inside a subset with a narrow age range (the elderly cohorts, CHBP) is unstable.", ""]
    for num, (model, name) in enumerate((("P4ens", "P4 ensemble"), ("P3ens", "P3 ensemble")), 1):
        out += [f"### Table 4.{num}. {name}: MAE / slope under each correction, by cohort and subset", "",
                "| cohort | subset | n | " + " | ".join(m for _, m in METHODS) + " |", "|---|---|---|" + "---|" * len(METHODS)]
        for c in COHORTS:
            x = g[(g.cohort == c["key"]) & (g.model == model)]
            for group in ("all", "normal", "abnormal", "unlabelled"):
                r = x[x.group == group].set_index("method")
                if r.empty:
                    continue
                cells = [f"{r.loc[k, 'mae']:.2f} / {r.loc[k, 'slope']:.2f}".replace("-", "−") for k, _ in METHODS]
                label = SUBSET[c["key"] in REPORT_COHORTS][group]
                out.append(f"| {c['name'] if group == 'all' else ''} | {label} | {int(r.loc['none', 'n']):,} | "
                           + " | ".join(cells) + " |")
        out.append("")
    p4 = g[(g.model == "P4ens") & (g.group == "all")].pivot(index="cohort", columns="method", values=["mae", "slope"])
    ok = [c["key"] for c in COHORTS if c["use"] == "yes"]
    cost = (p4["mae"]["cole"] - p4["mae"]["none"]).loc[ok]
    sl0, sl1 = p4["slope"]["none"].loc[ok], p4["slope"]["quadratic_gap"].loc[ok]
    fail = p4["mae"]["quadratic_gap"].loc[[c["key"] for c in COHORTS if c["use"] != "yes"]]

    def q(cohort, group):
        r = g[(g.model == "P4ens") & (g.cohort == cohort) & (g.group == group) & (g.method == "quadratic_gap")]
        return float(r["mae"].iloc[0])

    out += ["**What the tables show:**",
            f"- **On the cohorts where the models work** ({', '.join(c['name'] for c in COHORTS if c['use'] == 'yes')}) "
            f"the raw slope of the P4 ensemble is {sl0.min():.2f} to {sl0.max():.2f} and the quadratic gap correction "
            f"brings it to {sl1.min():.2f} to {sl1.max():.2f}: a correction fitted on validation data transfers.",
            f"- **Removing the bias without the true age costs accuracy:** Cole inversion raises the MAE by "
            f"{cost.min():.1f} to {cost.max():.1f} years on these cohorts, and recalibration changes it by less than 0.1 years.",
            "- **An abnormal report stays harder after correction.** The correction is one function of age, so it removes "
            "what came from the subsets' different ages and nothing else. With the quadratic gap correction the mean "
            f"absolute gap is {q('temple', 'abnormal'):.2f} years for abnormal against {q('temple', 'normal'):.2f} for normal "
            f"reports in Temple, {q('harvard', 'abnormal'):.2f} against {q('harvard', 'normal'):.2f} in Harvard, and "
            f"{q('harvard', 'unlabelled'):.2f} for Harvard recordings without a report. In TDBRAIN patients and controls "
            f"do not differ ({q('tdbrain', 'abnormal'):.2f} against {q('tdbrain', 'normal'):.2f}).",
            "- **Where the models fail, no correction repairs them.** After the quadratic gap correction the mean absolute "
            f"gap is still {fail.min():.1f} to {fail.max():.1f} years on CHBP, NMT, the elderly disease cohorts and the MIPDB "
            "adults. The slope can come close to 1 there (Tel Aviv) without the error falling: the fault is a "
            "cohort-wide offset and a lost ranking, not the trend with age that the correction was fitted for.", ""]
    return out


def write_doc(t, path, bias_csv=None):
    best = {c["key"]: t[(t.cohort == c["key"]) & (t.group == "all")].sort_values("mae").iloc[0] for c in COHORTS}
    out = ["# All test cohorts and all models in one place", "",
           "Generated by `scripts/all_cohorts_summary.py` from the saved predictions; regenerate it instead of editing it.",
           "Adults (18 and over) only. The models never saw the external cohorts (zero-shot): nothing in them was used to",
           "fit, tune, select or calibrate a model. Numbers for sub-groups, rejection by confidence and bias correction are",
           "in the document named for each cohort.", "",
           "## 1. The cohorts", "",
           "### Table 1.1. Cohorts, with the scored adults", "",
           "| cohort | role | who | scored units (people) | age, mean ± SD | recording chain | scored signal | details |",
           "|---|---|---|---|---|---|---|---|"]
    for c in COHORTS:
        r = best[c["key"]]
        out.append(f"| {c['name']} | {c['role']} | {c['who']} | {int(r.n):,} ({int(r.n_subjects):,}) | "
                   f"{r.age_mean:.1f} ± {r.age_sd:.1f} | {c['chain']} | {c['protocol']} | `{c['doc']}` |")
    out += ["", "A unit is a recording, or a session (the mean of its runs) for TDBRAIN and Dortmund. HBN (a paediatric study) is",
            "onboarded but kept out of every report.", "",
            "## 2. Accuracy per model", "",
            "With prediction $\\hat y$ and age $y$ over the $N$ units of a cohort, $\\bar y$ its mean age:", "",
            "$$ \\mathrm{MAE} = \\frac{1}{N}\\sum \\lvert \\hat y - y \\rvert, \\qquad \\mathrm{bias} = \\frac{1}{N}\\sum (\\hat y - y), "
            "\\qquad R^2 = 1 - \\frac{\\sum (\\hat y - y)^2}{\\sum (y - \\bar y)^2} \\tag{2.1} $$", "",
            "Intervals are 95 % percentile intervals over 2,000 bootstrap resamples of people.", ""]
    heads = " | ".join(name for _, name in MODELS)
    for num, what, title in (("2.1", "mae", "MAE in years with a 95 % interval, all scored adults"),
                             ("2.2", "r2", "$R^2$ / Pearson r, all scored adults"),
                             ("2.3", "bias", "Bias in years (prediction − age), all scored adults")):
        out += [f"### Table {num}. {title}", "", f"| cohort | {heads} |", "|---|" + "---|" * len(MODELS)]
        for c in COHORTS:
            cells = [cell(t, c["key"], key, "all", what) for key, _ in MODELS]
            if what == "mae":
                k = int(np.argmin([float(x.split()[0]) if x else np.inf for x in cells]))
                cells[k] = f"**{cells[k]}**"
            out.append(f"| {c['name']} | " + " | ".join(cells) + " |")
        out.append("")
    out += ["### Table 2.4. Healthy or normal-report adults only: MAE / bias in years", "",
            "Cohorts that also hold patients or abnormal reports. Healthy = a normal EEG report (Temple, Harvard, NMT) or the",
            "cohort's healthy controls.", "", f"| cohort | n | {heads} |", "|---|---|" + "---|" * len(MODELS)]
    for c in COHORTS:
        h = t[(t.cohort == c["key"]) & (t.group == "healthy")]
        if h.empty:
            continue
        cells = []
        for key, _ in MODELS:
            r = h[h.model == key]
            cells.append("" if r.empty else f"{r.iloc[0].mae:.2f} / {signed(r.iloc[0].bias)}")
        out.append(f"| {c['name']} | {int(h.iloc[0].n):,} | " + " | ".join(cells) + " |")
    out += ["", "## 3. What each cohort is good for", "",
            "### Table 3.1. Verdict per cohort", "",
            "| cohort | usable as an age benchmark | best model, MAE (y) | note |", "|---|---|---|---|"]
    names = dict(MODELS)
    for c in COHORTS:
        r = best[c["key"]]
        out.append(f"| {c['name']} | {c['use']} | {names[r.model]}, {r.mae:.2f} | {c['note']} |")
    ok = [c["key"] for c in COHORTS if c["use"] == "yes" and c["role"].startswith("external")]
    p4 = t[(t.group == "all") & (t.model == "P4ens") & t.cohort.isin(ok)]
    rob = t[(t.group == "all") & (t.model == "E2LrobF")].set_index("cohort")
    e2l = t[(t.group == "all") & (t.model == "E2L")].set_index("cohort")
    wins = [c["name"] for c in COHORTS if rob.loc[c["key"], "mae"] < e2l.loc[c["key"], "mae"]]
    cost = (rob["mae"] - e2l["mae"])[lambda d: d > 0]
    out += ["", "**What the tables show:**",
            f"- **On the {len(ok)} external cohorts that resemble the training data** "
            f"({', '.join(c['name'] for c in COHORTS if c['key'] in ok)}) the P4 ensemble reaches "
            f"{p4.mae.min():.2f} to {p4.mae.max():.2f} years zero-shot, with $R^2$ {p4.r2.min():.2f} to {p4.r2.max():.2f}: "
            "at or below the error on the held-out Temple and Harvard patients.",
            "- **The ensembles beat the window model E2L on every cohort where the models work;** P4 and P3 are within "
            "0.3 years of each other there.",
            f"- **The device-robust model is better than E2L only on the cohorts where the raw-input models fail** ({', '.join(wins)}), "
            f"and worse everywhere else, by {cost.min():.1f} to {cost.max():.1f} years.",
            "- **Five cohorts are not age benchmarks for the present models:** NMT (released band-limited), the three "
            "elderly disease cohorts (their controls are predicted 7 to 11 years too young) and the MIPDB adults (31 "
            "recordings). They serve for contrasts inside a cohort, or not at all.", ""]
    if bias_csv:
        out += bias_section(bias_csv)
    open(path, "w").write("\n".join(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--doc", default="")
    ap.add_argument("--bias-by-group", default=None, help="bias_correction_cohorts.py --by-group CSV of the same cohorts")
    args = ap.parse_args()
    t = pd.DataFrame([r for c in COHORTS for r in cohort_rows(c)])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t.to_csv(args.out, index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(t[["cohort", "model", "group", "n", "n_subjects", "age_mean", "mae", "mae_lo", "mae_hi", "bias", "r", "r2",
                 "slope"]].round(2).to_string(index=False))
    if args.doc:
        write_doc(t, args.doc, args.bias_by_group)
        print(args.doc)


if __name__ == "__main__":
    main()
