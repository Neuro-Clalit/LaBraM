"""Build notebooks/clinical_records_analytics.ipynb (cells only; execute it afterwards):

    python scripts/make_clinical_analytics_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/clinical_records_analytics.ipynb
    python scripts/notebook_to_pdf.py notebooks/clinical_records_analytics.ipynb
"""
import os

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s, tags=None: cells.append(nbf.v4.new_code_cell(s.strip("\n"), metadata={"tags": tags} if tags else {}))
FINDINGS = os.path.join(os.path.dirname(__file__), "clinical_analytics_findings.md")

md(r"""
# Health records and EEG reports: TUAB, the TUH training cohort and HEEDB

📄 **[PDF version](clinical_records_analytics.pdf)**

What the three corpora we train on say about their patients, beyond the EEG: **age and sex**, the
**normal / abnormal** verdict of the EEG report, the patients' **pathologies** (seizure disorders, stroke,
dementia, psychiatric diagnoses, headache and migraine, sleep disorders, tumours and others) and the **types of
abnormality** the reports name (slowing, epileptiform discharges, seizures, suppression, ...). Every statistic is
computed on one harmonised per-recording table (`labram/eval/clinical_analytics.py`), so the corpora read side
by side.

* **TUAB** (v3.0.1, 2,993 recordings): the normal / abnormal benchmark; its test split is our TUAB eval set.
* **TUH cohort** (`all_both_v3_anymiss`, 31,312 recordings): the TUH part of E2 / E2L / E2P.
* **HEEDB** (`heedb_routine_v1`, 71,805 adult routine recordings): the Harvard part of the same runs.

HEEDB is under a data use agreement: every HEEDB cell computed from fewer than 10 recordings is blank, and no
row-level data is shown.
""")

code(r"""
import os, sys
sys.path.insert(0, os.path.abspath(".."))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from labram.eval import clinical_analytics as ca
from labram.eval.age_plots import assert_max_columns, numbered_table, panel_labels

pd.set_option("display.max_columns", 40)
df = ca.load_all()
CORPORA = ["TUAB", "TUH cohort", "HEEDB"]
COLORS = {"TUAB": "#7a7a7a", "TUH cohort": "#e07b39", "HEEDB": "#3a6ea5"}
df["age_group"] = ca.age_group(df["age"])
PX = ca.pathology_names(df)
AB = ca.abnormal_type_names(df)


def finish(fig, axes=None):
    fig.tight_layout()
    if axes is not None and len(axes) > 1:
        panel_labels(list(axes))
    assert_max_columns(fig)
    plt.show()


def heat(ax, table, title, fmt="{:.0f}", cmap="viridis", vmax=None):
    data = table.to_numpy(dtype=float)
    im = ax.imshow(np.ma.masked_invalid(data), aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(table.shape[0]), table.index, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if data[i, j] < (vmax or np.nanmax(data)) * 0.6 else "black")
    ax.set_title(title)
    return im
""")

# ------------------------------------------------------------------ 1
md(r"""
## 1. Sources and definitions

### Table 1.1. Where each kind of information comes from

| | TUAB / TUH cohort | HEEDB |
|---|---|---|
| Age, sex | EDF header (age), cohort metadata | BDSP metadata (`AgeAtVisit`, sex) |
| Normal / abnormal | TUAB folder label; else the NEDC report finding (TUH cohort) | the EEG report (BDSP findings), unlabelled when the session has no report |
| Pathologies | NEDC TUEG clinical sheet: 15 Y/N diagnosis columns per session, plus keyword mentions in its free-text "Other diagnosis" | per-patient ICD-10 codes (neurology extract): TUH-aligned `dx_*` flags and ICD-10 groups `icd_*` |
| Abnormality types | the sheet's "abnormal due to" text, mapped to reason categories (`meta_analysis.REPORT_REASONS`) | report finding groups (`fg_*`) |
| Missing information | session not in the sheet → unknown | patient without any ICD-10 record → unknown |

Three caveats shape every comparison below:

* **HEEDB pathologies are per patient and undated.** An ICD-10 code may predate or postdate the EEG; the TUH
  sheet describes the session. HEEDB rates are therefore "ever diagnosed", TUH rates "noted at this EEG".
* **TUH free-text categories are mentions.** Headache / migraine, sleep disorder, tumour, syncope and the other
  keyword categories (Table 1.2, marked *mention*) count a session when its "Other diagnosis" text names the
  condition; a session that does not name it is counted as not mentioned, which is not evidence of absence.
  These rates are **lower bounds**.
* **The NEDC sheet leaves its diagnosis columns almost empty for TUAB sessions** (seizure 0.1 %, epilepsy 0.04 %
  against 8.4 % / 3.4 % for the other TUEG sessions of the cohort), although 50 % of them have "Other diagnosis"
  text and 35 % an "abnormal due to" reason. TUAB's near-zero diagnosis rates mean *not annotated*, not healthy.

A corpus that does not code a category at all (e.g. HEEDB's neurology extract has no psychiatric or injury
codes besides depression and ADHD) is left blank, never 0.
""")

md(r"""
### Table 1.2. Pathology categories and their source columns

`tuh` / `heedb` list the columns OR-ed into the category (`kw_*` = keyword mention in the TUH free text,
`dx_*` / `icd_*` = coded diagnosis); `–` = not coded in that corpus.
""")
code(r"""
rows = []
for name, (tuh, heedb) in ca.PATHOLOGIES.items():
    rows.append({"category": name, "tuh": ", ".join(tuh) or "–", "heedb": ", ".join(heedb) or "–",
                 "TUH kind": "mention (lower bound)" if name in ca.KEYWORD_PATHOLOGIES else "Y/N column"})
display(numbered_table(pd.DataFrame(rows).set_index("category"), "1.2", "Pathology categories"))
""")

md(r"""
### Table 1.3. Recordings with the pathology known, and positives

Per corpus and category: `known` = recordings whose status for the category is known (TUH: the session is in the
sheet; HEEDB: the patient has an ICD-10 record), `positive` = those with the category. HEEDB counts under 10
are blank.
""")
code(r"""
kc = ca.known_counts(df)
t = kc.pivot(index="pathology", columns="corpus", values=["known", "positive"]).reindex(PX)
t = t.reindex(columns=pd.MultiIndex.from_product([["known", "positive"], CORPORA]))
pos = t["positive"]["HEEDB"]
t[("positive", "HEEDB")] = pos.where((pos >= ca.SMALL_CELL) | (pos == 0))
display(numbered_table(t, "1.3", "Known and positive recordings per category", precision=0))
""")

# ------------------------------------------------------------------ 2
md(r"""
## 2. Who the recordings come from: age, sex and the EEG verdict

### Table 2.1. Corpus overview

Recordings, distinct patients, age (mean, SD, years), the share of women, the EEG report's verdict
(normal / abnormal / unknown, % of recordings) and the share with health-record information (Table 1.1).
""")
code(r"""
display(numbered_table(ca.overview(df).reindex(CORPORA), "2.1", "Corpus overview", precision=1))
""")

md(r"""
### Figure 2.1. Age distribution by EEG verdict

Share of recordings per 5-year age bin (each curve sums to 100 %), for normal (solid) and abnormal (dashed)
reports; adults 18+. (a) TUAB and the TUH cohort; (b) HEEDB.
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
bins = np.arange(18, 96, 5)
for ax, corpora in zip(axes, (["TUAB", "TUH cohort"], ["HEEDB"])):
    for c in corpora:
        for status, ls in (("normal", "-"), ("abnormal", "--")):
            a = df[(df.corpus == c) & (df.status == status)]["age"].dropna()
            h, _ = np.histogram(a, bins=bins)
            ax.plot(bins[:-1] + 2.5, 100 * h / max(h.sum(), 1), ls, color=COLORS[c], label=f"{c} {status} (n={len(a):,})")
    ax.set_xlabel("age (years)"); ax.set_ylabel("% of recordings"); ax.legend(fontsize=8); ax.grid(alpha=.3)
axes[0].set_title("TUAB and TUH cohort"); axes[1].set_title("HEEDB")
finish(fig, axes)
""")

for k, corpus in enumerate(["TUAB", "TUH cohort", "HEEDB"], start=2):
    md(rf"""
### Table 2.{k}. {corpus}: recordings by age group, sex and EEG verdict

Counts per age group (rows) and sex × verdict (columns); recordings with unknown sex are left out.
{"Cells under 10 recordings are blank (DUA)." if corpus == "HEEDB" else ""}
""")
    code(rf"""
t = ca.age_sex_status(df, "{corpus}")
{"t = t.where(t >= ca.SMALL_CELL)" if corpus == "HEEDB" else ""}
display(numbered_table(t, "2.{k}", "{corpus}: age x sex x verdict", precision=0))
""")

md(r"""
### Figure 2.2. Share of abnormal EEGs by age group and sex

Abnormal % among recordings with a known verdict, Eq. (2.1), per age group; women solid, men dashed.
(a) TUAB and the TUH cohort; (b) HEEDB. A point needs at least 10 recordings.

$$
a_{g} = 100\,\frac{\#\{i \in g : \text{report}_i = \text{abnormal}\}}{\#\{i \in g : \text{report}_i \in \{\text{normal}, \text{abnormal}\}\}}
\tag{2.1}
$$

where $g$ is a set of recordings (an age group of one sex and corpus).
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
for ax, corpora in zip(axes, (["TUAB", "TUH cohort"], ["HEEDB"])):
    for c in corpora:
        for sex, ls in (("F", "-"), ("M", "--")):
            g = df[(df.corpus == c) & (df.sex == sex) & df.status.isin(["normal", "abnormal"])]
            r = g.groupby("age_group", observed=True)["status"].agg(lambda s: 100 * s.eq("abnormal").mean() if len(s) >= ca.SMALL_CELL else np.nan)
            ax.plot(r.index.astype(str), r.values, ls, marker="o", color=COLORS[c], label=f"{c} {sex}")
    ax.set_xlabel("age group"); ax.set_ylabel("abnormal % (Eq. 2.1)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
axes[0].set_title("TUAB and TUH cohort"); axes[1].set_title("HEEDB")
finish(fig, axes)
""")

# ------------------------------------------------------------------ 3
md(r"""
## 3. Pathologies

For a corpus and category $p$, the prevalence counts only the recordings whose status for $p$ is known
(Table 1.3):

$$
\pi_{p} = 100\,\frac{\#\{i : x_{ip} = 1\}}{\#\{i : x_{ip} \in \{0, 1\}\}}
\tag{3.1}
$$

where $x_{ip} = 1$ when recording $i$ has category $p$, $0$ when it is known not to (or, for a mention category,
not mentioned), and unknown otherwise (`clinical_analytics.prevalence`).

### Table 3.1. Prevalence per corpus (%)

Eq. (3.1) per category and corpus; *mention* categories (Table 1.2) are lower bounds for TUH; blank = not coded
(or a HEEDB count under 10).
""")
code(r"""
prev = ca.prevalence(df).T[CORPORA]
display(numbered_table(prev, "3.1", "Prevalence per corpus (%)", precision=1))
""")

md(r"""
### Figure 3.1. Prevalence by age group

Eq. (3.1) per category (rows) and age group (columns), %. (a) TUH cohort; (b) HEEDB. Categories not coded in
the corpus are left out; HEEDB cells under 10 positives are blank.
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, c in zip(axes, ["TUH cohort", "HEEDB"]):
    p = ca.prevalence(df[df.corpus == c], by="age_group").droplevel(0)
    p = p.loc[:, p.notna().any()].T
    heat(ax, p, c, fmt="{:.0f}", vmax=np.nanpercentile(p.to_numpy(dtype=float), 95))
finish(fig, axes)
""")

md(r"""
### Table 3.2. Prevalence by sex (%)

Eq. (3.1) per corpus and sex.
""")
code(r"""
s = ca.prevalence(df[df.sex.isin(["F", "M"])], by="sex")
s = s.T.reindex(columns=pd.MultiIndex.from_product([CORPORA, ["F", "M"]]))
s.columns = [f"{a} {b}" for a, b in s.columns]
display(numbered_table(s, "3.2", "Prevalence by sex (%)", precision=1))
""")

md(r"""
### Table 3.3. How often the EEG is abnormal with and without each pathology

Among recordings with a known verdict, the abnormal share, Eq. (2.1), of those with the category (`with`) and
of those known not to have it (`without`); categories with at least 10 recordings.

$$
\Delta a_p = a_{\{i:\,x_{ip}=1\}} - a_{\{i:\,x_{ip}=0\}}
\tag{3.2}
$$

is the difference in percentage points (`clinical_analytics.abnormal_rate_by_pathology`).
""")
code(r"""
ar = ca.abnormal_rate_by_pathology(df)
ar["Δ (pp)"] = ar["abnormal % with"] - ar["abnormal % without"]
t = ar.pivot(index="pathology", columns="corpus", values="Δ (pp)").reindex(columns=CORPORA)
t2 = ar.pivot(index="pathology", columns="corpus", values="abnormal % with").reindex(columns=CORPORA)
out = pd.concat({"abnormal % with": t2, "Δ (pp), Eq. 3.2": t}, axis=1).reindex(PX).dropna(how="all")
display(numbered_table(out, "3.3", "Abnormal EEG share with / without each pathology", precision=1))
""")

md(r"""
### Figure 3.2. Abnormal-EEG excess by pathology

$\Delta a_p$ (Eq. (3.2), percentage points) per category: (a) TUH cohort; (b) HEEDB. Positive = the EEG is more
often abnormal in patients with the category than in those without it.
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
for ax, c in zip(axes, ["TUH cohort", "HEEDB"]):
    d = ar[ar.corpus == c].set_index("pathology")["Δ (pp)"].sort_values()
    ax.barh(d.index, d.values, color=[COLORS[c] if v >= 0 else "#999" for v in d.values])
    ax.axvline(0, color="k", lw=.8); ax.set_xlabel("Δ abnormal % (pp)"); ax.set_title(c); ax.grid(alpha=.3, axis="x")
    ax.tick_params(axis="y", labelsize=8)
finish(fig, axes)
""")

md(r"""
### Figure 3.3. Headache / migraine: prevalence by age and sex

Eq. (3.1) for the headache / migraine category per age group, women solid, men dashed: (a) TUAB and the TUH
cohort (mentions in the "Other diagnosis" text, lower bounds); (b) HEEDB (ICD-10 headache disorders, G43/G44).
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, corpora in zip(axes, (["TUAB", "TUH cohort"], ["HEEDB"])):
    for c in corpora:
        for sex, ls in (("F", "-"), ("M", "--")):
            g = df[(df.corpus == c) & (df.sex == sex)]
            p = ca.prevalence(g, by="age_group")["headache_migraine"].droplevel(0)
            ax.plot(p.index.astype(str), p.values, ls, marker="o", color=COLORS[c], label=f"{c} {sex}")
    ax.set_xlabel("age group"); ax.set_ylabel("headache / migraine % (Eq. 3.1)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
axes[0].set_title("TUAB and TUH cohort (mentions)"); axes[1].set_title("HEEDB (ICD-10)")
finish(fig, axes)
""")

md(r"""
### Table 3.4. HEEDB: co-occurrence of pathologies

For each row category, the share (%) of its patients' recordings that also carry the column category:

$$
c_{pq} = 100\,\frac{\#\{i : x_{ip} = 1,\ x_{iq} = 1\}}{\#\{i : x_{ip} = 1,\ x_{iq} \in \{0,1\}\}}
\tag{3.3}
$$

(`clinical_analytics.co_occurrence`); the diagonal is 100. Cells under 10 recordings are blank.
""")
code(r"""
names = [p for p in PX if df.loc[df.corpus == "HEEDB", f"px_{p}"].notna().any()]
co = ca.co_occurrence(df, "HEEDB", names)
display(numbered_table(co, "3.4", "HEEDB co-occurrence (%)", precision=0))
""")

md(r"""
### Table 3.5. TUH cohort: co-occurrence of pathologies

Eq. (3.3) for the TUH cohort, categories with at least 50 positives.
""")
code(r"""
kc_t = kc[(kc.corpus == "TUH cohort") & (kc.positive >= 50)].pathology.tolist()
display(numbered_table(ca.co_occurrence(df, "TUH cohort", kc_t), "3.5", "TUH cohort co-occurrence (%)", precision=0))
""")

# ------------------------------------------------------------------ 4
md(r"""
## 4. What makes an EEG abnormal: abnormality types

Among abnormal recordings whose report gives a reason (TUH: an "abnormal due to" text) or findings (HEEDB),
the share naming abnormality type $t$:

$$
s_t = 100\,\frac{\#\{i \in A : y_{it} = 1\}}{\#A}
\tag{4.1}
$$

where $A$ is that set of abnormal recordings and $y_{it}$ marks type $t$ in recording $i$'s report
(`clinical_analytics.abnormal_types`). A report can name several types, so the shares do not sum to 100. TUH
types come from keyword rules on free text, HEEDB types from BDSP's structured findings; a type HEEDB does not
code (posterior rhythm, excess beta, asymmetry) is blank.

### Table 4.1. Abnormality types per corpus (%)

`n` = abnormal recordings with a reason / findings (the denominator of Eq. (4.1)).
""")
code(r"""
at = ca.abnormal_types(df).reindex(CORPORA)
display(numbered_table(at, "4.1", "Abnormality types (%)", precision=1))
""")

md(r"""
### Figure 4.1. Abnormality types by age group

Eq. (4.1) per type (rows) and age group (columns), %: (a) TUH cohort; (b) HEEDB.
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
for ax, c in zip(axes, ["TUH cohort", "HEEDB"]):
    t = ca.abnormal_types(df[df.corpus == c], by="age_group").droplevel(0).drop(columns="n")
    t = t.loc[:, t.notna().any()].T
    heat(ax, t, c, vmax=100)
finish(fig, axes)
""")

md(r"""
### Tables 4.2–4.3. Which abnormality types go with which pathology

The lift of type $t$ among abnormal recordings with pathology $p$ over all abnormal recordings of the corpus:

$$
L_{pt} = \frac{s_t(A \cap \{i : x_{ip} = 1\})}{s_t(A)}
\tag{4.2}
$$

with $s_t$ of Eq. (4.1) (`clinical_analytics.type_by_pathology_lift`). $L > 1$: the type is over-represented
in that pathology (e.g. epileptiform discharges in epilepsy). Pathologies with at least 10 such recordings;
HEEDB cells under 10 positives are blank.
""")
md(r"""
### Table 4.2. TUH cohort: lift of abnormality types by pathology
""")
code(r"""
display(numbered_table(ca.type_by_pathology_lift(df, "TUH cohort"), "4.2", "TUH cohort lift (Eq. 4.2)", precision=2))
""")
md(r"""
### Table 4.3. HEEDB: lift of abnormality types by pathology
""")
code(r"""
lh = ca.type_by_pathology_lift(df, "HEEDB")
display(numbered_table(lh.loc[:, lh.notna().any()], "4.3", "HEEDB lift (Eq. 4.2)", precision=2))
""")

# ------------------------------------------------------------------ 5
if os.path.exists(FINDINGS):
    md(open(FINDINGS).read())

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, "notebooks/clinical_records_analytics.ipynb")
