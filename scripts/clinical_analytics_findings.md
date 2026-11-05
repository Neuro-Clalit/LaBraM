## 5. Findings

**Demographics and verdict (Tables 2.1–2.4, Figures 2.1–2.2).**

* **Age and sex mix.** The three corpora are adult and close in age (mean 49 years for TUAB, 52 for the TUH
  cohort, 54 for HEEDB) and about half women (51–53 %).
* **Report coverage.** TUAB is balanced by design (51 % normal). The TUH cohort is 53 % abnormal. Half of the
  HEEDB recordings have no report verdict (49 % unknown); among the rest, 60 % are abnormal.
* **Abnormal share rises with age** in every corpus (Eq. (2.1), Figure 2.2): from 30–44 % under 40 to 72–77 % at
  80+. At every age men are slightly more often abnormal than women (TUAB 52 vs 47 %, TUH cohort 61 vs 55 %,
  HEEDB 61 vs 60 %).
* **Consequence for training.** Age and abnormality are strongly confounded. A brain-age model trained on all
  recordings sees mostly abnormal EEGs at old age, and a pathology model can lean on age-related EEG change
  (cf. the bias-corrected brain-age gap in `docs/brain_age_bias_correction.md`).

**Pathologies (Tables 3.1–3.5, Figures 3.1–3.3).**

* **The corpora describe pathology very differently.**
  * HEEDB's per-patient ICD-10 history is rich: seizure disorders 35 %, headache 28 %, sleep disorders 22 %,
    depression 20 %, stroke / cerebrovascular 15 %, movement disorders 15 %, dementia 8 %, tumours 6 %.
  * The TUH sheet notes much less per session: seizure / epilepsy 9 %, stroke 4 %, substance use 4 %, and keyword
    mentions such as altered mental status 8 % or syncope 4 %.
  * TUAB's diagnosis columns are essentially empty (Section 1).
  * Rates are therefore comparable *within* a corpus, not across corpora.
* **Headache / migraine** is the most common non-seizure category in HEEDB (28 %).
  * **Sex:** about 1.4× more frequent in women (32 vs 23 %); the TUH mentions show the same sex ratio (TUAB 7.4
    vs 2.9 %, TUH cohort 3.0 vs 1.3 %).
  * **Age:** in TUH it is a disease of the young (3–4 % of mentions under 50, < 1 % above 60); HEEDB's lifetime
    codes stay at 23–32 % across ages (Figure 3.3).
  * **In TUH it marks a normal EEG:** 10 % of TUAB and 33 % of TUH-cohort recordings with a headache mention are
    abnormal, against 46 % and 58 % without (Table 3.3).
  * **In HEEDB it has no such effect** (60 % vs 60 %), because the ICD code is lifetime history and co-occurs
    with other neurological disease: of HEEDB headache patients' recordings, 52 % also carry a seizure disorder,
    38 % depression and 39 % a sleep disorder (Table 3.4).
* **Pathologies that raise the abnormal share most** (Eq. (3.2), Table 3.3):
  * stroke (TUH cohort 82 % vs 56 % abnormal, +26 pp);
  * anoxic injury (+24 pp) and altered mental status (+23 pp);
  * brain tumour (+14 pp TUH, +19 pp HEEDB) and CNS infection (+18 pp HEEDB);
  * dementia (+15 pp TUH, +15 pp HEEDB).
* **Pathologies that do not raise it.** Psychiatric and benign neurological categories go with an equal or
  *lower* abnormal share:
  * TUH cohort: depression −13 pp, anxiety −9, ADHD −14, MS −6, headache −25, syncope −24.
  * HEEDB: depression −1, MS −3, headache 0, ADHD −16.
  * Their patients are younger and their diagnoses leave no consistent EEG trace, so they are hard targets for
    an EEG classifier.
* **Sex differences beyond headache:**
  * substance use is twice as frequent in men in TUH (5.6 vs 2.9 %, TUAB 6.0 vs 2.5 %);
  * in HEEDB, depression (23 vs 17 %) and multiple sclerosis (1.9 vs 0.9 %) are more frequent in women;
  * stroke is slightly more frequent in men in both corpora (4.2 vs 3.7 %, 15.5 vs 13.7 %).
* **Comorbidity is the rule in HEEDB:** 43 % of the dementia patients' recordings also carry a seizure
  disorder, and 33 % stroke (Table 3.4). Per-pathology targets are therefore far from mutually exclusive, and a
  multi-label (not multi-class) formulation is the right one for the pathology heads.

**Abnormality types (Tables 4.1–4.3, Figure 4.1).**

* **Slowing dominates.**
  * TUH: diffuse slowing in 61 % and focal slowing in 44 % of abnormal reasons, epileptiform discharges in 34 %,
    suppression / low voltage in 20 %, seizures in 4 %.
  * HEEDB's structured findings name generalised slowing in 87 %, epileptiform or periodic discharges in 38 %,
    focal slowing in 23 % and a seizure in 17 % of abnormal recordings.
  * The higher HEEDB generalised-slowing and seizure rates partly reflect structured coding (every mention
    counts) against keyword rules on a short TUH reason text.
* **Types follow the pathologies as expected** (lift, Eq. (4.2)):
  * TUH: seizures are 4.4× over-represented and epileptiform discharges 1.3× in seizure / epilepsy patients;
    suppression / low voltage is 2.5× over-represented after anoxic injury; focal slowing 1.7× with brain
    tumours and 1.3× after stroke.
  * HEEDB shows the same focal-slowing lift for tumours (1.7×) and the epileptiform lift for seizure disorders
    (1.3×).
* **Rare TUH groups, wide error bars.** In the TUH cohort the rarer categories (MS, ADHD, schizophrenia,
  dementia) have under 100 positives each. Their prevalence and lifts carry wide uncertainty and should not be
  over-read.

**What this means for the pathology model (E2P).** E2P (TUH + HEEDB, status + 17 diagnosis groups,
`dataset_maker/make_multitask_targets.py`) inherits these properties:

* **HEEDB dominates most groups.** It provides most positives for seizure, stroke, dementia, depression and all
  of headache, sleep, tumour and movement disorders.
* **TUH-only groups.** The psychiatric, substance and TBI groups come from TUH only (HEEDB unknown).
* **Undated labels.** HEEDB groups are lifetime diagnoses, so EEG-invisible history (e.g. a remote headache
  code) caps the reachable ROC-AUC.
* **Expected detectability** (Table 3.3): stroke, tumour, CNS infection, dementia and the seizure groups should
  be detectable above chance; depression, ADHD, MS and headache will be close to it.
