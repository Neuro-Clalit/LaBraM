# Open external cohorts after Dortmund: SRM, AHEPA, Iowa and Tel Aviv Parkinson's, NMT

Onboarded 2026-11-17 with `scripts/external_runs/open_cohorts.sh <cohort> <step>` (skill `onboard-eeg-dataset`). All are
test-only: nothing in them is used to fit, tune, select or calibrate a model, and their ages only score. They are the
open-licence candidates of Table 23.1 of `docs/tdbrain_onboarding.md`; CAUEEG and NMT-4K-EEG need a signed agreement and
the TUM pain dataset's licence is unverified, so they are not part of this round.

## 1. Summary

1. **SRM (healthy, 17–71 years) is predicted at the level of the other healthy cohorts:** P3 ensemble 4.75 years
   [4.13, 5.42], $R^2 = 0.79$ on 152 recordings of 110 adults, zero-shot (Table 5.1).
2. **The three elderly disease cohorts are not usable as accuracy benchmarks.** Their healthy controls are predicted
   7 to 11 years too young, 4 to 7 years more than in-domain recordings of the same ages, with a wider error spread
   (Table 6.1). The label-free signal check finds no gain or band difference (Table 4.1); the cause is not identified.
3. **Inside each cohort, patients look older than the controls recorded with the same setup** (Table 7.1):
   Alzheimer's disease +6.3 years [1.0, 11.6]; frontotemporal dementia +4.8 [−1.2, 10.8]; Parkinson's disease +3.0
   [−0.4, 6.5] in Iowa and +2.6 [−0.6, 5.8] in Tel Aviv, +2.8 [0.5, 5.2] pooled. The gap does not follow MMSE, MoCA or
   UPDRS inside the patient groups.
4. **NMT** (routine clinical EEG, Pakistan, normal / abnormal): Section 8.

## 2. Data, terms and storage

### Table 2.1. The cohorts

| cohort | source, licence | people (recordings) | groups | age (y) | recording | protocol |
|---|---|---|---|---|---|---|
| `srm_v1` | OpenNeuro ds003775, CC0 | 111 (153: 42 have a second session 2–13 months later) | healthy | 17–71, 38.6 ± 13.6 | BioSemi ActiveTwo, 64 channels (10-10), 1024 Hz, unfiltered, released average-referenced | 4 min eyes closed, cut out of an evoked-potential paradigm |
| `ahepa_v1` | OpenNeuro ds004504, CC0 | 88 (88) | 36 Alzheimer's, 23 frontotemporal dementia, 29 controls; MMSE | 44–79, 66.2 ± 7.4 | Nihon Kohden EEG-2100, the 19 10-20 channels, 500 Hz, time constant 0.3 s, 70 Hz high cut | eyes closed, 5–21 min (median 13.6) |
| `iowapd_v1` | OpenNeuro ds004584, CC0 | 149 (149) | 100 Parkinson's, 49 controls; MoCA, UPDRS | 48–86, 69.3 ± 8.0 | BrainVision cap, 63–66 channels (10-10), reference Pz, 500 Hz | eyes open, 2–6 min (median 2.4) |
| `telavivpd_v1` | OpenNeuro ds007526, CC0 | 144 (144 resting runs) | 116 Parkinson's, 28 controls; MoCA, UPDRS, LEDD | 39–84, 65.1 ± 9.3 | EGI Geodesic 400, 60 EEG channels with 10-10 labels + vertex reference, 250 Hz | eyes open, 4 min (a walking run is not converted) |

All four are CC0: no agreement, no login, redistribution allowed. Sources are the public OpenNeuro bucket
(`aws s3 sync --no-sign-request`; the preprocessed `derivatives` of AHEPA are not downloaded). Storage:
`/data/datasets/EEG-public/{SRM,AHEPA,IOWAPD,TELAVIVPD}/{raw,meta,cohorts}`, mirrored to
`s3://eeg-data-public/<NAME>/v1/` (bucket checked private before every upload). None of the four is in LaBraM's
pre-training corpus (Table 23.2 of `docs/tdbrain_onboarding.md`); SRM and the Iowa set are in the pre-training corpus
of other foundation models (REVE, LEAD).

## 3. Recording chain and readers

### Table 3.1. What each reader has to handle (`dataset_maker/make_external_npy.py`)

| cohort | finding of the audit | what the reader does |
|---|---|---|
| SRM | the EDF headers have an empty physical unit, which MNE reads as volts; with microvolts the alpha peak is a few µV²/Hz and the 64 channels sum to zero, as an average reference must | scale to microvolts; remove each channel's mean (DC-coupled); keep the average reference |
| SRM | the listed age is that of the first session | a second-session age adds the time between the two `acq_time` stamps |
| AHEPA | README and sidecar disagree on the reference (Cz or A1 / A2) and on the filters (70 Hz or "0.4–50"). In the data Cz is not flat and the spectrum falls smoothly to 100 Hz without a 50 Hz edge | keep the 19 channels as exported; the models' common average reference removes any common reference |
| Iowa | reference Pz, so Pz is not in the file (four channel layouts, none has it) | add Pz as the zero reference channel and re-reference the 19 channels to their mean, which gives Pz its signal back exactly |
| Tel Aviv | `VREF` is the vertex reference, i.e. Cz, and flat; DC offsets up to 75 mV; 22 runs carry a `break cnt` mark | rename `VREF` to Cz and re-reference the 19 channels to their mean; remove the offset; keep the longest continuous piece |

Every recording then goes through the training pipeline (`make_TUH_npy.preprocess_raw`: 0.1–75 Hz, line removal at 50
and 60 Hz, 200 Hz, µV, float16). With $x_c$ the recorded channels, referenced to an electrode $q$ that is one of the 19
(Pz in Iowa, Cz in Tel Aviv), adding $x_q = 0$ and subtracting the mean gives the true common average reference:

$$ x_c - \frac{1}{19}\sum_{k=1}^{19} x_k = (s_c - s_q) - \frac{1}{19}\sum_{k}(s_k - s_q) = s_c - \frac{1}{19}\sum_{k} s_k \tag{3.1} $$

with $s$ the potentials against any common point; for $c = q$ the left side is $-\frac{1}{19}\sum_k x_k$.

### Table 3.2. The converted cohorts

| cohort | recordings converted | windows of 10 s | windows per recording, median (min) | scored without a trim |
|---|---|---|---|---|
| `srm_v1` | 153 of 153 | 3,672 | 24 (24) | yes |
| `ahepa_v1` | 88 of 88 | 7,013 | 80 (30) | no: the usual 60 s at each end |
| `iowapd_v1` | 149 of 149 | 2,327 | 14 (12) | yes |
| `telavivpd_v1` | 144 of 144 | 3,456 | 24 (24) | yes |

No recording failed. All 19 channels are present in every recording.

## 4. Label-free signal check

`scripts/domain_shift_analysis.py`, as for Dortmund: features on what the models see, against healthy Temple and
Harvard validation recordings weighted to each cohort's age distribution (for description only).

### Table 4.1. Medians against the training data (`docs/data/external_eval/<cohort>/feature_shift_summary.csv`)

Reference columns: Harvard / Temple at the cohort's ages. Power in µV²/Hz.

| feature | SRM | ref. | AHEPA | ref. | Iowa | ref. | Tel Aviv | ref. |
|---|---|---|---|---|---|---|---|---|
| channel standard deviation (µV) | 13.1 | 12.3 / 9.8 | 12.2 | 11.5 / 9.5 | 11.1 | 11.8 / 9.6 | 11.4 | 11.7 / 9.5 |
| power 0.1–1 Hz | 101 | 83 / 43 | 92 | 69 / 43 | 64 | 72 / 44 | 62 | 70 / 43 |
| power 8–13 Hz | 2.35 | 0.91 / 1.02 | 1.21 | 0.86 / 0.97 | 1.01 | 0.86 / 0.92 | 2.38 | 0.86 / 0.96 |
| power 30–45 Hz | 0.076 | 0.056 / 0.063 | 0.076 | 0.055 / 0.043 | 0.119 | 0.058 / 0.042 | 0.075 | 0.058 / 0.046 |
| power 45–70 Hz | 0.040 | 0.028 / 0.033 | 0.042 | 0.030 / 0.024 | 0.090 | 0.031 / 0.024 | 0.051 | 0.030 / 0.025 |
| relative alpha power | 0.32 | 0.16 / 0.17 | 0.20 | 0.18 / 0.17 | 0.15 | 0.17 / 0.17 | 0.33 | 0.17 / 0.17 |
| aperiodic exponent | 1.28 | 1.38 / 1.31 | 1.21 | 1.26 / 1.43 | 1.10 | 1.27 / 1.44 | 1.14 | 1.28 / 1.39 |
| alpha peak frequency (Hz) | 10.0 | 10.0 / 10.0 | 9.5 | 9.5 / 9.5 | 8.5 | 9.5 / 9.5 | 9.0 | 9.5 / 9.5 |
| mean absolute correlation between channels | 0.34 | 0.45 / 0.41 | 0.36 | 0.44 / 0.41 | 0.46 | 0.44 / 0.41 | 0.35 | 0.44 / 0.41 |

**What Table 4.1 shows:**
- **No gain and no band problem in any cohort.** Amplitudes are 11 to 13 µV, between Temple and Harvard, and the power
  above 45 Hz is at or above the training data's. This is unlike CHBP (amplitude 0.6×, 4 % of the 45–70 Hz power).
- **More alpha than hospital recordings** in SRM and Tel Aviv (2.4 against 0.9–1.0 µV²/Hz) and in the AHEPA controls
  (relative alpha 0.35 against 0.17 for training recordings from age 60): alert volunteers against routine clinical
  EEG, where about half the recordings contain drowsiness.
- **Iowa has three times the 45–70 Hz power** and the flattest spectrum (exponent 1.10): eyes open, more muscle
  activity.
- **Lower correlation between channels** in SRM, AHEPA and Tel Aviv (0.34–0.36 against 0.41–0.45).

## 5. Zero-shot accuracy

Every window (AHEPA after the usual 60 s trims, the others without a trim); models and metrics as in
`docs/dortmund_onboarding.md` Section 6; intervals from 2,000 bootstrap resamples of people. Recordings of adults only
(one SRM participant is 17).

### Table 5.1. Accuracy per recording (`docs/data/external_eval/<cohort>/external_cohort_table.csv`)

MAE in years with a 95 % interval / bias / r.

| cohort | n | E2L | device-robust E2L | P3 ensemble | P4 ensemble |
|---|---|---|---|---|---|
| SRM, healthy | 152 | 5.01 [4.37, 5.70] / −0.1 / 0.88 | 8.02 [7.09, 9.00] / +6.6 / 0.85 | **4.75 [4.13, 5.42] / −1.3 / 0.90** | 4.84 [4.23, 5.50] / −1.1 / 0.89 |
| AHEPA, all | 88 | 12.04 [10.38, 13.87] / −6.8 / −0.09 | **10.69 [9.10, 12.36] / −4.4 / −0.07** | 11.54 [9.90, 13.31] / −4.5 / −0.07 | 11.23 [9.61, 12.92] / −4.3 / −0.08 |
| AHEPA, controls | 29 | 13.93 / −12.2 / −0.02 | 11.74 / −10.1 / −0.03 | 11.93 / −9.3 / 0.04 | 11.94 / −9.6 / 0.03 |
| Iowa, all | 149 | 12.64 [11.28, 14.18] / −9.5 / 0.28 | **10.09 [9.01, 11.33] / −7.1 / 0.40** | 10.96 [9.76, 12.25] / −7.1 / 0.38 | 10.91 [9.74, 12.10] / −7.8 / 0.38 |
| Iowa, controls | 49 | 15.07 / −13.5 / 0.25 | 10.44 / −9.7 / 0.51 | 12.20 / −10.2 / 0.40 | 12.74 / −10.9 / 0.34 |
| Tel Aviv, all | 144 | 9.23 [8.21, 10.31] / −6.9 / 0.60 | **7.89 [6.93, 8.82] / −4.3 / 0.54** | 8.21 [7.29, 9.22] / −4.8 / 0.65 | 8.66 [7.67, 9.70] / −6.0 / 0.63 |
| Tel Aviv, controls | 28 | 9.38 / −7.8 / 0.45 | 7.84 / −4.3 / 0.42 | 9.12 / −6.3 / 0.43 | 9.81 / −7.0 / 0.40 |

**What Table 5.1 shows:**
- **SRM joins LEMON, TDBRAIN and Dortmund:** 4.75 to 5.01 years with the raw-input models, $R^2$ 0.77–0.79, almost no
  bias. Rejecting the 20 % least certain recordings gives 4.39 to 4.60 years (P4 ensemble). The device-robust model is
  3 years worse, with the young-age offset it has everywhere (+6.6 years at a mean age of 39).
- **The three elderly cohorts fail,** in the same direction: everyone is predicted too young, the controls by 7 to 13
  years, and in AHEPA the prediction does not follow age at all.
- **The device-robust model is the best model on all three** (by 0.3 to 0.8 years against the ensembles and 1.3 to
  2.6 against E2L) and the worst on SRM: a sign that part of the error goes with amplitude or band, although
  Table 4.1 shows no shift of their medians.

## 6. Why the elderly controls look too young

### Table 6.1. Controls against in-domain recordings of the same ages, P4 ensemble

In-domain: test recordings of the training cohort with a normal EEG report, weighted to the controls' age distribution
(first 5 minutes). Bias and standard deviation of the gap in years.

| controls | n | mean age | bias | SD of the gap | in-domain bias, Temple / Harvard | in-domain SD | excess bias |
|---|---|---|---|---|---|---|---|
| AHEPA | 29 | 67.9 | −9.6 | 10.5 | −3.7 / −2.8 | 6.2 / 7.0 | −6 to −7 |
| Iowa | 49 | 70.9 | −10.9 | 10.7 | −5.8 / −4.1 | 6.6 / 7.5 | −5 to −7 |
| Tel Aviv | 28 | 60.8 | −7.0 | 8.3 | −3.2 / −0.9 | 5.8 / 7.4 | −4 to −6 |
| SRM | 153 | 38.6 | −1.0 | 6.2 | +0.5 / +1.6 | 5.2 / 7.3 | −1.5 to −2.6 |
| Dortmund, 60 and over (for comparison) | 180 | 65.6 | −4.2 | | | | |

What is established:

- **Part is the known old-age compression.** In-domain recordings of these ages are under-predicted by 1 to 6 years,
  and Dortmund's healthy volunteers from 60 on by 4.2 years.
- **An excess of 4 to 7 years remains,** with an error spread of 8 to 11 years against 6 to 7 in-domain. It is specific
  to these three cohorts: SRM and Dortmund, both research recordings of healthy volunteers, do not show it.
- **It is not a gain or a band limit** (Table 4.1), the two causes found for CHBP.

What is not established: the cause. The candidates cannot be separated with these data:

- **Population and state.** A hospital patient with a normal report at 70 is not a healthy volunteer of 70: the
  training recordings are routine clinical EEG with drowsiness and medication, and the AHEPA controls have twice the
  relative alpha power of training recordings from age 60. A model calibrated on patients may read an alert,
  alpha-rich elderly EEG as a younger patient. In the AHEPA controls the gap goes with the alpha peak frequency
  (partial Spearman −0.41 given age) and with the correlation between channels (−0.55), in Iowa with delta and theta
  power (±0.30); with 29 and 49 controls these are leads, not findings.
- **Protocol.** Iowa and Tel Aviv are eyes-open only, 2 to 4 minutes. In Dortmund eyes-open and eyes-closed runs are
  predicted alike (`docs/dortmund_onboarding.md` Table 7.1), which speaks against this cause.
- **Device.** Each cohort has its own recording system (clinical Nihon Kohden export, BrainVision referenced to Pz,
  EGI net at 250 Hz). Against E2L the device-robust model lowers the controls' bias by 2 to 4 years in all three,
  and still leaves −4 to −10 years.

The forward and inverse tests of the skill (make training recordings look like the cohort; normalise the cohort toward
the training data) were not run: with 28 to 49 controls per cohort they would not separate these causes.

**Consequence.** These three cohorts are not accuracy benchmarks for our models and should not be read as such. They
can serve one purpose, the comparison of patients and controls inside a cohort, where both groups share the recording
chain, the protocol and the recruitment (Section 7).

## 7. Patients against controls inside a cohort

`scripts/clinical_cohort_gap.py`. With the gap $g_r = \hat y_r - y_r$, on a patient group and the controls of the same
cohort,

$$ g_r = b_0 + b_1\,x_r + b_2\,y_r + b_3\,m_r + \varepsilon_r \tag{7.1} $$

where $x_r$ is 1 for a patient and $m_r$ is 1 for a man; $b_1$ is how much older the group looks than the controls at
the same age and sex, with heteroscedasticity-robust (HC3) standard errors. The age term is needed because the gap
falls with age (compression) and the groups differ in age. Inside a patient group, the relation of the gap to a
clinical score $s$ is the correlation of the two residuals after a linear fit on age:

$$ r_{g,s \mid y} = \operatorname{corr}\big( g - \hat g(y),\ s - \hat s(y) \big) \tag{7.2} $$

### Table 7.1. Adjusted gap against the cohort's controls (`docs/data/external_eval/<cohort>/gap_by_group.csv`)

$b_1$ of Eq. (7.1) in years with a 95 % interval and p.

| cohort, group | patients / controls | E2L | device-robust E2L | P3 ensemble | P4 ensemble |
|---|---|---|---|---|---|
| AHEPA, Alzheimer's disease | 36 / 29 | +6.4 [1.1, 11.6], 0.02 | +7.5 [2.6, 12.3], 0.004 | +5.5 [−0.1, 11.1], 0.06 | +6.3 [1.0, 11.6], 0.02 |
| AHEPA, frontotemporal dementia | 23 / 29 | +5.3 [−0.8, 11.4], 0.09 | +5.0 [−0.5, 10.4], 0.08 | +4.3 [−2.0, 10.7], 0.19 | +4.8 [−1.2, 10.8], 0.13 |
| AHEPA, both | 59 / 29 | +5.7 [1.0, 10.3], 0.02 | +6.1 [1.9, 10.4], 0.006 | +4.7 [−0.2, 9.6], 0.06 | +5.4 [0.7, 10.0], 0.03 |
| Iowa, Parkinson's disease | 100 / 49 | +3.7 [0.0, 7.5], 0.05 | +1.8 [−1.1, 4.7], 0.24 | +3.2 [−0.6, 7.0], 0.10 | +3.0 [−0.4, 6.5], 0.09 |
| Tel Aviv, Parkinson's disease | 116 / 28 | +2.8 [−0.3, 5.9], 0.08 | +2.2 [−0.8, 5.2], 0.15 | +3.0 [−0.3, 6.3], 0.07 | +2.6 [−0.6, 5.8], 0.11 |
| Parkinson's disease, both cohorts pooled | 216 / 77 | +3.2 [0.8, 5.6] | +2.0 [−0.1, 4.1] | +3.1 [0.6, 5.6] | +2.8 [0.5, 5.2] |

The pooled row is the inverse-variance weighted mean of the two cohorts' estimates.

**What Table 7.1 shows:**
- **Dementia patients look about 5 to 6 years older than the controls of the same clinic,** Alzheimer's disease 6.3
  years (P4 ensemble, p = 0.02). The direction and size are the same with all four models; with 36 patients and 29
  controls the interval is wide (1 to 12 years).
- **Parkinson's disease: about 3 years, the same in two independent cohorts.** Neither cohort alone is significant;
  pooled, the interval excludes 0 for the raw-input models (+2.8 years [0.5, 5.2] for the P4 ensemble). With the
  amplitude-normalised model the estimate is smaller (+2.0 [−0.1, 4.1]).
- **These agree with the Harvard record-based estimates** of `docs/tdbrain_onboarding.md` Table 22.2: dementia +2.1
  years overall and +2.6 with an abnormal EEG, movement disorders +1.8. The clinic cohorts give larger values for
  dementia, as expected for diagnosed, symptomatic patients (mean MMSE 18 and 22) against a diagnosis code anywhere in
  a hospital record.
- **The gap does not track severity inside a group:** its partial correlation with MMSE is +0.13 (Alzheimer's) and
  −0.12 (frontotemporal), with MoCA −0.04 and −0.14, with UPDRS +0.09 and −0.06 (P4 ensemble; all p > 0.1).

Cautions. The groups are small, each estimate rests on one cohort's controls (28 to 49 people), and Section 6 shows
that the models are outside their reliable range on these cohorts; a contrast inside a cohort removes a shift common to
both groups, not one that interacts with the disease. Medication is not controlled (the Parkinson's patients take
levodopa). Nothing here is a diagnostic claim.

## 8. NMT

Routine clinical EEG of the Pak-Emirates Military Hospital, Rawalpindi (Khan et al. 2022; Zenodo
10.5281/zenodo.21629135, CC BY 4.0): 2,417 recordings of unique patients, 2,002 normal and 415 abnormal, with age and
sex. The 13.4 GB archive was still downloading from Zenodo when this section was written; conversion and scoring follow
with `scripts/external_runs/open_cohorts.sh nmt all`, and the results will be added here.

## 9. Reproduce

```bash
for c in srm ahepa iowapd telavivpd; do
  scripts/external_runs/open_cohorts.sh $c fetch     # public OpenNeuro bucket
  scripts/external_runs/open_cohorts.sh $c mirror    # private S3 copy
  scripts/external_runs/open_cohorts.sh $c all       # convert + score + tables
done
```

Table 4.1: `scripts/domain_shift_analysis.py --cohort <name>=<cohort> --ref-cohort <tuh_heedb_v1>` (`--ext-trim 0` for
the three short cohorts). Table 6.1: the test split of `own_5min_models.json`, normal reports, weighted with
`domain_shift_analysis.age_weights`.

## 10. References

- Hatlestad-Hall C, Rygvold TW, Andersson S. SRM Resting-state EEG. OpenNeuro ds003775. doi:10.18112/openneuro.ds003775.v1.0.0
- Miltiadous A et al. 2023. A dataset of scalp EEG recordings of Alzheimer's disease, frontotemporal dementia and
  healthy subjects from routine EEG. *Data* 8(6):95. doi:10.3390/data8060095. OpenNeuro ds004504.
- Singh A, Cole R, Espinoza A, Cavanagh J, Narayanan N. Rest eyes open. OpenNeuro ds004584.
  doi:10.18112/openneuro.ds004584.v1.0.0
- PD-EEG: Resting-State & Walking EEG in Parkinson's Disease. Tel Aviv Sourasky Medical Center. OpenNeuro ds007526.
  doi:10.18112/openneuro.ds007526.v1.0.2
- Khan HA et al. 2022. The NMT Scalp EEG Dataset. Zenodo. doi:10.5281/zenodo.21629135
