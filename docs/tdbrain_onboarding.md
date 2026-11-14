# TDBRAIN onboarding: external test set for brain age, with clinical labels

Status: release 3.1 downloaded, extracted and QC-checked 2026-11-14 (Section 7). Conversion
and scoring wait for the participants table (age, sex, indication), which is not in the EEG zip.

TDBRAIN (Two Decades-Brainclinics Research Archive for Insights in Neurophysiology,
van Dijk et al. 2022, *Sci Data* 9:333, [doi:10.1038/s41597-022-01409-z](https://doi.org/10.1038/s41597-022-01409-z))
is a single-site clinical EEG archive from Nijmegen (NL), recorded 2001–2021. It is
onboarded the way LaBraM evaluates LEMON, MIPDB and CHBP (`docs/external_eval_results.md`,
`docs/chbp_onboarding.md`): as a **test-only external cohort** that the brain-age models
trained on TUH + HEEDB (`tuh_heedb_v1`) score without retraining. Unlike those healthy-volunteer
corpora it is mostly **psychiatric patients**, and it carries the clinical labels, so the
brain-age gap can be read per diagnosis.

Code:
- `dataset_maker/make_external_npy.py --dataset tdbrain`: BIDS BrainVision to the npy cohort
  format through the training pipeline (`make_TUH_npy.preprocess_raw`). It also writes the
  clinical table `clinical.parquet` (`tdbrain_clinical_table`) and puts the clinical flags in
  the manifest;
- `scripts/external_runs/tdbrain.sh`: fetch, unpack, private S3 mirror, convert, score, report;
- `scripts/eval_age_by_cohort.py --trim-start-sec / --trim-end-sec`: trim overrides. A
  TDBRAIN file is 2 min long, and the training trims (60 s at each end) would drop all of it;
- `scripts/external_eval_report.py`: TDBRAIN subgroups (age bands, diagnoses, eyes closed / open);
- `tests/test_tdbrain.py`: a synthetic BIDS tree (BrainVision, participants.tsv with decimal
  commas, a blinded `sub-19*` subject, a derivatives copy) converted end to end.

## 1. The dataset

### Table 1.1. TDBRAIN as published (van Dijk et al. 2022, via PubMed Central PMC9198070)

| | |
|---|---|
| participants / sessions | 1,274 / 1,346 (some participants have several sessions, e.g. before and after treatment) |
| age, sex | 5–88 y (abstract: 5–89), mean 38.7 ± 19.2; 620 female |
| main indications | MDD 426, ADHD 271, SMC (subjective memory complaints) 119, OCD 75; also healthy controls and smaller groups |
| diagnosis status | `formal_status`: a DSM-5 diagnosis (clinician and/or structured interview above a clinical cut-off) **or** a referral indication (unofficial, from a GP / psychologist); `UNKNOWN` where missing, which "does not mean these are healthy participants" |
| blinded replication set | 30 % of the MDD / ADHD / OCD diagnostic and outcome labels are withheld: `participant_id` starting with `sub-19`, `REPLICATION` in participants.tsv |
| protocol | 2 min eyes open (`restEO`), then 2 min eyes closed (`restEC`); then an auditory oddball and a visual 1-back task (only behavioural data released for those) |
| EEG | 26 channels, 10-10 (Compumedics Quickcap / ANT Waveguard, Ag/AgCl), 500 Hz, low-pass 100 Hz before digitisation, DC-coupled, 24 bit; virtual ground, re-referenced offline to averaged mastoids (A1/A2), ground AFz, < 10 kΩ |
| other channels | vertical + horizontal EOG, ECG (Erb's point), EMG (right masseter) |
| line frequency | 50 Hz (the Netherlands) |
| format | BIDS, BrainVision (`.vhdr/.vmrk/.eeg`) per session and task + `.json` / `channels.tsv`; a `derivatives/` copy as `.csv`; `participants.tsv` + `.json` in both trees |
| other data | gender, age, height, weight, sleep, education, alcohol and drug use, NEO-FFI item scores, day-of-measurement questions; for the published treatment studies, primary outcomes (e.g. rTMS in MDD n = 176, neurofeedback in ADHD n = 102, SMA-rTMS in OCD n = 17) |
| medication | allowed and **not** systematically tracked; short half-life drugs (methylphenidate) skipped on the morning of the EEG |
| size | "15 GB" (brainclinics.com), split zip files |

## 2. Access, DUA and storage

### Table 2.1. How the data can be obtained (checked 2026-11-12)

| route | result |
|---|---|
| brainclinics.com/resources | ORCID login, then the DUA, then the download page with the split zips. **The only working route.** It has to be done by a person, in a browser |
| Synapse `syn25671079` | Our Synapse account (`BenningerLab`) can list the files (`TD-BRAIN-DATASET.zip` syn26253372, `TD-BRAIN-SAMPLE.zip` syn26241849, `TD_BRAIN_code.zip` syn26241847, `participants.tsv` syn26468893) but `canDownload` is false. Only the authors and their 2021 review team have DOWNLOAD in the ACL. The project wiki says to download from brainclinics.com while the Synapse terms are being set up |

The DUA text (`/data/datasets/EEG-public/TDBRAIN/meta/TD_BRAIN-DataUseAgreement.pdf`, from
the Synapse wiki) sets these rules:

### Table 2.2. DUA terms and what they mean here

| DUA clause | our handling |
|---|---|
| 1. comply with institutional / ethics rules | IRB / ethics cover for the secondary use is the user's call |
| 2. no re-identification, no linkage to other databases that could identify | nothing is joined to TDBRAIN except our own model predictions |
| 3. no redistribution, **not even within the research group**, unless each person has accepted the DUA themselves | the copy sits in the private `s3://eeg-data-public/TDBRAIN/v1/` (same rule as HEEDB: `check_private` before every sync, never public). Only people who have accepted the TDBRAIN DUA themselves may read it. Notebooks and docs show aggregates only |
| 4. no changes to the raw data | the zips and the extracted files stay as downloaded; conversion only reads them |
| 5. credit: link to brainclinics.com/resources and cite van Dijk et al. | in every result note or figure using TDBRAIN |
| 6. the code is MIT-licensed, with the same credit | applies if we reuse their preprocessing code |

| | local | S3 (private) |
|---|---|---|
| zips as downloaded | `/data/datasets/EEG-public/TDBRAIN/zips` | `s3://eeg-data-public/TDBRAIN/v1/zips` |
| extracted BIDS tree | `/data/datasets/EEG-public/TDBRAIN/raw` | `s3://eeg-data-public/TDBRAIN/v1/raw` |
| paper + DUA PDFs | `/data/datasets/EEG-public/TDBRAIN/meta` (downloaded) | `s3://eeg-data-public/TDBRAIN/v1/meta` |
| cohort | `/data/datasets/EEG-public/TDBRAIN/cohorts/tdbrain_v1` | `s3://eeg-data-public/TDBRAIN/v1/cohorts/tdbrain_v1` |

## 3. What a recording carries: EEG, annotations, "report", health record

TDBRAIN has no clinical EEG report and no medical record. In the per-recording tables
we use for TUH and HEEDB (`labram/eval/clinical_analytics.py`), its fields map as follows:

### Table 3.1. TDBRAIN fields against the TUH / HEEDB per-recording schema

| our field | TUH / HEEDB source | TDBRAIN |
|---|---|---|
| signal | EDF | BrainVision `restEC` and `restEO`, 2 min each, converted as **two recordings per session** (`sub-<id>_ses-<n>_<task>_tdbrain`) |
| annotations | EDF annotations / HEEDB `annotations.csv` | BrainVision markers (`.vmrk`). The converter counts them per file (`annotations` in the manifest): a mid-file `New Segment` would mean a discontinuity. The eyes-open / eyes-closed split is in the file names (`task`). No artifact marks are shipped; the authors' preprocessing code (`TD_BRAIN_code.zip`) detects EMG, jumps, kurtosis, voltage swings, blinks, bridging and extreme correlations, if we want them |
| report verdict (`label`, `abnormal_report`) | NEDC sheet / HEEDB findings | **none**: no EEG was read clinically. `label` is `normal` for healthy controls only, and patients stay unlabelled (`unlabelled_*` in the per-cohort metrics). A psychiatric indication is not an EEG finding |
| diagnosis groups (`dx_*`) | ICD-10 (HEEDB), free text (TUH) | `indication`, split on `/ , ; + &`, gives `dx_mdd`, `dx_adhd`, `dx_ocd`, `dx_smc` and `dx_other` (anything else, e.g. tinnitus, burnout), plus `healthy`, `blinded` (sub-19* / REPLICATION) and `unknown`. `formal_status` says whether a diagnosis is formal (DSM) or a referral |
| medication (`med_*`) | ATC (HEEDB), NEDC drug classes (TUH) | not recorded |
| demographics | age, sex | age **with decimals** (per session), sex (decoded through the `participants.json` levels), plus education, height, weight, sleep, alcohol and drug use, all kept in `clinical.parquet` |
| questionnaires, outcomes | none | NEO-FFI items, day-of-measurement questions and treatment outcomes where published. Every participants.tsv column is kept unchanged in `clinical.parquet` |

## 4. Conversion

The signal path is the one every training and external cohort went through
(`make_TUH_npy.preprocess_raw`): 0.1–75 Hz band-pass, spectrum-interpolation line removal at
50 and 60 Hz, resampling to 200 Hz, µV, float16, the 23 TUH channels in model order. The
reference is kept as recorded (averaged mastoids), because the models apply a common average
reference at their input. TDBRAIN-specific choices:

- **19 of 23 channels.** All 19 10-20 positions exist (T7/T8/P7/P8 → T3/T4/T5/T6). A1/A2 are
  the reference and T1/T2 are not recorded; these four are zero-filled and flagged in
  `channel_mask`, as for CHBP and HEEDB.
- **DC offset removed first.** The recording is DC-coupled, so each channel's mean is
  subtracted before the 0.1 Hz high-pass. Otherwise a large offset leaves a filter transient
  over the first seconds of a 2-minute file. The synthetic test puts a 20 mV offset on every
  channel and checks that the output stays within ±200 µV.
- **Eyes closed and eyes open are separate recordings**, 12 whole 10 s windows each (120 s).
  They are scored separately and can be pooled per session afterwards. They are not
  concatenated: the files are separate acquisitions, and joining them would add a step.
- **Scoring windows.** The models were trained with a 60 s trim at each end and a 5-minute
  evaluation budget. Here every window is scored (`--trim-start-sec 0 --trim-end-sec 0
  --eval-minutes 0`). A second pass drops 10 s at each end (`inner`, 10 windows) to check
  that filter edges do not matter.
- **Age.** The age comes from the participants.tsv row of the same session
  (`age_session_matched`). It falls back to the subject's first row when the file has no
  session column. Rows without an age are dropped.

## 5. Evaluation plan

1. **QC against the training data**, as in `docs/chbp_onboarding.md` Table 2.1: channel std,
   alpha peak, relative band powers and PSD against the TUH / HEEDB normal references, eyes
   closed and eyes open separately. CHBP lost 1–2 years of MAE to an amplitude / spectral
   shift, so check first whether TDBRAIN (DC–100 Hz, mastoid reference, short files) shifts too.
2. **Zero-shot brain age** of E2, E2L, W4, L1, W7, X1, X3 (and the device-robust E2L from
   `scripts/external_runs/e2l_robust_full.sh`), with the metrics of `docs/chbp_onboarding.md`
   Eqs. (3.1)–(3.6). Groups: 5–17 / 18–44 / 45–88 years, healthy, MDD, ADHD, OCD, SMC,
   eyes closed vs open. The 5–17 group lies outside the training ages (MIPDB showed the models
   fail on children), so the headline numbers are for adults.
3. **Brain-age gap by diagnosis.** Bias-corrected gap (`brain_age_bias_correction.md`; the
   correction is fitted on the training model's own validation predictions, not on TDBRAIN)
   for MDD, ADHD, OCD and SMC against the healthy controls, adults only, one session per
   subject. Subjects with several sessions are grouped (cross-fitted recalibration folds
   already group by `subject`).
4. **Within-subject repeatability**: participants with two or more sessions give a test–retest
   check of predicted age, net of the true age change between the sessions.
5. Optionally, add `tdbrain=<cohort>` to `evaluation.test_sets`. For SageMaker that needs an
   entry in `DEFAULT_SAGEMAKER_DATA_S3_URIS` once the cohort is on S3.

## 6. Checklist

| step | status |
|---|---|
| paper + DUA read, access routes checked | done 2026-11-12 |
| converter, clinical table, report groups, eval trim overrides, tests | done, synthetic data only |
| ORCID login + DUA on brainclinics.com | done 2026-11-12 (zip password received) |
| zips on the server (`zips/`), extracted, zip mirrored to `s3://…/TDBRAIN/v1/zips/` | done 2026-11-14 (release 3.1, Section 7) |
| real tree checked: BDF, tasks, channels, units; reader adapted (BDF, end pad); signal QC on 80 files | done 2026-11-14 (Section 7) |
| participants table (age, sex, indication): not in the zip | **waiting for the user** (separate download) |
| `convert` + cohort to S3 | after the participants table |
| full-cohort QC, zero-shot scoring, per-diagnosis gap, results note | after the conversion |

### Things to verify on the real files

The code reads the formats published in the paper. These details were not visible before
download:

- the exact `participants.tsv` column names. The reader accepts `participants_ID` /
  `participant_id`, `sessID` / `session`, `gender` / `sex`, `indication`, `formal_status`,
  `DISC/REP`, and stops with the list of columns found if the id or age column is missing;
- whether `participants.json` gives the gender levels (otherwise `sex` stays None; it is not guessed);
- the BrainVision unit and resolution (MNE scales them). Check that the channel std is in
  the expected 5–20 µV range;
- the indication vocabulary (combined labels, spelling) for the keyword map
  `TDBRAIN_INDICATIONS`;
- the zip layout. The EEG zip is **password protected**: Brainclinics e-mails the password after the DUA. It is kept in `/data/datasets/EEG-public/TDBRAIN/.zip_password` (mode 600, never in git or on S3) or `$TDBRAIN_ZIP_PASSWORD`. `tdbrain.sh unpack` extracts with 7z (`p7zip-full`), which handles ZipCrypto, AES, `.z01` and `.zip.001` splits and nested zips.

## 7. Release 3.1 as received (2026-11-14)

The download is `TDBRAIN_Dataset_V3_1_Encr.zip` (15.5 GB, stored, AES-encrypted entries).
It extracts to `raw/TDBRAIN_Dataset_V3_1/` (19.5 GB, 9,128 files, macOS `.DS_Store` files
skipped). This release differs from the paper's (Section 1) as follows:

### Table 7.1. Contents of release 3.1

| | |
|---|---|
| subjects / folders | 1,300 subjects (`sub-87*` 284, `sub-88*` 1,016); **no `sub-19*`**, so this release holds no blinded replication subjects |
| sessions | 1,391 `restEC` and 1,387 `restEO` session files, plus `oddball` in 162 sessions (with `events.tsv`; not resting, not converted) |
| format | **BDF** (24 bit) + `_eeg.json` + `_channels.tsv`, BIDS 1.9.0; no BrainVision, no `derivatives/` |
| channels | 26 EEG (10-10) + VPVA, VNVB, HPHL, HNHR (EOG), Erbs (ECG), Mass (EMG) + BDF `Status` (all zero); all 19 of the 10-20 channels map |
| recording | 500 Hz, "linked mastoids (A1A2)", power line 50 Hz, Neuroscan NuAmps, DC-coupled (high-pass 0, low-pass 100 Hz in the header), µV |
| length | 120.0–121.0 s per file, of which about 1 s at the end is a **constant pad** (digital zero) up to the next whole 1-s BDF record |
| markers | none (`Status` channel all zero, no BDF annotations) |
| demographics | **none in the zip**: no `participants.tsv`, and the BDF patient field and date are anonymised (`<id> X X X`, `01.01.85`). Age, sex and indication have to come from a separately downloaded participants table |

**End pad.** The signal sits at a DC offset of several mV, so the pad, a step to 0, would
ring through the band-pass into the last two windows. In 79 of 80 sampled files, the last
10 s window had a median 78 times the channel std of the others and the one before it 3 times.
`open_tdbrain` therefore cuts the samples where no 10-20 channel changes, at both ends, before
removing the mean. `tests/test_tdbrain.py` pads its synthetic files in the same way and fails
without the cut.

### Table 7.2. Signal QC after the training pipeline, 80 random resting files vs 60 normal recordings per reference

For window $k$ of a recording (10 s, the 19 EEG channels $c$, first and last window of a
TDBRAIN file left out; windows 6–35 of a reference recording), the channel std is
$s_{k,c} = \operatorname{std}_t x_{k,c}(t)$, and the table gives the median over windows,
channels and recordings (7.1). For the band shares, the per-window Welch PSD $P(f)$
(5 s segments, mean removed per window) is medianed over windows and channels, and
$\rho_{[a,b)} = \sum_{a \le f < b} P(f) / \sum_{0 < f < 45} P(f)$ (7.2). The alpha peak is the
argmax of the O1/O2 PSD in 6–14 Hz (7.3).

$$ \tilde s = \operatorname{median}_{k,c,\text{rec}}\ s_{k,c} \tag{7.1} $$

$$ \rho_{[a,b)} = \frac{\sum_{a \le f < b} P(f)}{\sum_{0 < f < 45} P(f)} \tag{7.2} $$

| source | n | channel std (µV) (7.1) | after CAR (µV) | ρ below 1 Hz (7.2) | ρ 1–4 Hz | ρ 8–13 Hz | alpha peak (Hz) (7.3) | share of samples > 300 µV |
|---|---|---|---|---|---|---|---|---|
| TUH normal (training) | 60 | 12.3 | 9.9 | 0.45 | 0.16 | 0.09 | 8.8 | 0.001 |
| HEEDB normal (training) | 60 | 22.7 | 12.6 | 0.62 | 0.15 | 0.06 | 8.6 | 0.056 |
| TDBRAIN eyes closed | 45 | 15.4 | 10.1 | 0.36 | 0.15 | 0.20 | 9.6 | 0.000 |
| TDBRAIN eyes open | 35 | 11.8 | 8.2 | 0.36 | 0.23 | 0.11 | 9.4 | 0.000 |

**What Table 7.2 shows:** unlike CHBP (`docs/chbp_onboarding.md` Table 2.1, std 5.9 µV),
TDBRAIN arrives at the models' input with a training-like amplitude. Its median channel std
of 15.4 / 11.8 µV lies between TUH and HEEDB. The DC-coupled recording does not leave more
slow power than the training data after the 0.1 Hz high-pass: the share below 1 Hz is 0.36
against 0.45 / 0.62. Eyes closed shows the expected alpha (ρ 0.20, peak 9.6 Hz), about twice
the eyes-open share, as the paper reports. No sampled file had a sample above 300 µV. After
the cut, the first and last windows match the inner ones: median std ratio 1.00 / 1.07, 95th
percentile 1.43 / 1.51. 72 of the 80 files give 12 windows and 8 give 11.
