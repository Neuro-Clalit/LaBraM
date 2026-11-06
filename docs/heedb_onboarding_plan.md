# Plan: onboard the Harvard EEG Database (HEEDB v4.1) for brain age + pathology

Status: approved, 2026-11-01.

## Context
We have credentialed BDSP access to HEEDB v4.1 (granted 2026-11-01 to account 574441342949,
`AWS_PROFILE=mfa`). Today our brain-age / normal-abnormal models train on TUH only (TUAB ~3k
recordings, TUEG cohorts `all_both_v3*`). HEEDB is roughly 5–20× larger and, unlike TUH (where
NEDC no longer ships reports), it has **per-session report findings and per-patient ICD-10 /
medication histories**. That gives us more brain-age data, many more normal/abnormal labels, and
new clinical classes (seizure disorder, cerebrovascular disease, dementia/degeneration,
encephalopathy-like slowing, and others). Goal: (1) pull the main cohorts and run analytics,
(2) understand the categories well enough to choose cohorts that help our model, (3) mirror them
privately to S3 and preprocess them into the **same npy contract as TUH**, (4) train on HEEDB+TUH
together.

### What we verified (read-only, 2026-11-01)
Access point `s3://arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-credentialed-access-point/EEG/`:

| | |
|---|---|
| Layout | `bids/<site>/sub-<site><pid>/ses-<N>/eeg/sub-…_ses-N_task-{EEG,cEEG}_eeg.edf` + `channels.tsv`, `eeg.json`, `annotations.csv` (some sessions also have `_Xltek.csv`, `_PersystSpikes.csv`), `ses-N/…_scans.tsv` (acq_time) |
| Sites | S0001, S0002, I0002, I0003, I0004 (+I0004-SEEG), I0008, I0009, ELUCID, Neurotech, norse, teleEEG |
| Recording metadata `eeg-metadata/<site>_eeg_metadata_*.csv` | S0001 119.5k recordings / 46k patients / 974k h (rEEG 41.6k, cEEG 60k, EMU, OR; ServiceName Routine 47.9k, LTM 61.9k); S0002 72.6k / 32k / 605k h (rEEG 25.8k); I0002 75.8k / 1.05M h; I0003 61.3k / 681k h; I0008, I0009 small (DOB instead of age). **≈330k recordings, ≈3.3M hours in total** |
| Routine EEG | S0001 rEEG: median 65 min (p10 33, p90 97). One sample: 256 Hz, 50 channels (C3…T6, A1/A2, Fpz, LOC/ROC, ECG, chin, resp), µV, old 10-20 names (T3/T4/T5/T6), referential. About 92 MB per hour, so routine S0001+S0002 (~77k) ≈ **7 TB raw EDF**. cEEG files are 0.8–1.5 GB each, so all of HEEDB is on the order of 300 TB and is **not** something to mirror |
| Age / sex | `AgeAtVisit` per session (≥99.9% non-null at S0001/S0002/I0002/I0003); `HEEDB_patients.csv` 109k patients, age mean 42 ± 28, 31k under 18, 119 over 100, 2 negative (clean these). Sex M 55.8k / F 53.4k. `DateOfDeath` for 21.9k S0001 sessions |
| Report findings (per session; join on `SiteID, BDSPPatientID, SessionID`) | S0001 90.7k, S0002 39.1k, I0002 9.9k, I0003 37.5k rows → ≈177k labelled sessions. normal ≈48k vs abnormal ≈125k. ~30 findings flagged with provenance `report` / `annotation` / `verified`: spikes, gen slowing, foc slowing, seizure, status, lpd, gpd, lrda, grda, bs (burst suppression), low voltage, pdr, sleep stages, plus syndromes (bects, eses, jme, cjd, …) |
| ICD-10 neuro (per patient, `HEEDB_ICD10_for_Neurology.csv`, 108.7k rows) | Seizure disorders 30.2k, Headache 21.3k, Sleep 17.6k, Cerebral degeneration 15.1k, Cerebrovascular 11.5k, Movement 11.4k, Cerebral lobe dysfunction 11.1k, Behavioural/cognitive 5.8k, Tumours 4.8k, Developmental delay 4.0k, … |
| Medications (per patient, `HEEDB_Medication_ATC.csv`) | ATC codes grouped by first level; N03 (antiseizure), N05 (psycholeptics), N01/N02 (anaesthetics/opioids) give "sedated / on ASM" flags like `tuh_clinical.DRUG_CLASSES` |
| Patient IDs | `BDSPPatientID` is unique across sites (108.3k unique in 109.2k rows); BIDS folder = `sub-<SiteID><pid>` |

DUA: no redistribution and no re-identification. HEEDB is stored under `s3://eeg-data-public/HEEDB/v4.1/` — private despite the bucket's name (all public-access blocks on, no bucket policy, no public ACL); the copy tool refuses to run if it ever becomes publicly readable.
It must never be made public or shared outside this account (2026-11-02: the user chose this
bucket over a dedicated one; `infra/heedb_storage` remains as an optional KMS bucket).

### Data report (2026-11-01): what changed the plan
`notebooks/heedb_data_report.ipynb` (+ PDF) runs the TUH analytics on HEEDB from the metadata, all
72,118 adult-routine EDF headers (S3 range reads) and a 360-recording signal sample. The report
changes these parts of the plan:
- **Download set measured:** 72,115 adult routine sessions = 376,751 objects (EDF + sidecars),
  **5.48 TB** (`python -m dataset_maker.heedb_index download-list` → `scan/download_list.parquet`).
- **Flat signal is common:** S0001/S0002 routine files are flat (all-zero) for long stretches in
  their first half (93 % / 70 % of sample files flat ≥ 25 % of the time), so flat-window
  removal is **mandatory** at conversion time (not the opt-in window QC).
- **Channels:** 70 % of files lack A1/A2 and all lack T1/T2 (19 of 23 common), so train with
  missing-channel handling, as in the TUH `anymiss` cohorts. 68 % of files are already 200 Hz.
- **Mains:** a 60 Hz line in 83 % of files (stronger than TUH); the TUEG recipe (60 Hz spectrum
  interpolation) removes it.
- **Report findings are noisy:** negated mentions are counted (generalized slowing in 45 % and
  epileptiform activity in 15 % of *normal* reports). Use the normal/abnormal call and the rare,
  specific findings (GPD, LPD, GRDA, LRDA, burst suppression, focal slowing; ≈ 90 % or more abnormal)
  as targets; validate the common findings before they become heads.
- **Health records:** the ICD file is a neurology extract (no TBI or psychiatric codes beyond
  depression/ADHD); medications are lifetime. The normative cohort is "report-normal + no *known*
  neuro dx" = 7,632 sessions (the TUH sedation term would leave 511).
- **Split:** 1,057 patients appear at two sites, so group the patient split across sites.

### What we reuse (TUH pipeline)
`download_tuh.py` (chunk → S3 → verify → state) · `tuh_metadata.read_edf_signal_header` /
`recording_row` · `tuh_index.build_index` / `summarize` / `leakage_audit` · `tuh_clinical`
(`normative_mask` pattern) · `make_TUH_npy.preprocess_recording` / `cmd_convert` / `cmd_merge`
(S3 input, shards, float16, `manifest.json`) · `scripts/submit_batch_npy.py` (AWS Batch Spot
array, `infra/batch_dataprep`) · `labram.data.window_qc` · `tueg_patient_split.hash_split` ·
`make_cohort_dataset` (manifest + `age_metadata.json` + `age_split.json`) · `TUABAgeNpyLoader` via
`data.dataset=TUAB_AGE, data_format=npy` · `labram/features` `--kind cohort --stems_file` ·
`labram/eval/meta_analysis.py`, `age_plots.py` · `scripts/eval_age_by_cohort.py`.

---

## Step 1 — Pull the metadata + a pilot download, first analytics (≈1 week)

1. **Storage.** `s3://eeg-data-public/HEEDB/v4.1/{metadata,raw,scan,index,npy,cohorts}/` (private
   bucket, us-east-1, same region as BDSP). `dataset_maker/download_heedb.py` checks before every run
   that the bucket is still not publicly readable.
2. **Metadata mirror** (~160 MB): copy `HEEDB_Metadata/` and `eeg-metadata/` to
   `…/metadata/` and to `/data/datasets/HEEDB/v4.1/metadata/` (local copy kept off git).
3. **Recording index without downloading EDF** — new `dataset_maker/heedb_index.py`:
   - `list`: walk `bids/<site>/` with `s3 ls --recursive` (or the S3 Inventory of the access point if
     BDSP provides one) → one row per EDF (`site, subject, session, task, path, bytes`).
   - `headers`: read only the EDF header with an **S3 range GET** (`256 + 256·ns` bytes) through
     `read_edf_signal_header` adapted to a byte buffer → `sfreq, n_signals, labels, duration_sec,
     n_records`. ~330k range GETs, minutes on Batch, no data copied.
   - `join`: + `eeg-metadata` (age, sex, service, folder, duration, date of death) + report
     findings (session) + ICD-10 / ATC (patient) → `heedb_index.parquet`, same column names as
     `tuh_index` where they overlap (`stem, subject, session, age, sex, sfreq, n_signals,
     duration_sec, missing_channels, abnormal`) plus `site, task, service, finding_*, icd_*,
     atc_*, label_provenance`.
   - Map channels to the 23 `TUH_CHANNELS` with a BIDS-aware `channel_map` (bare names, both 10-20
     name sets: T7/T8/P7/P8 → T3/T4/T5/T6) and record `missing_channels`.
4. **Download: all adult routine sessions in one pass (≈72k sessions, ≈6.3 TB).** Adult =
   `AgeAtVisit` 18–89 (no paediatric cases in v1). Routine = ServiceName `Routine` / folder `rEEG`
   (duration proxy 15 min–2 h for I0002/I0003, which lack a service label), 10 min–4 h long.
   Counted 2026-11-01:

   | Site | Adult routine sessions | Patients | Hours | ≈ TB raw | With report label (normal / abnormal) |
   |---|---|---|---|---|---|
   | S0001 | 33,656 | 26,856 | 35.7k | 3.3 | 25,604 (10,818 / 14,767) |
   | S0002 | 25,273 | 20,119 | 22.4k | 2.1 | 10,451 (3,249 / 7,193) |
   | I0002 | 11,641 | 9,973 | 9.4k | 0.9 | from I0002 findings file |
   | I0003 | 1,605 | 1,280 | 1.2k | 0.1 | (site is mostly paediatric) |
   | **Total** | **≈72k** | **≈58k** | **≈69k** | **≈6.3** | |

   The whole set (≈72k) is copied S3→S3 into `…/raw/` in one resumable pass (same region, so
   no egress); the selection is written to `download_list.parquet` so the copy is reproducible.
   The notebook report (`notebooks/heedb_data_report.ipynb`) is built from the metadata and the
   EDF headers (range reads) and does not wait for the copy.
   - Later: paediatric routine sessions, then cEEG/LTM/EMU segments extracted directly.
5. **First analytics** — `notebooks/heedb_data_report.ipynb` (house style per
   `notebook-docs-style`; PDF export), driven by a new `labram/eval/heedb_analysis.py`:
   recordings/patients/hours per site × service, age and sex distributions, sessions per
   patient, sfreq / channel-set / duration mix, missing-age / bad-age rates, normal vs abnormal
   prevalence by age decade and site, coverage of findings / ICD / ATC.

**Exit criterion:** `heedb_index.parquet` covers every EDF; we know exactly how many routine
recordings with age + normal/abnormal exist, per site, and their size in TB.

## Step 2 — Category analytics: which data and labels can help our model (≈1 week)
In the same notebook (section 2) plus `labram/eval/heedb_analysis.py`:
1. **Label taxonomy & agreement.** Define target sets:
   - *Pathology (session):* `abnormal`; grouped findings — epileptiform (spikes, lpd, gpd, bects,
     eses), seizure/status, slowing (gen, foc, grda, lrda), burst suppression / low voltage, PDR
     abnormal. Prefer `verified` > `report` > `annotation`; measure how much they disagree.
   - *Clinical (patient):* ICD-10 categories (seizure disorders, cerebral degeneration,
     cerebrovascular, cerebral lobe dysfunction, behavioural/cognitive, developmental delay,
     tumours, headache as a "near-normal" control) and ATC flags (ASM, benzodiazepine,
     antipsychotic, sedation). Flag that ICD/ATC are **patient-level, not dated** — they leak future
     diagnoses into earlier sessions, so treat them as weak labels.
2. **Normative ("healthy") cohort for brain age**, mirroring `tuh_clinical.normative_mask`:
   routine EEG, report `normal`, no seizure/degeneration/cerebrovascular/tumour/lobe-dysfunction
   ICD, not on N03/N05 sedatives, age 18–89 (paediatric cases are deferred).
   Report counts by age decade vs TUH `normal_both_v3`.
3. **Domain shift vs TUH:** after a pilot npy conversion (Step 3.4 on the 2k pilot), compare
   per-channel band power, 1/f slope, amplitude and line-noise spectra, HEEDB vs TUAB
   (`labram/features --kind cohort`), per site. Tells us whether a per-site normalisation or
   site token is needed.
4. **Cheap signal checks before GPU time:**
   - Run our best existing TUH brain-age model (CV ensemble) **zero-shot** on the pilot →
     MAE per site / age decade / normal vs abnormal (does TUH transfer?).
   - Handcrafted-feature baselines (`labram/features/modeling.py`, ridge / HistGB; works unchanged
     on HEEDB npy because it uses the same 23-channel `STORED_CHANNELS` contract; pass `--out`) on the pilot for
     age and for each candidate class → which classes are learnable from EEG at all, and the
     brain-age gap per ICD category (is it larger for degeneration / cerebrovascular?).
5. **Decision table** → cohort definitions for v1: which sites, services (routine first; EMU/LTM
   as cropped segments later), which labels become heads, minimum label count per class.

**Exit criterion:** an agreed cohort spec (`heedb_routine_v1`, `heedb_normal_v1`, label list) with
counts and expected TB.

## Step 3 — Mirror to S3 + preprocess into the TUH npy contract (≈1–2 weeks, mostly compute)
1. **Downloader** — `dataset_maker/download_heedb.py`, generalising `download_tuh.py`: the source is
   the access point instead of NEDC rsync; chunk by `<site>/sub-*` blocks (~1k patients); copy
   S3→S3 with `s5cmd cp` (server-side, no local disk) of the cohort's EDF + sidecars; verify size and
   ETag; resumable `download_state.json`. Input is `download_list.parquet`: all ≈72k adult
   routine sessions (≈6.3 TB) in one pass. **No full cEEG mirror**: for LTM/EMU, later extract 20–30 min awake
   segments by EDF byte-range reads (records are fixed-size) straight to npy.
2. **Converter** — make `make_TUH_npy.py` corpus-agnostic rather than forking it:
   - pluggable `channel_map` (TUH regex vs BIDS bare names + `channels.tsv` type; old/new 10-20 names);
   - pluggable stem (`sub-S0001<pid>_ses-<N>_task-EEG`) and scan rows from `heedb_index.parquet`;
   - identical signal path: referential channels → 23 `TUH_CHANNELS` (zero-filled + `channel_mask`),
     0.1–75 Hz band-pass, line removal `interp` at 60 Hz (US mains, same as TUEG `interp50-60`),
     resample to 200 Hz, µV, float16, `[T, 23]`, whole 10 s windows, drop < 60 s;
   - write to `s3://eeg-data-public/HEEDB/v4.1/npy/<variant>/recordings/<stem>.npy` + `manifest.json`
     (`format_version: 2`, plus `corpus: heedb`, `site`).
3. **Run** on AWS Batch Spot (`scripts/submit_batch_npy.py --corpus heedb`, reading the private
   mirror); then `make_TUH_npy merge`, `labram.data.window_qc` → `window_stats.npz`,
   `scripts/verify_npy_format.py`.
4. **Pilot first**: the 2k pilot end-to-end (feeds Step 2.3/2.4), then all ≈72k adult routine sessions.
5. **Splits** — patient-level `hash_split` (new seed namespace `heedb`) into train/val **plus a
   fixed held-out HEEDB test set** (e.g. 10% of patients, stratified by site), saved as
   `heedb_patient_split.json`; assert zero patient overlap. HEEDB and TUH patients cannot overlap
   (different hospitals), so the TUAB-eval test set stays untouched.
6. **Cohort builds** with `make_cohort_dataset.py` extended by a HEEDB selector:
   `heedb_routine_v1` (all routine, age + abnormal + findings), `heedb_normal_v1` (normative),
   and `tuh_heedb_v1` (TUH `all_both_v3` + HEEDB routine in one manifest: symlinked/prefixed
   recordings, `age_metadata.json` with `label` + new multi-label fields, `age_split.json`).

Storage estimate: raw ≈6.3 TB (≈ $145/month Standard, less on Intelligent-Tiering).
npy float16 at 23 ch × 200 Hz ≈ 33 MB/h → ≈2.3 TB for the ≈69k hours (≈0.8 TB if trimmed to 30 min per session).

### Status 2026-11-02: data ready, transfer baseline (E0)
- **Raw copy:** 376,751 objects (5.475 TB) in `s3://eeg-data-public/HEEDB/v4.1/raw/`, verified per site.
- **Conversion** (`processed_npy_interp50-60_flat0.5_float16`, Batch + recovery pass): 72,051 recordings,
  16.27 M windows, 45,181 h kept, 1.50 TB. Flat windows dropped: I0002 7.8 %, I0003 16.9 %,
  S0001 31.9 %, S0002 49.4 %. Not converted:
  - 46 recordings with known errors (44 with fewer than 19 channels, 2 entirely flat);
  - 20 with an invalid or short header;
  - 44 sampled at 2-16 kHz (`scan/excluded_by_sfreq.csv`).
- **Patient split:** `index/heedb_patient_split.json`, 101,812 / 12,770 / 12,723 patients.
- **Cohorts in S3** (`HEEDB/v4.1/cohorts/`):
  - `heedb_routine_v1`: 71,805 recordings;
  - `heedb_normal_v1`: 7,619 recordings (6,017 / 803 / 799).
- **E0, transfer:** TUH brain-age models scored on 2,100 held-out HEEDB test recordings (700 normal /
  700 abnormal / 700 unlabelled), `docs/data/heedb_transfer_e0.csv`:

  | Model | TUH MAE | HEEDB MAE | Bias | Slope (prediction on age) | MAE after recalibration |
  |---|---|---|---|---|---|
  | `exp4base` (EEG only) | ≈ 7.4 | 11.20 | −6.1 | 0.48 | 9.10 |
  | `exp4anymiss-noear-nolabel` (no ear channels) | ≈ 7.5 | **10.02** | −3.9 | 0.52 | **8.72** |
  | `cvaugval` 5-fold median (with condition vectors) | 7.1–7.4 | 10.18 | −4.0 | 0.53 | 8.94 |

  The error is mostly compression of predictions toward the mean age, and HEEDB is older than TUH:
  people in their 70s and 80s are underpredicted by 14–19 years. A linear recalibration fitted on
  half the patients and scored on the other half (by patient) recovers about 1.3–2.1 years. The
  model without ear channels transfers best (HEEDB lacks A1/A2 in 70 % of files). The
  recalibrated 8.7 is the bar that E1/E2 (training on HEEDB) must beat.

### Training on SageMaker: what works (2026-11-02)
- **Do not stream a full HEEDB cohort with FastFile.** Random 10-second windows over S3 made E1
  I/O-bound: 6.5 s per step, about 24 h per epoch. Both FastFile jobs were stopped.
- **Use compact cohorts in File mode.** `make_compact_cohort.py --src s3://… --dst s3://… --processes 12`
  keeps only the windows a run reads (at most 20 min per training recording, plus the 5 min
  evaluation budget):

  | Compact cohort | Size | Recordings | Instance |
  |---|---|---|---|
  | `heedb_routine_v1_compact` | 669 GB | 71,805 | ml.g5.8xlarge (900 GB NVMe) |
  | `tuh_heedb_v1_compact` | 894 GB | 103,117 | ml.g5.16xlarge (1.9 TB NVMe) |

  An E1 epoch takes about 45-50 min (13.4k steps). Account limits: one instance of each g5
  size for training, and at most 1,024 GB EBS per training instance (`sagemaker.volume_size_gb`).
- **Window QC for all 72k recordings** is in `processed_npy_…/window_stats.npz`: 16.27 M windows,
  10.26 % above 800 µV (TUH 12.1 %), 0.32 % with a flat channel.
- **Runs:**
  - E1 `labram-heedb-e1-routine-noear-2026-11-02-16-20-24-821`
  - E2 `labram-heedb-e2-tuh-heedb-noear-2026-11-02-17-04-13-850`

  Both use the recipe of `exp4anymiss-noear-nolabel` (19 channels, no condition input, mixup, lr 2e-4,
  batch 128, 5 min per recording per epoch, 8 epochs with early stopping). ClearML project `LaBraM/heedb`.

### Status 2026-11-03: first E1/E2 results, checkpoints lost, reruns
Both jobs completed, but they were submitted with the default `output.output_dir`
(`./checkpoints/finetune_tuab_age/`), which lies outside `/opt/ml/model`. The checkpoints stayed in
the container and were deleted with it, so there is no `model.tar.gz` and no TUAB-eval score.
`sagemaker_entry.py` now moves any output, log or CV directory outside the uploaded folders into
the model directory (commit `4ac80c4`). Case MAE in years at the best epoch, from the job logs:

| Run | Train data | Best epoch | Val MAE (normal / abnormal) | Test MAE (normal / abnormal) | Test set |
|---|---|---|---|---|---|
| E0 `exp4anymiss-noear-nolabel` | TUH | – | – | 10.02 (9.86 / 10.64), recalibrated 8.72 | 2,100-recording HEEDB subset |
| E1 | `heedb_routine_v1` | 8 of 8 | 7.52 (7.44 / 8.14) | **7.30** (7.48 / 7.79) | HEEDB test, 7,338 recordings |
| E2 | `tuh_heedb_v1` | 5 of 8 (early stop) | 7.36 (6.97 / 7.85) | **7.13** (7.16 / 7.62) | HEEDB test + TUAB eval, 7,612 recordings |

E1 was still improving at its last epoch (val 7.67 → 7.61 → 7.52 over epochs 6-8). Both already
beat the recalibrated transfer baseline (8.72), but E2's test mixes TUAB eval with HEEDB, so the
TUAB-eval comparison with `exp4anymiss-noear-nolabel` (7.48) needs the reruns' checkpoints:
- E1 rerun `labram-heedb-e1-routine-noear-r2-2026-11-03-05-34-25-665`
- E2 rerun `labram-heedb-e2-tuh-heedb-noear-r2-2026-11-03-05-59-02-058`

Same recipe; they also log the per-dataset (`harvard_*` / `temple_*`) and `train_eval` metrics.
Their scores on TUAB eval and on the 2,100-recording HEEDB subset (per site, per age decade,
2-fold patient recalibration: `labram/eval/heedb_analysis.py::brain_age_tables`, which reproduces
`docs/data/heedb_transfer_e0.csv`) go to `docs/data/tuab_eval_heedb_comparison.csv` and
`docs/data/heedb_e1_e2.csv`.

### E1 results (rerun, 2026-11-03)
The E1 rerun completed with its checkpoint: best epoch 6 of 8, val 7.50 / HEEDB test 7.27
(7,338 recordings); the first run ended at 7.52 / 7.30. The checkpoint is scored with
`scripts/eval_age_by_cohort.py --pred-stats-from-log` (predictions de-normalised with E1's own
training statistics, mean 53.92, std 19.82 years). Case MAE, years:

| Test set | Model | All | Normal | Abnormal | Unlabelled |
|---|---|---|---|---|---|
| TUAB eval (274) | TUH-only `exp4anymiss-noear-nolabel` | 7.48 | 5.75 | 9.54 | – |
| TUAB eval (274) | best TUH single model (`cvaugval_f4`) | 7.10 | 5.11 | 9.48 | – |
| TUAB eval (274) | **E1 (HEEDB only)** | **6.90** | 5.83 | **8.18** | – |
| HEEDB subset (2,100) | E0 `exp4anymiss-noear-nolabel` | 10.02 | 9.86 | 10.64 | 9.57 |
| HEEDB subset (2,100) | E0, recalibrated (2-fold by patient) | 8.72 | 8.77 | 9.17 | 8.20 |
| HEEDB subset (2,100) | **E1** | **7.37** | 7.44 | 7.85 | 6.82 |
| HEEDB subset (2,100) | E1, recalibrated | 6.89 | 6.91 | 7.33 | 6.42 |

- **Never trained on TUH, E1 is the best model on TUAB eval so far** (6.90 vs 7.10–7.48 for the TUH
  models), with the gain on abnormal recordings (8.18 vs about 9.4–9.6). Its bias on TUAB eval is +1.3
  years (+2.1 on normal), against −0.6 to −1.8 for the TUH models.
- **On HEEDB,** E1 reduces MAE by 2.7 years against the transfer baseline and by 1.4 against the
  recalibrated baseline. Bias falls from −3.9 to −0.7 years and the prediction-on-age slope rises
  from 0.52 to 0.67, so there is less compression toward the mean age.
- **Per site** (E0 → E1): I0002 9.73 → 7.15, S0001 10.44 → 7.81, S0002 9.40 → 6.81, I0003 (n = 37)
  10.72 → 5.82. S0001 remains the hardest site.
- **Per age decade, the elderly remain the weak point.** 70s: bias −14.1 → −8.1, MAE 14.1 → 8.4;
  80+: bias −19.6 → −13.1, MAE 19.6 → 13.1. The 18–29 group is overpredicted (+7.9 to +8.3). The
  remaining slope (0.67) is the usual regression-to-the-mean of brain-age models; age-bin
  sampling or loss balancing (`data.age_bin_sampling`, `loss.balance`) are the candidate fixes.

### E2 results and the E0 / E1 / E2 comparison (2026-11-03)
The E2 rerun (TUH `all_both_v3_anymiss` + HEEDB routine, 103,117 recordings) completed with its
checkpoint. The best epoch is 8 of 8 (val 7.43; epoch 5 had 7.44), with HEEDB+TUAB test 7.25. The
offline TUAB-eval score equals the in-training `temple_*` numbers (6.70 / 5.13 / 8.56), so the two
evaluation paths agree. `E1+E2 mean` averages the two models' per-recording predictions. Case MAE,
years (`docs/data/tuab_eval_heedb_comparison.csv`, `docs/data/heedb_e1_e2.csv`):

| TUAB eval (274) | All | Normal | Abnormal | Bias | R² |
|---|---|---|---|---|---|
| TUH-only `exp4anymiss-noear-nolabel` | 7.48 | 5.75 | 9.54 | −0.64 | 0.71 |
| best TUH single model `cvaugval_f4` | 7.10 | **5.11** | 9.48 | −0.36 | 0.72 |
| TUH 5-fold ensemble `cvaugval_ens` | 7.22 | 5.36 | 9.44 | −1.17 | 0.72 |
| E1 (HEEDB only) | 6.90 | 5.83 | 8.18 | +1.29 | 0.76 |
| **E2 (TUH + HEEDB)** | **6.70** | 5.13 | 8.56 | −0.22 | 0.76 |
| E1+E2 mean | **6.60** | 5.34 | **8.09** | +0.54 | 0.77 |

| HEEDB subset (2,100) | All | Normal | Abnormal | Unlabelled | Recalibrated (all) | Slope |
|---|---|---|---|---|---|---|
| E0 `exp4anymiss-noear-nolabel` | 10.02 | 9.86 | 10.64 | 9.57 | 8.72 | 0.52 |
| E1 | 7.37 | 7.44 | 7.85 | 6.82 | 6.89 | 0.67 |
| E2 | 7.42 | 7.43 | 8.02 | 6.79 | 6.80 | 0.66 |
| E1+E2 mean | **7.36** | **7.40** | 7.90 | **6.78** | 6.80 | 0.67 |

- **TUAB eval:** E2 is the best single model (6.70; 0.40 better than the best TUH model, 0.78 better
  than the same recipe trained on TUH only) and almost unbiased (−0.2). It keeps the TUH models'
  normal-recording accuracy (5.13 vs 5.11) and improves abnormal recordings by about 0.9 years.
  E1 is better on abnormal recordings (8.18) but over-predicts normal ones (+2.1 bias, 5.83). The two
  are complementary: their mean gives 6.60 overall.
- **HEEDB:** E1 and E2 are equivalent (7.37 vs 7.42; per site within ±0.1 except I0003, n = 37).
  Adding TUH costs nothing on Harvard data. The old-age bias is the same in both (70s −8.1/−8.3,
  80+ −13.1/−13.3).
- **Per epoch** (`docs/figures/heedb_e1_e2_epochs.png`):
  - (a) TUAB eval: E2 is below the TUH-only run from epoch 1 (7.90 vs 8.79) and at every later epoch.
  - (b) The gain is mostly on abnormal recordings.
  - (c) On the shared TUH val split, E2 reaches 7.31 vs the TUH-only run's best 7.81, so HEEDB also
    helps on TUH data.
  - (d) Train-eval vs val per dataset.
- **Bias correction** (`docs/brain_age_bias_correction.md`):
  - Corrections are fitted on val normal recordings: Cole, linear (de Lange & Cole) and the
    quadratic gap of Gemein et al. 2024.
  - After correction the age trend is gone. E2's corrected TUAB normal gap MAE is 4.24
    (SOTA paper 6.60).
  - Abnormal recordings show a brain-age gap 2.4–4.1 years above normal ones in every model
    (AUROC 0.62–0.67). Raw, the gap shows no difference, because of the age confound.
- **Over-fitting:** E2's train-eval MAE fell to 5.37 (Temple 4.75 at epoch 7) while val stayed
  at 7.43–7.48 from epoch 5. The TUH part is memorised faster than HEEDB.
- **Next:**
  - **E3:** the E2 recipe plus the abnormal head (normal/abnormal AUROC/BAcc on TUAB eval).
  - **Old-age compression:** age-bin sampling or loss balancing.
  - **Checkpoint selection:** pick the epoch on a dataset-balanced val metric.
  - **Ensembles:** E1+E2 (or several seeds) as an ensemble for reporting.

### E2c: E2 + clinical condition inputs, trained locally (2026-11-04)
- **What:** the E2 recipe and split (`tuh_heedb_v1`, same training-target statistics 53.52 / 19.29)
  with `model.session_features=head`.
  - The input is the per-recording condition vector `conditions_v1.npz`: report status
    (normal / abnormal / not available) plus 14 pathology flags.
  - TUH flags come from the NEDC sheet; HEEDB's `dx_*` flags come from patient ICD-10 codes
    (`make_condition_features.py --extra-clinical`).
  - The vector goes through Linear(17→32) and is concatenated with the CLS feature before the
    regressor.
- **Where:** trained on this machine (A10G, `/data/runs/heedb/run_e2c.sh`), in 21.9 h. Loading
  was data-bound: 6 loader workers against random reads from gp3, GPU idle about 60%.
  `data.num_workers=20` is advised for local runs.

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Val E2c | 8.96 | 8.23 | 7.78 | 7.85 | 7.45 | 7.46 | 7.45 | **7.40** |
| Val E2 | 8.89 | 8.36 | 8.07 | 7.92 | 7.44 | 7.48 | 7.48 | **7.43** |
| Train-eval Harvard / Temple, E2c | 8.80 / 7.90 | 7.78 / 7.07 | 6.97 / 6.42 | 7.12 / 5.96 | 6.43 / 5.20 | 6.29 / 5.05 | 6.20 / 4.79 | 6.14 / 4.69 |
| HEEDB test E2c (E2) | 9.08 (8.93) | 8.18 (8.28) | 7.67 (8.02) | 7.82 (7.87) | 7.32 (7.24) | 7.36 (7.34) | 7.32 (7.30) | 7.28 (7.27) |
| TUAB eval E2c (E2) | 8.11 (7.90) | 7.30 (7.33) | 7.07 (7.23) | 7.02 (6.97) | 6.79 (6.86) | 6.87 (6.79) | 6.83 (6.67) | 6.82 (6.70) |

Final checkpoint (epoch 8), scored offline:

| Test set | E2c | E2 |
|---|---|---|
| TUAB eval: all / normal / abnormal | 6.82 / 5.20 / 8.76 | 6.70 / 5.13 / 8.56 |
| HEEDB full test (7,338) | 7.29 | 7.27 |
| HEEDB 2,100 subset | 7.35 | 7.42 |

- **Accuracy:** the clinical inputs speed up early training (−0.3 years at epoch 3) but do not
  improve the final age accuracy (within ±0.1 year of E2).
- **Recommendation:** E2 stays the reference model. It is EEG-only and needs no clinical labels at
  inference.
- **Gap separation:** with the quadratic bias correction, E2c separates abnormal from normal
  recordings slightly more (`docs/brain_age_bias_correction.md`). E2c receives the abnormal label
  as input, so that separation is not EEG evidence.

### E2L: E2 trained 16 epochs, locally (2026-11-04 → 11-06)
- **Recipe.** E2 (TUH + HEEDB, same split, EEG-only), 16 epochs with 4 warm-up epochs, 20 → 12 loader
  workers, on the local A10G (`/data/runs/heedb/run_e2l.sh`, then `resume_e2l.sh` / `resume_e2l_bs256.sh`).
  - Epochs 1–6 ran at batch 512 / lr 4e-4. Throughput was no higher than at 256 (the GPU saturates at 256;
    `scripts/benchmark_finetune_speed.py`), and val jumped from 7.67 to 8.41 at epoch 6.
  - From epoch 7: batch 256 / lr 2.8e-4 on the same cosine schedule, resumed from the epoch-6 checkpoint.
- **Speedups found while running it:**
  - `pread` loader (`NPY_READ`): exact window reads instead of memory-map faults, read amplification
    4.4× → 1.06×; the g5.2xlarge's 106 MB/s EBS baseline was the limit.
  - Linear-time window pooling in evaluation (`aggregate_windows`): ≈ 2.5 min → 0.16 s per call.
  - `evaluation.train_eval_every=4`.
  - Result: epoch time 2h50m → 68 min (60.5 training + 7 evaluation), against 89 min for E2 on SageMaker.
- **Result** (best epoch 12, val 7.24; epochs 13–16 at 7.30–7.41 as the learning rate reached 0):

| | E2L | E2 | E1 | TUH only (`Exp4noearEEG`) |
|---|---|---|---|---|
| TUAB eval all / normal / abnormal | **6.55 / 4.91 / 8.50** | 6.70 / 5.13 / 8.56 | 6.90 / 5.83 / 8.18 | 7.48 / 5.75 / 9.54 |
| HEEDB test (7,338) | **7.02** | 7.27 | 7.27 | 9.82 |
| Train-eval Harvard / Temple at the last epoch | 5.30 / 3.66 | 6.05 / 4.69 | 6.14 / – | – |

- **Over-fitting:** E2L over-fits more (Temple train-eval 3.66 vs val about 7.3), so the epoch is picked on val.
- **Significance:** the gain over E2 (0.15 on TUAB, 0.25 on HEEDB) is below the 0.38-year 2-run noise threshold
  of the brain-age summary, i.e. a trend.
- **Summary notebook:** `notebooks/brain_age_research_summary.ipynb`, phase 4: Tables 2.11–2.13 and
  Figures 2.2–2.3, with the age error per pathology.

### Per-dataset reporting (from the runs after E1/E2)
Every fine-tune now reports each split overall, per cohort (normal / abnormal / unlabelled)
and per dataset x cohort (`harvard_*`, `temple_*`, e.g. `temple_abnormal_mae`). A
`train_eval` split (up to 500 training recordings per dataset, scored like val/test) shows
overfitting per dataset. E1/E2 were launched before this change. Their per-dataset numbers
come from `scripts/eval_age_by_cohort.py`, which reports the same groups.

### Health-record analytics and E2P, the pathology classifier (2026-11-05)
- **Health-record fix.** `heedb_index.health_record`: a patient has an ICD-10 record when listed in the
  neurology ICD file or `ICD10Count > 0`. Before, only `ICD10Count > 0` counted, but it is 0 or missing for many
  coded patients (all of I0008/I0009, 15 % of S0001).
  - `dx_*` are now known for 90 % of adult routine recordings instead of 43 %.
  - Positives grew by about 50 %: epilepsy 15.4K → 23.1K, stroke 5.5K → 7.4K.
  - `heedb_index.parquet` and `heedb_routine_v1/labels.parquet` were regenerated (backups `*.bak_20261105`).
  - The E2c condition vectors were built before the fix.
- **Analytics.** `notebooks/clinical_records_analytics.ipynb` (+ PDF) covers TUAB, the TUH cohort and HEEDB:
  - age × sex × report verdict;
  - 21 pathology categories (headache/migraine, sleep, tumour and others), by age group and sex, with the
    abnormal-EEG excess and co-occurrence;
  - report abnormality types and their lift per pathology.
- **E2P** (`/data/runs/heedb/submit_e2p.sh`, SageMaker `labram-heedb-e2p-pathology-2026-11-05-19-51-36-294`).
  - **Recipe:** E2L's scenario on the same split (16 epochs / 4 warm-up, batch 256, lr 2.8e-4, patience 5) as a
    status + 17-group classifier without age prediction. This is `main`'s Exp4cls recipe: multi-task head,
    class-balanced pathology loss, `loss.age_weight=0`, epoch selected on pathology macro ROC-AUC.
  - **Targets:** `multitask_targets_v2.npz`, the 13 NEDC groups + headache / sleep / tumour / movement disorders.
  - **Hardware:** one ml.g5.12xlarge with 4-GPU DDP (global batch 256 = 4 × 64), the first real-GPU use of
    `distributed.nproc_per_node`.

### Local training on all data (decided 2026-11-02)
The user wants to be able to train locally on all of TUH + HEEDB, not only on SageMaker.
- **Disk:** `/data` (`vol-0e556358439755c10`, gp3, 500 MB/s, 6000 IOPS) is 1.5 TB with 1.465 TB used.
  The HEEDB npy is ≈ 1.66 TB: 68.7k header hours × 73 % kept after flat removal × 33.1 MB/h,
  measured on the pilot (45.2 GB for 1,364 kept hours). Cohorts hard-link their recordings, so they
  add ≈ 0. **Grow to 4 TB** (≈ 78 % used); use 5 TB if paediatric or cEEG excerpts follow. Then
  `sudo resize2fs /dev/nvme1n1` (ext4, online).
- **Steps after the resize:**
  1. `s5cmd sync s3://eeg-data-public/HEEDB/v4.1/processed_npy_interp50-60_flat0.5_float16/* /data/datasets/HEEDB/v4.1/processed_npy_interp50-60_flat0.5_float16/`
  2. `python -m labram.data.window_qc --root <that dir>` writes `window_stats.npz`.
  3. `python -m dataset_maker.make_heedb_cohort --experiment routine|normal --source <local npy> --out /data/datasets/HEEDB/v4.1/cohorts/<name>` (hard links).
  4. Train with `data.data_path=<local cohort>`; NVMe staging (`data.stage_dir=auto`) copies what fits and
     reads the rest from `/data`.
- **Done 2026-11-03.** `/data` was grown to 4 TiB (gp3, 6000 IOPS, 500 MB/s); `resize2fs` online.
  - The npy sync: 72,051 recordings, 1,496,393,638,528 bytes, identical to S3 (count and bytes),
    `window_stats.npz` included (step 2 not needed).
  - Local cohorts, all hard links (`/data` stays at 2.8 TB used, 1.1 TB free). Each `age_split.json`
    is byte-identical to its S3 cohort, so local and SageMaker runs see the same split:

    | Cohort | Recordings | Local path |
    |---|---|---|
    | `heedb_routine_v1` (E1) | 71,805 | `/data/datasets/HEEDB/v4.1/cohorts/heedb_routine_v1` |
    | `heedb_normal_v1` | 7,619 | `/data/datasets/HEEDB/v4.1/cohorts/heedb_normal_v1` |
    | `tuh_heedb_v1` (E2, TUH + HEEDB) | 103,117 | `/data/datasets/HEEDB/v4.1/cohorts/tuh_heedb_v1` |

    The joint cohort comes from `make_joint_cohort.py` with a local `--out` (hard links from the
    local part cohorts).

## Step 4 — Train on HEEDB + TUH (≈2 weeks)
1. **Code support**
   - **Combined cohort = one S3 prefix.** SageMaker gets one `dataset` channel per job
     (`submit_sagemaker.stage_s3_inputs`), and mixing is offline today (`make_cohort_dataset`). So
     `tuh_heedb_v1` is built in the **private** bucket: HEEDB npy + server-side copies of the TUH
     `all_both_v3` recordings (or `make_compact_cohort.py` to keep only the windows used), with one
     `manifest.json` carrying a `corpus` column. Access-point ARNs fail `validate_s3_uri`; that
     doesn't matter, because training reads only from our private bucket.
   - **Per-source sampling weights** (extend `labram/data/samplers.py`, `train_minutes_per_epoch`)
     so HEEDB does not swamp TUH; config `data.source_weights={tuh: 1, heedb: 1}`.
   - **Multi-label targets.** `age_metadata.json` is parsed strictly as `RecordingMetadata(**fields)`,
     so extra keys would break it. Put the new labels in a separate `labels.parquet` sidecar
     (stem → abnormal, finding groups, ICD groups, ATC flags; NA = unknown) loaded like
     `session_features.load_session_features`.
   - **Masked multi-label head + loss**: generalise `MultiTaskHead` (`labram/layers/mlp_head.py`) and
     `MultiTaskAgeCriterion` (`labram/losses/regression.py`) from `[age, abnormal]` to
     `[age, abnormal, findings…, icd…]`, with BCE masked where a label is NA (TUH rows only have age +
     abnormal). There is no multi-label path today (`build_classification_criterion` and
     `utils/eval_metrics.py` are binary/multiclass only), so add per-head AUROC / PR-AUC / BAcc.
     Age MAE stays the selection metric. `FinetuneModelConfig.validate` must allow it.
   - Optional `site` embedding (like `abnormal_embed` in `condition_on_abnormal`) if Step 2.3
     shows a strong site shift.
   - Relax the hard-coded `TUAB_AGE` checks (`window_selection.py`, `age_sampling.py`,
     `run_finetune.py`) or register `data.dataset=EEG_MULTI` in `bundles.py`.
   - Eval: `scripts/eval_cv_folds.py` hardcodes the TUH index for its `source` column; read
     `corpus`/`site` from the cohort manifest instead. `scripts/eval_age_by_cohort.py` already scores
     any run on another cohort (`--data-path`, `--train-data-path`).
   - SageMaker: grant the execution role read access to the private bucket + KMS key; FastFile mode.
   - Pre-training (E5 only): the npy format cannot feed `build_pretraining_dataset` (HDF5
     `ShockDataset`); add a small npy→`[C, window]` pretraining Dataset that reads channels from
     the manifest.
2. **Experiment ladder** (`scripts/submit_heedb_experiments.sh`, ClearML project `LaBraM/heedb`;
   analyse with the `analyze-experiments` skill, leakage-free shared eval sets):
   - E0: current best TUH model evaluated on HEEDB test (transfer baseline; from Step 2.4).
   - E1: brain age on `heedb_normal_v1` alone → HEEDB test + TUAB eval.
   - E2: brain age on `tuh_heedb_v1` (TUH + HEEDB) → does TUAB-eval MAE improve over `all_both_v3`?
   - E3: + abnormal head (MT1-style) → TUAB normal/abnormal BAcc/AUROC vs LaBraM-base 0.814/0.902.
   - E4: + finding-group heads, then E4b + ICD-group heads (masked, down-weighted) → per-class
     AUROC; check that the ICD heads don't hurt age MAE or abnormal AUROC; brain-age gap per category.
   - E5 (optional): continue masked pre-training on HEEDB routine + cEEG segments
     (`labram.runs.pretrain`, frozen VQNSP), then repeat E2/E3.
   - Report every run on three fixed tests: TUAB eval, TUH `all_both_v3` test, HEEDB test (per site).
3. **Docs**: `docs/heedb_onboarding.md` (sources, DUA rules, index schema, cohorts, results),
   CLAUDE.md summary, and a memory update.

---

## Files to add or change (main ones)
- New: `dataset_maker/heedb_index.py`, `dataset_maker/download_heedb.py`,
  `labram/eval/heedb_analysis.py`, `notebooks/heedb_data_report.ipynb`,
  `scripts/submit_heedb_experiments.sh`, `docs/heedb_onboarding.md`, `infra/` private-bucket stack,
  tests `tests/test_heedb_index.py`, `tests/test_bids_channel_map.py` (synthetic EDF/BIDS).
- Change: `labram/data/tuh_metadata.py` (header from bytes; BIDS channel map),
  `dataset_maker/make_TUH_npy.py` (pluggable corpus), `scripts/submit_batch_npy.py` (`--corpus heedb`),
  `dataset_maker/make_cohort_dataset.py` (HEEDB selector, multi-label sidecar),
  `labram/data/samplers.py`, `labram/models` `MultiTaskHead`, `labram/losses` multitask criterion,
  `labram/data/bundles.py` / `window_selection.py`.

## Verification
- Unit tests on synthetic BIDS-style EDF: header range-read equals full read; channel map
  T7→T3 etc.; npy from a HEEDB-style EDF matches the TUH path on the same signal; masked
  multitask loss ignores missing labels. `pytest tests/ -v`.
- Data checks: index row count = `s3 ls` EDF count; `verify_npy_format.py` on the pilot; patient
  disjointness asserted on save/load; DUA check that the destination bucket stays
  non-public (`download_heedb.check_private`) and that no notebook output holds identifiers.
- End-to-end debug: pilot cohort → 1-epoch fine-tune on 1 GPU → eval report, before any full run.

## Decisions
- **Raw mirror (decided):** routine EEG only, in the private bucket. cEEG/LTM/EMU are never
  mirrored; segments are extracted by byte-range reads into npy.
- **Adults only, one pass (decided):** v1 is age 18–89, all ≈72k adult routine sessions
  (≈6.3 TB) downloaded at once. Paediatric data comes later. The Step 2 normative cohort and every Step 3/4 cohort use this adult filter.
- **v1 targets (decided):** age + abnormal + grouped report findings **+ ICD-10 group heads**
  (seizure disorders, cerebral degeneration, cerebrovascular, cerebral lobe dysfunction,
  behavioural/cognitive, developmental delay, tumours; headache as a near-normal control).
  ICD labels are per patient and undated, so: (a) use them only on sessions of patients with at least
  one ICD record (`ICD10Count > 0`), so "no code" is not read as "healthy" for patients with no
  history (mask instead); (b) give the ICD heads a lower loss weight than the findings heads; (c) report them
  separately as weak-label heads. Step 2 confirms which ICD groups are learnable before they
  become heads.
- **Storage (decided 2026-11-02):** `s3://eeg-data-public/HEEDB/v4.1/` (private bucket).
