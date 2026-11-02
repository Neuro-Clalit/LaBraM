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
