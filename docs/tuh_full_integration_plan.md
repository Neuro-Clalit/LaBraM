# Plan: full TUH corpus for brain age + pathology

Status: draft, 2026-10-09. Goal: train and evaluate **brain-age regression** and
**pathology classification** on the whole TUH EEG family, not just TUAB.

## Ground truth as of today

| | |
|---|---|
| Local TUAB | `/data/datasets/EEG-public/TAUB/TUH_Abnormal/v3.0.0` (EDF 60 GB, pickles 141 GB, npy 71 GB) |
| S3 | `s3://eeg-data-public/{TUH_Abnormal,models}/` (account 574441342949) |
| `/data` free | **141 GB**. Too small to hold TUEG EDF, which is roughly 1.6–1.7 TB (check this with an rsync dry run) |
| TUH server | `www.isip.piconepress.com:22` can be reached from this box. **No NEDC credentials are configured here**; TUAB was rsynced from a laptop (`labram/data/upload_taub.py`) |
| Labels | Age and sex come from the EDF header (`labram/data/tuh_metadata.py`) for **every** TUH recording. TUH no longer ships reports, so pathology labels exist **only in the sub-corpora** |

What each corpus contributes. All of them are subsets of TUEG and share subject IDs:

| Corpus | Age/sex | Pathology label | Granularity |
|---|---|---|---|
| TUEG (whole) | ✓ | – | – |
| TUAB | ✓ | normal / abnormal | session |
| TUEP | ✓ | epilepsy / no-epilepsy | subject |
| TUSZ | ✓ | seizure events (+ type) | time-annotated |
| TUSL | ✓ | slowing vs seizure vs background | time-annotated |
| TUEV | ✓ | 6 event classes | time-annotated |
| TUAR | ✓ | artifacts (use for QC, not as targets) | time-annotated |

## Steps

### 1. Download, both locally and to S3
- Get NEDC rsync credentials (request form on the TUH downloads page). This blocks everything else.
- First, `rsync -n --stats` per corpus to get exact sizes and file counts.
- With 141 GB free, process **in chunks**: rsync one TUEG top-level dir → `s5cmd cp` to
  `s3://eeg-data-public/TUH_EEG/<corpus>/<version>/` → verify (count, bytes, ETag sample; reuse
  `scripts/upload_tuab_to_s3.sh`) → delete the local EDF. Keep the sub-corpora (TUEP/TUSZ/TUSL/TUEV,
  small) local as well.
- Deliverable: `scripts/download_tuh.sh <corpus>`. It is resumable, chunked, and writes a per-corpus `download_manifest.json`.

### 2. Analyze the data
- Extend `make_TUAB_age scan` into a generic `tuh_scan` that, for every EDF, records subject, session, montage
  (`01_tcp_ar`/`02_tcp_le`/`03_tcp_ar_a`/`04_tcp_le_a`), sfreq, duration, channel set, age, and sex → one
  `recordings.parquet` per corpus.
- Join the sub-corpus labels on `(subject, session, token)`.
- Report and figures (`labram/eval/age_plots.py` style): age histogram per corpus and per label, sex balance,
  normal/abnormal/epilepsy prevalence by age decade, recordings per subject, montage/sfreq/duration
  mix, and the `Age:999`/missing-age rate.
- **Leakage audit**: overlap of subjects across corpora, and in particular TUAB-eval subjects that appear in TUEG.

### 3. Data prep into the compact npy format
- Go **EDF → npy directly** and skip the 10 s pickle stage. Keep the `processed_npy` contract unchanged
  (`recordings/<stem>.npy` float32 `[T, 23]` at 200 Hz, 0.1–75 Hz band-pass, 50 Hz notch, µV) so
  `TUABAgeNpyLoader` reads it unchanged.
- Handle the montage differences: `*_a` montages have no A1/A2, so add a per-recording channel mask to the manifest rather than zero-filling silently.
  Also handle the different native sfreqs (250/256/400/512/1000 Hz) and a minimum duration of 60 s.
- Run sharded on SageMaker Processing (generalise `scripts/submit_tuab_npy_conversion.py`), reading EDF from S3.
- Decide: float32 (~same size as the EDF) vs **float16** (half the size, no longer bit-identical).
- Deliverable: `dataset_maker/make_TUH_npy.py`, plus one `manifest.json` per corpus and a **global**
  `tuh_index.parquet` (recording → corpus, subject, age, sex, labels, channel mask, path).

### 4. Code support for training and eval
- A `data.dataset=TUH_MULTI` bundle built from `tuh_index.parquet` with filters (corpus, label
  availability, case filter, age range) and **per-corpus mixture weights** for sampling.
- Multi-task targets with missing labels: an age head (always present) plus pathology heads. The loss is masked
  where a label is absent. Keep `task=age` and `task=binary` working on their own.
- A **global subject-disjoint split**: subjects in TUAB eval (and our pinned `age_split.json` test set) are held out
  across *all* corpora, so the TUAB benchmark stays comparable to published numbers.
- Eval: per-corpus and per-cohort metrics (extends `scripts/eval_age_by_cohort.py`), plus the brain-age gap by
  pathology group.

### 5. Check the scenarios against published SOTA
- **Pathology (TUAB)**: the BIOT/LaBraM/CBraMod protocol (official train/eval split, 10 s windows, balanced accuracy, AUROC,
  AUC-PR). Reference point: LaBraM-Base reports roughly 0.814 BAcc / 0.902 AUROC.
- **Brain age**: Engemann et al. 2022 (NeuroImage, M/EEG brain-age benchmark, TUAB) and
  Gemein et al. (deep-learning brain age on TUH, state-vs-trait brain-age gap). Pull their exact
  protocols (normal-only training? subject split? recording-level aggregation?) and numbers.
- Output: a table "our protocol vs theirs" and one config per published protocol, so the comparisons are like for like.

### 6. Run it
- **Debug**: one TUEG chunk (a few hundred recordings) plus TUEP, end to end on 1 GPU locally (`debug_e2e_out`-style):
  scan → npy → index → train 1 epoch → eval report.
- **Full**: SageMaker, File/FastFile mode on the npy prefix. Baselines first (TUAB-only scenario D vs
  TUH-all for age; TUAB vs TUAB+TUEP for pathology), then multi-task.

### 7. Onboarding pipeline and skills for new or mixed EEG data
- `dataset_maker/onboard/` gets a `DatasetAdapter` interface: `discover()` → `read_metadata()` →
  `read_labels()` → `standardize()` (channels → 10-20 / mask, resample, filter) → `write_npy()` →
  `validate()`. TUH corpora become the first adapters, and non-TUH sets (e.g. a hospital dataset) plug in next.
- A **mixed-data** contract: every source emits rows into the same index schema, and the training mixture is set
  only by config weights.
- Claude skills in `.claude/skills/`:
  - `onboard-eeg-dataset`: write the adapter, run the scan, convert, validate, and register in the index.
  - `analyze-eeg-dataset`: produce the step-2 report and leakage audit for any index slice.
- Docs: `docs/data_onboarding.md`.

## Open decisions
1. NEDC credentials: who requests them, and where do they live (Secrets Manager like ClearML)?
2. Storage: chunked download-to-S3 with 141 GB local (recommended), or attach a ~3 TB gp3 volume.
3. Corpus scope for v1: TUEG + TUAB + TUEP (recommended), with TUSZ/TUSL/TUEV later.
4. npy dtype: float32 or float16.
5. LaBraM-base was pre-trained on TUEG, so check its pretraining exclusions before treating TUEG-derived
   test sets as unseen.
