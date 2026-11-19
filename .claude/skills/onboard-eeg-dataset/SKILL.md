---
name: onboard-eeg-dataset
description: How to onboard a new EEG dataset (external test cohort or new training data) for the LaBraM brain-age models — access and DUA handling, download, recording-chain audit (filters, reference, EDF calibration, acquisition subgroups, channel labels, protocol), conversion with the training pipeline, label-free QC against Temple / Harvard, the zero-shot evaluation protocol, diagnosing a degradation, label-free fixes, and the extra checks before a dataset may be used for training. Load before downloading, converting, scoring or training on any new EEG corpus (e.g. CHBP, LEMON, MIPDB, TDBRAIN, HEEDB-like data).
---

# Onboarding a new EEG dataset

Worked examples: `docs/chbp_onboarding.md` and `docs/chbp_domain_shift.md`
(CHBP, Synapse), `docs/external_eval_results.md` (LEMON, MIPDB),
`docs/tdbrain_onboarding.md` (TDBRAIN, strict DUA, clinical labels),
`docs/dortmund_onboarding.md` (Dortmund, OpenNeuro, CC0),
`docs/open_cohorts_onboarding.md` (SRM, AHEPA, Iowa / Tel Aviv PD, NMT),
`docs/hbn_onboarding.md` (HBN adults, reference only),
`docs/heedb_onboarding_plan.md` (HEEDB, training data). Every onboarded
dataset, with its storage and status, is listed in Section 12.

The rule above everything: **an external test cohort is never used to fit,
tune, select or calibrate anything.**
- Its ages only score.
- Its signals may feed label-free, cohort-level statistics (a spectrum, a
  gain), and those must be reported as test-time adaptation.
- Even aggregate label information counts as leakage: matching a reference to
  the cohort's age distribution, or using its mean age.

## 1. Access, terms and storage

- **Read the terms (DUA) first.** Note them in a memory file: redistribution,
  in-lab sharing, publication rules. Examples: HEEDB, never public; TDBRAIN,
  no in-lab sharing.
- **Access that needs a person: stop and ask the user.** This covers an ORCID
  or browser login, a DUA to sign, a request form, or an ethics approval.
  Examples: TDBRAIN (brainclinics.com), CAUEEG, NMT-4K-EEG. Don't look for a
  way around a closed ACL: TDBRAIN's Synapse entry lists but never downloads.
  Open data (OpenNeuro, `fcp-indi`) is fetched with
  `aws s3 sync --no-sign-request`, so no AWS identity is sent.
- **The GitHub repo is public.** Never commit per-subject or per-recording rows
  of a DUA-restricted cohort (TDBRAIN, HEEDB, TUH). That includes subject ids
  with signal statistics, even without ages or diagnoses. Commit aggregates
  only:
  - add the cohort to `AGGREGATE_ONLY` in `scripts/external_eval_report.py`;
  - use small-cell suppression in notebooks;
  - grep the diff for subject ids (`sub-[0-9]{8}`, `CBM…`, stems) before
    every push.
  If such rows were pushed, a revert is not enough: the history must be
  rewritten and the old branch never merged (the TDBRAIN `qc.csv` of
  2026-11-15 → `tdbrain-onboarding-clean`). Open-licence cohorts (CC0,
  CC BY) may keep per-recording rows.
- **Tokens never go into code, commits or chat.**
  - Keep them in the tool's config file (e.g. `~/.synapseConfig`, mode 600) or
    an environment variable.
  - If a token was pasted in a chat, ask the owner to revoke it after use.
- **Storage convention:**
  - local: `/data/datasets/EEG-public/<NAME>/{raw,meta,cohorts}`;
  - S3 mirror: `s3://eeg-data-public/<NAME>/v1/{raw,meta,cohorts}`.
  - The bucket is private despite its name. Run
    `dataset_maker/download_heedb.py::check_private` before every upload.
  - Mirror **everything that cannot be fetched again without a person**:
    participants and demographics tables, DUA PDFs, encrypted zips. TDBRAIN's
    participants table is a separate manual download; it sat only on local
    disk until 2026-11-19.
  - Archive passwords (TDBRAIN zip) live in a mode-600 file or an env var,
    never in git or on S3.
  - **After mirroring, list every prefix that the doc's storage table names**
    (`aws s3 ls`). The table must match S3, and the dataset must be in the
    registry of Section 12.
- **Add the cohort to `sagemaker.data_s3_uris`** in `labram/configs/defaults.py`
  *and* in every default JSON config, so jobs can mount it as an
  `evaluation.test_sets` channel.

## 2. Download

- **List before downloading:** file types, sizes, releases
  (`dataset_maker/download_synapse.py --list`, or the corpus's own index).
  Keep a `listing.jsonl` with ids, sizes and md5s.
- **Download only what is needed** (EEG, sidecars, participant tables; skip
  MRI). Make it resumable, retry transient errors, and respect API rate limits.
- **Adults only (age ≥ 18).** The models are trained on adults. From a
  paediatric study download only the adult participants and select them from
  the participants table first (HBN: `--min-age 18`). HEEDB v1 is adults only
  too.
- **Check the counts against the data paper.** CHBP's paper reports 282
  participants, but only 250 have EEG.
- **Several releases:** compare the signals directly (md5 and a sample-level
  difference). CHBP 1.0 and 1.1 are identical; only the headers differ. Pick
  the release that the published benchmarks used.

## 3. Audit the recording chain (before converting)

Write one table per dataset with:

| item | where to look | why it matters |
|---|---|---|
| sampling rate, duration | EDF header, `*_eeg.json` | resampling, minimum length |
| **hardware filters** | `channels.tsv` (`low_cutoff` / `high_cutoff` / `notch`), EDF prefilter field | a 50 Hz low-pass removes content our models use for ranking ages (CHBP: r 0.70 → 0.44 in the forward test) |
| line frequency | sidecar, spectrum | 50 vs 60 Hz removal |
| reference | sidecar, channel labels (`-REF`, `-A12`, vertex) | a vertex reference zeroes Cz → re-reference on reading (MIPDB) |
| **EDF calibration** | physical / digital min / max, resolution, bit depth | a different gain or export scale reads as age (CHBP is 0.6× → +6 y) |
| **acquisition subgroups** | number of signals per file, encoding, cap, site, native rate | CHBP's 120-channel cap (16-bit) was much worse than its 64-channel cap (12-bit) |
| channel labels | raw labels | spaces (`F3 -REF`), suffixes (`-A12`), truncated labels (`F4 -`), 10-10 names (T7/T8/P7/P8 → T3/T4/T5/T6) |
| coverage | label → 10-20 map | the models need the 19 10-20 channels (A1 / A2 / T1 / T2 may be missing) |
| protocol | `events.tsv`, annotations | keep the resting part. CHBP: eyes closed, then eyes open / closed, cut at hyperventilation; photic and HV are not resting |
| clipping / range | fraction of samples at the digital limits | 12-bit files with ±246 µV range |
| **undocumented release processing** | median channel std against Temple / Harvard; PSD edges | NMT is released band-limited to about 1–35 Hz and stored at about 1/100 of real amplitude, although the header says µV. The reader's ×100 is a label-free assumption: state it. A forward test (Section 7) shows whether any model can survive it |
| ages | participants table | units, bins (LEMON 5-year bins → midpoints), redaction (TUH `Age:999`), children vs adults |

**Compare with the training data's recording chain:** TUH / HEEDB clinical
EEG, 0.1–75 Hz after preprocessing, absolute µV, drowsiness and artifacts
common.

## 4. Convert with the training pipeline

- **Add a reader** to `dataset_maker/make_external_npy.py::DATASETS`. Every
  corpus goes through `make_TUH_npy.preprocess_raw`, as the training cohorts
  did: 0.1–75 Hz, line removal at 50 / 60 Hz, 200 Hz, µV, float16, the 23 TUH
  channels with `channel_mask`.
- **The reader is the only corpus-specific part:** it matches labels, crops to
  the resting part, and re-references if needed.
- **Test on one recording first**, then convert all of them. Every failure must
  be explained and fixed, not dropped. Two CHBP files had label quirks.
- **Output:** `manifest.json`, `age_metadata.json`, and `age_split.json` with
  every recording in `test` for an external cohort. Recording stems must be
  unique.
- **Mirror to S3, privately.** Add a launcher `scripts/external_runs/<name>.sh`
  (download / convert / score / report steps).

## 5. Label-free QC against Temple / Harvard

Run `scripts/domain_shift_analysis.py`:
- per-recording robust features on exactly what the models see (first 5 min
  after a 60 s trim, 19 channels, CAR), with the median over windows because
  clinical recordings carry big artifacts;
- compare with **age-matched** Temple and Harvard normal reports;
- weight to the cohort's age histogram for *description only*.

Check these:
- **Amplitude:** channel std and the dynamic range used.
- **Band powers:** absolute and relative, *including below 1 Hz and 45–70 Hz*.
- **Aperiodic:** exponent and offset.
- **Alpha:** peak frequency and height.
- **Artifacts:** the share of windows above 100 µV.
- **Separability:** the AUC against each reference.
- **Spectrum:** the PSD ratio plotted with the specified filter response
  (`scripts/domain_shift_figures.py`). A deficit far below the filter curve
  means a population / state difference (no EMG), not the filter.
- **Acquisition groups:** with `scripts/recording_amplitude_features.py stats`,
  compare each group's median amplitude and 45–70 Hz share with the range of
  the training groups. Outside by more than about 1.5× → expect a biased,
  degraded result.
- **Everything per acquisition subgroup** (cap, site, device), not only for
  the whole cohort.

## 6. Zero-shot evaluation protocol

Score with:
```
scripts/eval_age_by_cohort.py --data-path <cohort> --splits test --pred-stats 53.52 19.29 --stage-dir none --run ...
```
- Use the first 5 minutes (training protocol) and every window
  (`--eval-minutes 0`).
- For SageMaker artifacts without `log/run.log`, give `--pred-stats`
  explicitly.

Report, with `scripts/external_age_metrics.py`:
- **MAE with bootstrap CI, bias, r, R², error slope, MAE per age bin and per
  subgroup.**
- **Two baselines:** the cohort's own mean, and the training mean (53.5 y), the
  honest external baseline.
- **Adults and children separately.**
- **An age-matched in-domain reference** (Temple / Harvard at the cohort's
  ages; see Table 3.1 of `docs/chbp_domain_shift.md`). It separates a real
  degradation from a narrow-age-range effect: a narrow range caps R² by
  itself.
- **Bias correction** only with trends fitted on the training domain's own
  validation predictions. Never fit it on the external cohort.
- **Literature numbers from the primary source** (paper table or the authors'
  results files), with protocol, n and CV design. Published within-cohort CV
  results are not zero-shot. Compare like for like, or say that it isn't.
- **Check whether the corpus was in a pre-training set.** A cohort outside
  LaBraM's pre-training corpus is a fair zero-shot test. Note other models'
  corpora too (Dortmund is in LEAD's), as in Table 23.2 of
  `docs/tdbrain_onboarding.md`.
- **Gap corrections that use the true age** (linear / quadratic, Section 10.1
  of the tech report) produce a residual, not an accuracy. Never report their
  mean absolute value as an MAE.

### Which cohorts may appear in a final report

Final reports are `docs/tech_report/` and any paper-style summary. Working
docs (`docs/<name>_onboarding.md`, result CSVs) keep every cohort.

- **Rejected as an age benchmark.** These cohorts are kept for contrasts inside
  the cohort only:
  1. **Too few adult units.** Below about 100, or when the 95 % interval of the
     MAE is wider than about 4 years (MIPDB adults, n = 31). The same rule
     applies to very small sub-groups in final tables. Drop them from final
     tables, figures and text, with one sentence saying they were scored and
     why they are not shown.
  2. **A released signal that no model variant corrects.** A band or scale
     differs from the training data (NMT: 1–35 Hz, undocumented ×1/100).
  3. **Healthy controls predicted with a cohort-wide offset of unknown
     cause.** AHEPA, Iowa PD and Tel Aviv PD controls (65–69 y) come out 7–11 y
     too young, more than the age compression explains. Use these cohorts only
     for patient-vs-control gaps inside the cohort
     (`scripts/clinical_cohort_gap.py`). A cohort-wide offset cancels in that
     difference.
- **Paediatric studies stay out of every final report,** even their adults
  (HBN: 18–22 y). Their results live only in their onboarding doc.
- **Keep large healthy cohorts with exact ages test-only.** Dortmund (20–70 y,
  a 5-year follow-up of 208 people) is never trained on.
- **Every scored cohort goes into the tech report's cohort table** (Table 17
  in Section 10.2), with n, MAE, status and the reason for any rejection.

## 7. If performance degrades: find the cause

1. **Age range or domain?** Check with the age-matched in-domain reference
   (Section 6).
2. **Forward tests on known truth.** Take an age-matched Temple / Harvard
   sample with `scripts/make_transformed_cohort.py`. Make it look like the new
   cohort (gain `scale:k`, `bandlimit:lo,hi`, …) and score the frozen models.
   - A pure gain moves the bias only.
   - A band limit breaks the ranking.
3. **Inverse tests:** normalize the new cohort toward the training data with
   label-free methods and rescore.
4. **Error vs features within the cohort** (partial Spearman correlation on
   age). Then check whether the cohort's feature values are *outside* the
   training support (CHBP: amplitude below anything young in training).

## 8. Fixes, in order of preference

1. **Per-cohort model choice** (`scripts/gated_ensemble.py`). Use E2L for
   cohorts inside the training range, and the device-robust model for cohorts
   outside it. Decide per cohort, not per recording: single recordings are
   rarely outliers.
2. **Spectrum matching per acquisition group** to a *label-free* Temple /
   Harvard reference (random adults, not age-matched):
   `make_transformed_cohort.py psdmatch`.
3. **Device-robust training:** `labram_plus.band_hz=[1,45]` plus
   `labram_plus.amplitude_norm=window`, or `augment.gain_prob` /
   `lowpass_prob` / `highpass_prob`.
   - The cost is about 0.3–0.6 y in-domain and 1.4–2 y on young adults,
     because amplitude is a U-shaped age cue.
   - Band-only keeps in-domain accuracy but also keeps a gain bias.
   - See Tables 10.6–10.10 of `docs/chbp_domain_shift.md`.

## 9. Before a new dataset is used for **training**

- **Decide first:** test-only, or training data? Keep at least one external
  cohort untouched as a pure test.
- **Subjects and leakage:**
  - subject-disjoint splits;
  - an audit for the same patients across corpora (`tueg_patient_split`, ID
    maps);
  - one recording per session.
- **Labels:** age provenance and redaction, report labels / health status,
  class balance, and the age distribution against the existing training
  cohorts.
- **Signal compatibility:** the recording-chain audit (Section 3) and QC
  (Section 5). A new device or band, or a gain outside the training range, can
  teach the model shortcuts.
- **Add it as an `evaluation.test_sets` entry first.** Score it zero-shot, then
  train and compare on shared, leakage-free sets (`analyze-experiments` skill)
  with the per-dataset breakdown.
- **Re-check every external cohort after training on it,** for gains and for
  losses.

## 10. Practical pitfalls (seen while onboarding CHBP)

- **zsh:** `$V:r…` is a history modifier (`:r` strips the extension). Write
  `${V}:…`. Quote `[...]` arguments (`'labram_plus.band_hz=[1.0,45.0]'`).
- **`pgrep -f <pattern>` inside a wait loop matches the loop itself.** Wait on
  a file instead.
- **Background tool commands stop after 2 h.** Start long training runs with
  `nohup setsid … &`.
- **The root disk (and `/tmp`) can fill up and break tests.** Use
  `pytest --basetemp=/data/tmp/...` and `TMPDIR`, but run `tests/test_staging.py`
  on the default temp (it depends on free space).
- **Default JSON configs are loaded strictly.** A new config field must be
  added to every `labram/configs/defaults/*.json`.
- **Float32 overflow in FFT filters:** write `(f/fc)^16` high-passes as
  `1/(1+(fc/f)^16)`.
- **Session features (`data.session_features`) drop external
  `evaluation.test_sets`.** Score those cohorts afterwards with
  `--session-features`.
- **SageMaker quota:** one ml.g5.12xlarge on-demand and one spot.
  - Spot restarts have hung at the first NCCL all-reduce.
  - ml.g6e capacity may never arrive.
  - `submit_device_robust.sh` takes `INSTANCE=` and `--set sagemaker.use_spot=true`.
- **Feature extraction with wavelet scattering** needs the repo `.venv`
  (kymatio 0.3 needs scipy < 1.17).

## 11. Deliverables

- `docs/<name>_onboarding.md`: data, recording chain, QC, results, following
  the `notebook-docs-style` skill (numbered equations, tables and figures).
- Tables in `docs/data/external_eval/<name>/` and figures in
  `docs/figures/<name>/`.
- Launcher `scripts/external_runs/<name>.sh`.
- A CLAUDE.md line if new code or options were added.
- A memory note for access and terms.
- S3 mirror, checked private, listed against the doc's storage table.
- A row in the registry below (Section 12).
- Tests for new code: the full suite must pass.

## 12. Onboarded datasets: storage and status

Local root `/data/datasets/EEG-public/<NAME>/` (HEEDB:
`/data/datasets/HEEDB/v4.1`). S3: private, key-only bucket
`s3://eeg-data-public/` with all four public-access blocks on. "In configs"
means the cohort is in `sagemaker.data_s3_uris`
(`labram/configs/defaults.py::DEFAULT_SAGEMAKER_DATA_S3_URIS` and every default
JSON config), so it can be an `evaluation.test_sets` channel. Training corpora
are given as an `s3://` `data.data_path` instead.

| dataset | source, licence | S3 (private) | cohort | in configs | status | doc |
|---|---|---|---|---|---|---|
| TUH (TUAB, TUEG, …) | NEDC rsync, DUA (no redistribution) | `TUH_EEG/<corpus>/<ver>/` | TUH cohorts, `tuab/v3.0.1/edf/processed_npy` | n/a (`data.data_path`) | training | memory `tuh-nedc-access` |
| HEEDB v4.1 | BDSP credentialed access point, DUA (never public) | `HEEDB/v4.1/{raw,metadata,index,processed_npy_…,cohorts}` | `heedb_routine_v1`, `heedb_normal_v1`, `tuh_heedb_v1(_compact)` | n/a (`data.data_path`) | training (adults, routine EEG) | `docs/heedb_onboarding_plan.md` |
| LEMON | `s3://fcp-indi/…/MPI-LEMON`, public | `LEMON/v1/{raw,meta,cohorts/lemon_v1}` | `lemon_v1` | yes | test, used | `docs/external_eval_results.md` |
| MIPDB | `s3://fcp-indi/…/EEG_Eyetracking_CMI_data`, public | `MIPDB/v1/{raw,meta,events,cohorts/mipdb_v1}` | `mipdb_v1` | yes | test; adults n = 31, out of final reports | `docs/external_eval_results.md` |
| CHBP | Synapse `syn22324937`, token | `CHBP/v1/{raw,cohorts/chbp_v1}` | `chbp_v1` | yes | test, used (recording-chain case) | `docs/chbp_onboarding.md`, `docs/chbp_domain_shift.md` |
| MNCS / HarMNqEEG | Synapse `syn26712693` | `MNCS/v1/raw` | — (cross-spectra, no EEG) | no | reference spectra only | `docs/chbp_onboarding.md` |
| TDBRAIN v3.1 | brainclinics.com, ORCID + DUA (no in-lab sharing) | `TDBRAIN/v1/{zips,raw,meta,cohorts/tdbrain_v1}` (meta holds the participants table) | `tdbrain_v1` | yes | test, used; aggregates only in git | `docs/tdbrain_onboarding.md` |
| Dortmund Vital Study | OpenNeuro ds005385, CC0 | `DORTMUND/v1/{raw,meta,cohorts/dortmund_v1}` | `dortmund_v1` | yes | test only, accepted | `docs/dortmund_onboarding.md` |
| SRM | OpenNeuro ds003775, CC0 | `SRM/v1/{raw,meta,cohorts/srm_v1}` | `srm_v1` | yes | test, accepted (small) | `docs/open_cohorts_onboarding.md` |
| AHEPA | OpenNeuro ds004504, CC0 | `AHEPA/v1/{raw,meta,cohorts/ahepa_v1}` | `ahepa_v1` | yes | contrasts only (controls offset) | `docs/open_cohorts_onboarding.md` |
| Iowa PD | OpenNeuro ds004584, CC0 | `IOWAPD/v1/{raw,meta,cohorts/iowapd_v1}` | `iowapd_v1` | yes | contrasts only (controls offset) | `docs/open_cohorts_onboarding.md` |
| Tel Aviv PD | OpenNeuro ds007526, CC0 | `TELAVIVPD/v1/{raw,meta,cohorts/telavivpd_v1}` | `telavivpd_v1` | yes | contrasts only (controls offset) | `docs/open_cohorts_onboarding.md` |
| NMT | Zenodo 10.5281/zenodo.21629135, CC BY 4.0 | `NMT/v1/{raw,meta,cohorts/nmt_v1}` | `nmt_v1` | yes | normal / abnormal contrast only (band-limited, scale) | `docs/open_cohorts_onboarding.md` |
| HBN adults | OpenNeuro ds005505–ds005516, CC-BY-SA 4.0 | `HBN/v1/{raw,cohorts/hbn_adults_v1}` | `hbn_adults_v1` | no (paths in `scripts/external_runs/hbn.sh`) | reference only, out of every final report | `docs/hbn_onboarding.md` |

Not onboarded, because each needs the user: CAUEEG (request form + ethics
approval), NMT-4K-EEG (DUA), TUM chronic pain on OSF (licence unverified).
