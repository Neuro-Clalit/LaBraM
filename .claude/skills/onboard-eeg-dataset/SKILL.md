---
name: onboard-eeg-dataset
description: How to onboard a new EEG dataset (external test cohort or new training data) for the LaBraM brain-age models — access and DUA handling, download, recording-chain audit (filters, reference, EDF calibration, acquisition subgroups, channel labels, protocol), conversion with the training pipeline, label-free QC against Temple / Harvard, the zero-shot evaluation protocol, diagnosing a degradation, label-free fixes, and the extra checks before a dataset may be used for training. Load before downloading, converting, scoring or training on any new EEG corpus (e.g. CHBP, LEMON, MIPDB, TDBRAIN, HEEDB-like data).
---

# Onboarding a new EEG dataset

Worked examples: `docs/chbp_onboarding.md` and `docs/chbp_domain_shift.md`
(CHBP, Synapse), `docs/external_eval_results.md` (LEMON, MIPDB),
`docs/heedb_onboarding_plan.md` (HEEDB, training data).

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
- **Tokens never go into code, commits or chat.**
  - Keep them in the tool's config file (e.g. `~/.synapseConfig`, mode 600) or
    an environment variable.
  - If a token was pasted in a chat, ask the owner to revoke it after use.
- **Storage convention:**
  - local: `/data/datasets/EEG-public/<NAME>/{raw,meta,cohorts}`;
  - S3 mirror: `s3://eeg-data-public/<NAME>/v1/{raw,meta,cohorts}`.
  - The bucket is private despite its name. Run
    `dataset_maker/download_heedb.py::check_private` before every upload.
- **Add the cohort to `sagemaker.data_s3_uris`** in `labram/configs/defaults.py`
  *and* in every default JSON config, so jobs can mount it as an
  `evaluation.test_sets` channel.

## 2. Download

- **List before downloading:** file types, sizes, releases
  (`dataset_maker/download_synapse.py --list`, or the corpus's own index).
  Keep a `listing.jsonl` with ids, sizes and md5s.
- **Download only what is needed** (EEG, sidecars, participant tables; skip
  MRI). Make it resumable, retry transient errors, and respect API rate limits.
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
- S3 mirror, checked private.
- Tests for new code: the full suite must pass.
