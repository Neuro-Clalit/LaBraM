# Handcrafted EEG features on TUAB

`notebooks/eeg_feature_extraction_tuab.ipynb` reviews the state of the art in EEG feature
extraction, with a focus on multi-scale methods (DWT/WPD/CWT wavelets, multiscale entropy,
filter-bank Riemannian covariances, wavelet scattering networks) and spectrogram features (STFT, multitaper, band-power dynamics, cepstral coefficients). It then benchmarks those feature families on TUAB for
**pathology** (normal vs abnormal) and **brain age** (normal recordings), and audits the data
preparation. It is a classical-ML reference point for the LaBraM fine-tunes, and a tool for
looking at the data.

## Running it

```bash
pip install -r requirements-features.txt          # PyWavelets, kymatio
# 1) extract features once (~45 min each on 5-6 cores; ~1.1 GB + small scattering cache)
NPY=/data/datasets/EEG-public/TUH_EEG/tuab/v3.0.1/edf/processed_npy
python -m labram.features.tuab --npy_root $NPY --n_jobs 6
python -m labram.features.tuab --npy_root $NPY --n_jobs 6 --kind scattering
python -m labram.features.tuab --npy_root $NPY --n_jobs 6 --kind spectrogram   # ~3 min
# 2) open / execute the notebook (kernel "LaBraM (.venv)")
jupyter nbconvert --to notebook --execute --inplace notebooks/eeg_feature_extraction_tuab.ipynb
```

Environment variables read by the notebook: `TUAB_NPY_ROOT`, `TUAB_FEATURE_CACHE`,
`TUAB_SCATTERING_CACHE`, `TUAB_SPECTROGRAM_CACHE`, `N_JOBS`.
If the cache is missing, the notebook builds it.

## Code

| Module | Contents |
|---|---|
| `labram/features/extractors.py` | Per-window extractors on `[..., C, T]` arrays. **Time:** Hjorth, line length, moments, ZCR. **Welch spectral:** absolute/relative band power, ratios, SEF95, spectral entropy, peak alpha, 1/f exponent + offset. **1 Hz log-PSD bins.** **DWT:** per-level log/relative energy + wavelet entropy, any pywt wavelet/depth. **Wavelet packets.** **Complex-Morlet CWT:** pywt `cmorB-C` parameterisation, evaluated in the frequency domain; per-window features use complex demodulation at a reduced rate. **Multiscale permutation entropy.** **Wavelet scattering**: kymatio `Scattering1D`, time-averaged log S1 and renormalised log S2/S1. **Spectrograms**: STFT and multitaper (DPSS) log-spectrogram mean/std over time, band-power dynamics (variability, lag-1 persistence, spectral flux), MFCC-style cepstral coefficients on a mel-like warped axis. Band-pass and covariances. |
| `labram/features/modeling.py` | Shared modelling layer: merges the caches (`load_feature_table`), builds feature sets with named column groups, the Riemann tangent-space transformer, a model factory (linear / HistGradientBoosting / XGBoost / random forest), subject-grouped CV + held-out test (`cross_validate`), group permutation importance and intrinsic (coefficient / gain / impurity) group importance, and a pickle-backed results cache. |
| `labram/features/export.py` | Training-ready export of the caches (`flat_v1`): one float32 `features.npy` matrix per dataset in the canonical column order, `fbcov.npy`, `meta.parquet` with targets and the notebook's splits (`fold_abn`, `fold_age`, `split` for TUAB; `cohort_split` for TUEG), `columns.parquet`, and `info.json` with checksums. `python -m labram.features.export --dataset tuab|tueg ...`. |
| `labram/features/dataset.py` | `load_flat` (column selection by group/family, memory-mapped; `design()` appends the covariances for `FBTangentSpace`) and `HandcraftedFeatureDataset`, a PyTorch `Dataset` with a train-rows standardiser. |
| `labram/features/riemann.py` | Riemannian (Karcher) mean and tangent-space vectors, NumPy-only. |
| `labram/features/tuab.py` | `extract_recording` / `extract_recording_scattering` / `extract_recording_spectrogram` / `extract_corpus` over the `processed_npy` export, plus the preparation variants (`PREP_STORED`, `PREP_CAR`, `PREP_REF`). Scattering runs at 100 Hz (anti-aliased resample of the 1–45 Hz signal) with J = 7 and J = 9, Q = (8, 1). |

### Reference preparation used by the benchmark

1. Start from the stored signal (`make_TUAB.py`: 0.1–75 Hz, 50 Hz notch, 200 Hz).
2. Keep the 19 scalp 10-20 channels (drop A1/A2/T1/T2) and apply a common average reference.
3. Zero-phase 1–45 Hz Butterworth band-pass on the continuous segment. This also removes the 60 Hz US mains noise, which the stored 50 Hz notch misses.
4. Skip the first minute, then take the next 5 min as 10 s windows.
5. Reject windows with |x| > 800 µV or a flat channel.
6. Compute features per window and take the median over clean windows.

The cache also holds per-window "core" features under the stored and CAR-only preparations,
for the preparation ablations.

## Models, importance, age as classes and the TUEG cohort (§8–§13)

The second half of the same notebook reuses the caches and splits to:
- fit 4 model families (linear, HistGradientBoosting, XGBoost, random forest) on 13 feature groups plus all groups combined, for both tasks (§8);
- report group-level feature importance (joint permutation and intrinsic) for the combined models, and run a **greedy forward selection of feature groups** for age with ridge regression, then refit the selected subsets with every model family (§9, §9.1; `modeling.forward_group_selection`, `modeling.one_se_choice`);
- test whether adding the normal/abnormal label, or a predicted probability of abnormality, improves age prediction on all recordings (§10);
- explain why ridge beats the trees for age: train/validation gaps, validation and learning curves, nested tuning, ridge + XGBoost stacking, feature scalers, lasso / elastic net (§11);
- predict age as 1-, 2- or 5-year classes (multinomial, ordinal Frank & Hall, XGBoost, random forest) and compare with ridge (§12);
- train ridge on the TUAB normal recordings, the TUEG healthy subset, or both pooled, and test on both cohorts, for every feature group (§13).

Results are cached next to the feature caches, so re-executing is cheap after the first run:

| Cache | Sections | Override |
|---|---|---|
| `model_results_v1.pkl` | §8–§12 | `TUAB_MODEL_RESULTS` |
| `model_results_l1_v1.pkl` | §11.7 (lasso / elastic net) | – |
| `subset_results_v1.pkl` | §9.1 (forward selection) | – |
| `cohort_results_v1.pkl` | §13 (TUAB + TUEG) | – |

The TUEG features (`tueg_healthy_cohort_v1.npz`, `TUEG_FEATURE_DIR`) come from
`python -m labram.features.tuab --kind cohort --stems_file ...`, which computes the same
families as the main extractor except scattering and multitaper, about 3× faster.
`modeling.cache_riemann_refs(True)` memoises the Riemannian reference means across refits of
the same folds (identical results, ~5× faster subset searches).

## Training-ready export (`flat_v1`)

```bash
NPY=/data/datasets/EEG-public/TUH_EEG/tuab/v3.0.1/edf/processed_npy
python -m labram.features.export --dataset tuab --npy_root $NPY
python -m labram.features.export --dataset tueg --npy_root $NPY \
    --npz /data/datasets/EEG-public/TUH_EEG/tueg/v2.0.2/features_handcrafted/tueg_healthy_cohort_v1.npz
```

| Export | Recordings | Per-channel columns | Location (local, and the same path under `s3://eeg-data-public/`) |
|---|---|---|---|
| TUAB v3.0.1 | 2,990 | 10,374 (12 families) + `fbcov` | `TUH_EEG/tuab/v3.0.1/edf/features_handcrafted/flat_v1/` |
| TUEG healthy subset v1 | 8,592 | 5,377 (10 families: no scattering / multitaper) + `fbcov` | `TUH_EEG/tueg/v2.0.2/features_handcrafted/flat_v1/` |

The export reproduces the notebook exactly: identical columns to `build_feature_set`, the same
folds, and the same results (ridge on the 1 Hz bins: CV 8.48 / test 7.62 y; forward top-4 set:
TUAB test 7.02 y, TUEG test 8.36 y). Each folder has a README with scikit-learn and PyTorch
examples.
