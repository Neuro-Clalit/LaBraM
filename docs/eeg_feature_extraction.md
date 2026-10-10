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
NPY=/data/datasets/EEG-public/TAUB/TUH_Abnormal/v3.0.0/edf/processed_npy
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
