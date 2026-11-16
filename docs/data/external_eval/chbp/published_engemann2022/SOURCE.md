Per-fold CHBP age-prediction results of Engemann et al. 2022, "A reusable benchmark of
brain-age prediction from M/EEG resting-state signals", NeuroImage 262:119521,
doi:10.1016/j.neuroimage.2022.119521.

Copied unchanged from `results/benchmark-<name>_dataset-chbp.csv` of
https://github.com/meeg-ml-benchmarks/brain-age-benchmark-paper (commit c57eba38c8b90dac9354f0f4f8148dbed7e56029,
fetched 2026-11-14), BSD 3-Clause licence (`LICENSE` here).

Protocol (from `compute_benchmark_age_prediction.py` and `config_chbp_eeg.py` of that repo):
sklearn `KFold(n_splits=10, shuffle=True, random_state=42)` over the participants with EEG,
sorted by id; one row per test fold (MAE, R² against the fold's own mean); eyes-open and
eyes-closed epochs pooled; 0.1–49 Hz; 10 s epochs; autoreject; 53 common channels.
The paper's CHBP table reports the mean ± SD (ddof 1) over these 10 rows.
