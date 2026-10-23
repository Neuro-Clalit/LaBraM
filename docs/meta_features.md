# TUAB meta features: what we know besides the EEG windows

The fine-tune only sees EEG windows and a binary normal/abnormal label. This page
covers what *else* is known per recording, how it is extracted, and what the
analysis in `notebooks/tuab_meta_features.ipynb` found.

```bash
python dataset_maker/make_TUAB_meta.py --root /path/to/TUH_Abnormal/v3.0.0/edf --n_jobs 8
# -> <root>/processed/meta_features.csv  (one row per EDF; ~25 min on 8 cores)
```

The sweep is read-only. It reads each EDF once with MNE and never touches the
window pickles. A file that fails is recorded in the `error` column instead of
aborting the run.

## Sources

| Source | Columns | Notes |
|---|---|---|
| Directory layout | `label`, `tuh_split` | the only clinical label TUAB ships |
| EDF patient field | `age`, `raw_age`, `sex`, `subject` | via `tuh_metadata.py`; `Age:999` = redacted 90+ (see `age_regression.md`) |
| EDF recording field | `session_number`, `token_number`, `year`, `equipment_tag` | the date is anonymised to 01-JAN-YYYY 00:00, so only the year survives; the session number is the patient's n-th TUH EEG |
| EDF signal headers | `sfreq`, `physical_max_uv`, `n_signals`, `has_emg`, `has_eog`, `has_photic_channel`, `has_sphenoidal`, `has_numbered_extra`, `duration_min` | amplifier and montage fingerprint |
| EKG lead | `ekg_hr_bpm`, `ekg_rr_std_ms`, `ekg_valid_frac` | R-peak detection on the 5–30 Hz band-passed lead, polarity-agnostic |
| Photic channel | `photic_performed`, `photic_stim_sec` | stimulator pulse train |
| Device trend channels | `trend_suppr_mean`, `trend_suppr_gt50_frac`, `trend_bursts_mean`, `trend_ibi_mean` | 1-Hz burst-suppression trends (`SUPPR`, `BURSTS`, `IBI`) written by the acquisition software; all-zero means not computed, so the value is NaN |
| EEG quality (19 scalp channels) | `q_*` | clipping, flat/high-amplitude 10-s epochs, flat/noisy electrodes, 50/60 Hz mains ratio |
| qEEG (common average, 0.5–45 Hz, median over 10-s epochs) | `qeeg_*` | relative band powers, slowing ratio, posterior dominant rhythm (PDR) frequency and prominence, aperiodic slope, hemispheric asymmetry, kurtosis |

**Not available anywhere in the data:** the reason a recording is abnormal,
diagnosis, medication, indication, vigilance state, race/ethnicity, exact
date/time. TUH stopped distributing the EEG reports, which held all of these.

## "Why abnormal": qEEG signatures

`labram/eval/meta_analysis.py::signature_flags` flags a recording when a qEEG
feature falls outside the 95th (or 5th) percentile of **normal training
recordings in the same age group**. So about 5% of normals are flagged per
signature by construction. The signatures are diffuse slowing, slow PDR, absent
PDR, asymmetry, focal (delta) asymmetry, sharp transients (kurtosis) and low
voltage. They are screening phenotypes, not diagnoses. Epileptiform discharges
are brief and seldom move a whole-recording statistic, so they are under-detected.

## Findings

Numbers are from the full TUAB v3.0.0 sweep (2,993 recordings, 2,329 subjects, 0 extraction failures).

### What meta information exists

| | coverage | note |
|---|---|---|
| Age | 99.5% | 15 unusable: 12 × `Age:999` (90+, redacted), 3 × `Age:0` |
| Sex | 100% | |
| Session number, recording year | 100% | year **1899** (16 recordings) and **2000** (29) are placeholder dates; treat as missing |
| Heart rate / HRV (EKG lead) | 100% | an EKG lead exists on every recording; R-peaks detected in all |
| Photic stimulation | 95% have the channel; performed in 72% | |
| Device burst-suppression trends | 60% | computed only by some acquisition systems |
| EMG / sphenoidal electrodes | 61% / 3.5% | |
| **Reason for abnormality, diagnosis, medication, vigilance** | **0%** | not distributed by TUH |

### Demographics

* Abnormal patients are **~10 years older** (mean 54.4 vs 44.0). The abnormal rate climbs from 31% (age 20–39) to 72% (80+).
* Sex is nearly balanced: 52% of male recordings are abnormal, vs 47% of female ones.
* 443 subjects have more than one recording, and **54 have both a normal and an abnormal one**.
* **Session number is a strong proxy for illness.** 36% of first EEGs are abnormal, rising to 81% at sessions 4–5 and 96% after more than 10 sessions.

### Data quality

* **The data is mostly clean.** Clipping is essentially absent, and the median recording has 1.4% artifact epochs. Quality tiers: **2,116 good / 791 fair / 86 poor**.
* **Poor quality concentrates in abnormal recordings.** Train has 64 poor abnormal recordings vs 15 poor normal ones; very ill and ICU patients produce more artifact.
* A handful of recordings are essentially unusable: 100% artifact epochs, or 6–8 dead electrodes (a few train recordings and one *eval* recording; the notebook lists them when run locally -- recording ids are not kept in git). All of them are labelled abnormal, so every window they contribute teaches "noise = abnormal".
* **Mains noise is not removed.** `make_TUAB.py` notches 50 Hz, but TUH is US data with 60 Hz mains. In the processed training windows the 60 Hz peak is a median ×2.2 above the neighbouring floor, and over ×10 in 17% of recordings.
* 3 abnormal train EDFs never made it into the processed windows.

### Why is a case abnormal? (putative, from qEEG)

| signature | abnormal | normal |
|---|---|---|
| diffuse slowing | **50%** | 5% |
| focal (delta) asymmetry | **30%** | 5% |
| slow posterior rhythm | **28%** | 4% |
| hemispheric asymmetry | **22%** | 6% |
| absent posterior rhythm | 12% | 5% |
| sharp transients (kurtosis), low voltage | 3–4% | 5–6% (no signal) |

* **72% of abnormal recordings show at least one signature**, vs 27% of normals. Many show several: slowing and slow PDR co-occur, and so do focal slowing and asymmetry.
* **28% of abnormal recordings show none.** These are most likely epileptiform-only abnormalities (brief spikes that whole-recording spectra cannot see), subtle findings, or label noise. They should be the hardest cases for any model.
* The kurtosis "spike" proxy does **not** work: kurtosis is *higher* in normals, because it is dominated by eye blinks and alpha bursts. Recovering epileptiform reasons needs event annotations (TUEV) or a TUEV-trained detector.
* **Context hints:** recordings *without* photic stimulation are **80% abnormal** (837 recordings), vs 37% with it. Sphenoidal electrodes mean 63% abnormal. More than 5 sessions means 90% abnormal.
* The device `SUPPR` trend is *higher* in normals (median 68% vs 30%). It is therefore **not** a clinical burst-suppression percentage; its semantics are unknown, so do not interpret it clinically.

### Can the meta features be used? Metadata-only model, recording level, official eval set

| features | AUROC | bal. acc |
|---|---|---|
| demographics (age, sex) | 0.689 | 0.639 |
| + history (session number, year) | 0.727 | 0.644 |
| acquisition only | 0.686 | 0.658 |
| quality only | 0.574 | 0.575 |
| physiology only (EKG, trends) | 0.783 | 0.721 |
| **qEEG only** | **0.890** | **0.813** |
| demographics + qEEG | 0.901 | 0.832 |
| patient-level (all except acquisition/history) | 0.907 | 0.831 |
| all meta features | 0.922 | 0.863 |

13 hand-crafted spectral summaries plus age already recover most of the TUAB label. In practice the label mostly means "background slowing or asymmetry beyond what is normal for this age". That makes a **strong, cheap baseline** that LaBraM has to beat at the *recording* level.

### Recommendations

1. **Use age (and sex) as inputs.** They are legitimately known at prediction time, and normal EEG ranges are age-dependent. The simplest route is late fusion: concatenate them to the pooled embedding before the head, or stack them on the case-level LaBraM score.
2. **Run a stacking experiment:** a gradient-boosted model on LaBraM's recording-level probability + qEEG + age. If it beats LaBraM alone, the encoder is missing information. If not, LaBraM has already absorbed it.
3. **Do not feed acquisition/history features to the model** (photic performed, session number, device tags, trend availability, sampling rate). They encode Temple's referral workflow and will not transfer to another hospital. **Do use them to stratify evaluation**: check LaBraM within photic yes/no and first-session vs repeat subgroups. A large drop within strata means the model learned the workflow.
4. **Use the putative reasons for error analysis and auxiliary targets.** Measure recall per phenotype, especially on the 28% of abnormals with no signature. Regressing slowing ratio / PDR as an auxiliary head is a cheap multi-task regularizer.
5. **Clean the data.** Drop or down-weight the 86 poor-tier recordings, or mask their bad epochs. Add a 60 Hz notch: this needs a reprocessing pass, and should be checked against the pre-trained checkpoint, which saw the same 60 Hz residue. Treat years 1899/2000 as missing.
6. **To get real abnormality reasons,** request the annotated TUH subcorpora (TUEV events, TUSZ seizures, TUSL slowing, TUEP epilepsy, TUAR artifacts) and join them to TUAB by subject id. They are not on disk or in S3 today, and the id overlap still has to be verified once they are downloaded.

## Files

| Path | Role |
|---|---|
| `labram/data/tuh_meta_features.py` | full EDF header parse + per-recording quality / qEEG / EKG / auxiliary features |
| `dataset_maker/make_TUAB_meta.py` | corpus sweep CLI → `meta_features.csv` |
| `labram/eval/meta_analysis.py` | feature groups, univariate AUC, signature flags, metadata-only baseline |
| `notebooks/tuab_meta_features.ipynb` | the statistics and figures |
| `tests/test_tuh_meta_features.py`, `tests/test_meta_analysis.py` | synthetic-signal tests |
