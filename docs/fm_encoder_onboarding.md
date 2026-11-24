# Onboarding CBraMod and REVE as encoders for brain age

**Question.** Can a different pretrained EEG foundation model replace the LaBraM encoder of the
brain-age model? The review "Replacing LaBraM for EEG Brain-Age Regression" ranks REVE first and
CBraMod second. This document covers what each model needs as input, how our train and test
cohorts supply it, the code that makes the encoder a configuration choice, and the first results.

**Status.**

- **Preprocessing: no re-conversion.** The stored cohorts serve both models. Each encoder gets
  an input adapter inside the model (Section 3).
- **Code: done.** `model.model=cbramod_base_patch200` or `reve_base_patch200` selects the encoder;
  loaders, training loops, bag training, losses, evaluation and checkpoints are unchanged
  (Section 5). Both ports reproduce the upstream networks exactly (Table 4.1).
- **Frozen encoders are weak for age.** With a ridge probe on 2,000 training recordings the best
  frozen encoder reaches 11.6 years of validation MAE (REVE), against 16.3 for predicting the
  mean age and 7.0 for the same probe on fine-tuned LaBraM features (Tables 6.1 and 6.2). The
  replacement question is therefore decided by fine-tuning, not by frozen features.
- **Input per encoder, selected on validation** (Table 6.2): common average reference for all
  three; CBraMod with a 0.3 Hz high-pass and a ±100 µV clip; REVE with a per-window z-score.
- **Fine-tune (stage B).** `CB4_L1` (P4 recipe with CBraMod, fold 0) is defined in
  `scripts/mil_runs/submit_grid.sh` and passes the dry run and a local run on the real fold.
  REVE trains at 0.29 times LaBraM's speed (Table 7.1) and is submitted only after the CBraMod
  result.

The target of every comparison is the best single model on the same split: **P4_L1 fold 0,
validation 6.59 years, test 6.06 years** (`docs/mil_stage1_results.md`, Table 13.14).

## 1. The two encoders

Facts in Table 1.1 were read from the authors' code and model cards, not from the review, which
was written from memory.

### Table 1.1. Encoders compared

Sizes are parameter counts of the encoder alone. "Tokens per 10 s" is channels × patches for our
19 channels.

| | LaBraM-base (current) | CBraMod | REVE-base |
|---|---|---|---|
| Paper | Jiang et al., ICLR 2024 | Wang et al., ICLR 2025 (arXiv:2412.07236) | El Ouahidi et al., NeurIPS 2025 (arXiv:2510.21585) |
| Layers, width | 12, 200 | 12, 200 | 22, 512 |
| Parameters | 5.8 M | 4.9 M | 69.2 M |
| Attention | all tokens | criss-cross: half the width across channels, half across time | all tokens |
| Channel identity | learned embedding per 10-20 name | none: a 19 × 7 convolution over a fixed channel order | Fourier code of electrode x, y, z and patch index |
| Patches | 200 samples | 200 samples | 200 samples, 20 overlap |
| Tokens per 10 s | 190 + class token | 190 | 209 |
| Pretraining data | about 2,500 h, 20 datasets | TUEG, about 9,000 h of 30 s segments below 100 µV | over 60,000 h, 92 datasets |
| Pretraining objective | masked prediction of VQ spectral codes | masked patch reconstruction | masked reconstruction + pooled-token loss |
| Code licence | MIT | MIT | MIT |
| Weight licence | MIT | Apache-2.0 | REVE Responsible Use License v1.0 |

## 2. Cohorts

- **Training, validation, in-domain test:** `tuh_heedb_v1`, 103,117 recordings (Temple 31,312,
  Harvard 71,805). The split is P4's fold 0 of the 5-fold subject split over train + val; the
  7,612 held-out test recordings (Harvard 7,338, Temple 274) are never trained on.
- **External test:** LEMON, Dortmund, TDBRAIN, SRM, CHBP.

All cohorts were converted by the same `dataset_maker/make_TUH_npy.py::preprocess_raw`, so
nothing is cohort-specific in what follows.

### Table 2.1. Pretraining overlap of each encoder with our cohorts

An encoder that saw a cohort during self-supervised pretraining (without labels) is not tested
zero-shot on it. REVE's list comes from a summary of its paper and from our own onboarding docs;
its dataset appendix was not read verbatim.

| Encoder | Saw during pretraining | Clean cohorts |
|---|---|---|
| LaBraM | TUH subsets (TUAR, TUEP, TUSZ, TUSL) | Harvard, all external |
| CBraMod | TUEG, so Temple train and test | Harvard, all external |
| REVE | TUH, TDBRAIN, SRM, NMT, HBN | Harvard, LEMON, Dortmund, CHBP |

The decision cohorts are therefore **Harvard test, LEMON and Dortmund**. Temple, TDBRAIN and SRM
are reported with the overlap flagged.

## 3. Preprocessing

**Decision: the stored npy cohorts are used as they are.** They hold 200 Hz signals, band-passed
0.1–75 Hz, with 50 and 60 Hz removed by spectrum interpolation, in µV as float16, on the recorded
reference; `data.drop_channels=[A1,A2,T1,T2]` leaves 19 channels. The 50 + 60 Hz interpolation is
the line removal for CBraMod and REVE too. Table 3.1 lists what each encoder expects and what the
model does on the fly.

### Table 3.1. Stored signal against each encoder's input

"Action" is applied inside the model (`FoundationRegressor.maybe_preprocess_input`), in the order
clip, common average reference, band limit, z-score, so training and every evaluation path see
the same signal. The selected options are those of Table 6.2.

| Property | Stored cohorts | CBraMod | REVE | Action |
|---|---|---|---|---|
| Sampling rate | 200 Hz | 200 Hz | 200 Hz | none |
| Unit | µV, divided by 100 by the trainer | µV / 100 | fine-tuning: µV / 100; pretraining: z-score per channel and session, clipped at 15 SD | CBraMod: none. REVE: z-score per window, clipped at 15 SD (`model.encoder.input_norm=zscore_window`) |
| Patches | 10 s windows | 200 samples | 200 samples, 20 overlap: 11 patches per 10 s | REVE adapter cuts the window into its patches |
| Band | 0.1–75 Hz | 0.3–75 Hz | pretraining 0.5–99.5 Hz; the authors fine-tune on 0.3–75 Hz data | CBraMod: 0.3 Hz high-pass (`labram_plus.band_hz=[0.3,99]`). 75–99.5 Hz is not in the files and is not rebuilt |
| Mains | 50 + 60 Hz interpolated | 50 + 60 Hz interpolated (pretrained with a 60 Hz notch) | 50 + 60 Hz interpolated (pretraining filter not stated) | none |
| Channels | FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 FZ CZ PZ | the same 19 in the same order | position looked up per name; all 19 are in its table | CBraMod: channels written into its 19 slots. REVE: lookup, an unknown name is an error |
| Reference | as recorded, differs by source | pretrained on Temple average-reference recordings | mixed | common average reference in the model (`labram_plus`), as for LaBraM |
| Large amplitudes | kept | pretrained on segments below 100 µV | clip at 15 SD | CBraMod: clip at ±100 µV (`model.input_clip_uv=100`) |
| Window | 10 s | pretrained on 30 s, any length works | pretrained on 10 s | 10 s for both, the same bags and split as P4 |

Three points behind the table:

- **Channel order.** CBraMod's pretraining script picks the channels in exactly our order, so no
  reordering is needed. The order matters: its only position code is a convolution over the
  channel × patch grid.
- **Amplitude range.** After the common average reference, 35–63 % of our windows have a peak
  above 100 µV (Table 4.3), the limit CBraMod's pretraining segments were selected by.
- **Narrow sources.** CHBP (0.5–50 Hz) and NMT (1–35 Hz) are band-limited at the recording, so a
  re-conversion would not widen them.

## 4. Input checks

`scripts/fm_input_check.py` runs three checks before any training.

### 4.1 Port parity

The vendored networks are compared with the upstream implementations on the same random input
(2 windows × 19 channels × 10 s), with the released weights loaded strictly into both.

### Table 4.1. Largest absolute difference between the vendored and the upstream network

"Tensors" is the number of weight tensors in the released file; every one has a matching
parameter. A difference of 0 means the outputs are bit-identical.

| Encoder | Tensors | Output compared | Largest difference |
|---|---|---|---|
| CBraMod | 211 | reconstruction, masked reconstruction, features | 0, 0, 0 |
| REVE-base | 140 | tokens (2 × 19 × 11 × 512), attention-pooled vector | 0, 0 |

### 4.2 Masked reconstruction with CBraMod

CBraMod's pretrained output layer reconstructs masked patches, so its pretraining task can be run
on our data without labels. Half of the channel × patch tokens of a window are masked, and the
relative error on the masked patches is

$$
e_w = \frac{\sum_{(c,p) \in M_w} \lVert \hat{x}_{w,c,p} - x_{w,c,p} \rVert^2}
           {\sum_{(c,p) \in M_w} \lVert x_{w,c,p} \rVert^2}
\tag{4.1}
$$

where $x_{w,c,p} \in \mathbb{R}^{200}$ is patch $p$ of channel $c$ of window $w$ in model units
after the variant's preprocessing, $\hat{x}$ the network's reconstruction, and $M_w$ the masked
(channel, patch) pairs, drawn once (Bernoulli 0.5) and shared by all variants
(`scripts/fm_input_check.py::recon_errors`). $e_w = 1$ means nothing was reconstructed.

### Table 4.2. Median reconstruction error of CBraMod per input variant

Median of Eq. (4.1) over 2,000 windows per cohort (test split, seeded sample). "Stored" is µV / 100
on the recorded reference; "CAR" adds the common average reference; the next two add a 0.3 Hz
high-pass or a ±100 µV clip to CAR. The two controls use CAR without the division by 100, and CAR
with the channels in a random order.

| Cohort | Stored | CAR | CAR + 0.3 Hz | CAR + clip | Control: no / 100 | Control: shuffled channels |
|---|---|---|---|---|---|---|
| LEMON | 0.287 | 0.367 | 0.266 | 0.354 | 1.001 | 0.610 |
| Dortmund | 0.311 | 0.403 | 0.296 | 0.393 | 1.001 | 0.634 |
| TDBRAIN | 0.297 | 0.457 | 0.330 | 0.446 | 1.001 | 0.699 |
| SRM | 0.582 | 0.586 | 0.466 | 0.572 | 1.000 | 0.737 |
| CHBP | 0.135 | 0.267 | 0.254 | 0.265 | 1.001 | 0.483 |
| Harvard + Temple | 0.298 | 0.415 | 0.289 | 0.376 | 1.000 | 0.665 |

What Table 4.2 shows:

- **The unit is right.** Without the division by 100 the network reconstructs nothing on any
  cohort.
- **The channel order is right.** A shuffled order raises the error on every cohort, by a factor
  of 1.3 to 1.8.
- **It cannot rank the reference.** A shared reference signal makes a masked channel easier to
  predict from the others, so "stored" scores below "CAR" by construction. The probe of Section 6
  decides the reference, and it prefers CAR (Table 6.2).

### 4.3 Amplitude audit

### Table 4.3. Amplitude of the model input per cohort

2,000 windows per cohort (test split), after the common average reference, in µV
(`scripts/fm_input_check.py::audit`). "Median channel SD" is the median over windows and channels
of the standard deviation of one channel over one window. "Peak above" is the share of windows
whose largest absolute sample exceeds the limit. "Clipped at 15 SD" is the share of samples a
per-window z-score clip at 15 SD would change.

| Cohort | Median channel SD | 95th percentile of channel SD | Peak above 100 µV | Peak above 800 µV | Samples above 100 µV | Clipped at 15 SD |
|---|---|---|---|---|---|---|
| LEMON | 9.7 | 41.9 | 54.9 % | 1.5 % | 1.1 % | 0.0 % |
| Dortmund | 10.4 | 37.6 | 55.7 % | 1.3 % | 0.8 % | 0.0 % |
| TDBRAIN | 10.4 | 37.9 | 54.1 % | 1.3 % | 1.0 % | 0.0 % |
| SRM | 13.2 | 35.5 | 38.4 % | 1.4 % | 0.5 % | 0.0 % |
| CHBP | 6.6 | 16.1 | 34.5 % | 0.7 % | 0.3 % | 0.0 % |
| Harvard + Temple | 14.7 | 290.2 | 63.2 % | 12.4 % | 7.1 % | 0.0 % |

Two consequences. In model units (µV / 100) a typical channel has a standard deviation of about
0.1, a tenth of the unit variance REVE was pretrained on; the per-window z-score removes that gap.
And more than half of the windows of most cohorts lie outside CBraMod's pretraining amplitude
range, which is why its clip is tested.

## 5. Code design

One rule: the rest of the pipeline calls the model exactly as before. A new model class,
`FoundationRegressor` (`labram/models/fm_regressor.py`), wraps a foreign encoder and implements
the call contract of `NeuralTransformer`:

- `model(x, channel_indices)` with `x` of shape `[batch, channels, patches, 200]` in µV / 100;
- `return_features=True` returns the head-input feature next to the output;
- `bag=...` runs the recording-level forward. `forward_bag` moved into
  `labram/models/bag_forward.py::BagForwardMixin`, which both model classes use.

The channel names a foreign encoder needs are recovered from the `channel_indices` the loops
already pass, so the training loops, the channel dropper, bag training, the feature cache and the
evaluation scripts are untouched.

- **Encoders** (`labram/models/backbones/`): `EncoderBackbone` is the interface (`encode`,
  `layer_id`, `load_pretrained`). `cbramod.py` and `reve.py` hold the upstream networks, vendored
  under their MIT licences with upstream parameter names, plus the adapters of Table 3.1.
  `reve_positions.json` holds the electrode positions of our channel names. Neither `transformers`
  nor remote code is needed at run time.
- **Selection.** `model.model` stays the single selector (timm registry:
  `cbramod_base_patch200`, `reve_base_patch200`, and tiny random variants for tests).
  `model.encoder` holds the options only these models read: `readout` (`mean`, or `attn` for
  REVE's pretrained pooling), `input_norm`, `clip_sd`.
- **One factory.** `labram/runs/model_factory.py::build_finetune_model` builds every fine-tune
  model; `run_finetune.get_model` and `labram/eval/loading.py` both call it. It rejects options a
  foreign encoder does not support (`codebook_reg`, `codebook_aux`, `lora`, session features, label
  conditioning, the attention head).
- **Weights.** `finetune_checkpoint.finetune` takes the released weight file.
  `load_finetune_checkpoint` recognises it before unpickling anything and loads it strictly into
  the backbone; one of our own run checkpoints takes the usual path, so a warm start works.
- **Layer-wise learning rate.** The model reports each parameter's layer through `layer_id`, and
  `LayerDecayValueAssigner` uses it. With $L$ encoder layers and decay $\gamma$, a parameter of
  layer $i$ trains at the scale

$$
s_i = \gamma^{\,L + 1 - i}, \qquad i = 0, \dots, L + 1
\tag{5.1}
$$

  where $i = 0$ is the input stem, $i = k$ transformer layer $k$, and $i = L + 1$ the readout, the
  head and the bag pool (`labram/runs/run_finetune.py`). For P4's $\gamma = 0.65$ the stem of the
  12-layer CBraMod trains at $s_0 = 0.0037$, as LaBraM's does. For the 22-layer REVE the same
  $\gamma$ would give $s_0 = 5 \times 10^{-5}$, so its job uses $\gamma = 0.8$ ($s_0 = 0.0059$).

Tests: `tests/test_fm_backbones.py` (call contract, channel layout, layer ids, strict weight
loading, input options, factory) and `tests/test_fm_finetune.py` (window-level run and the P4
recipe end to end with both encoders, then re-scoring the run from its folder).

## 6. Stage A: frozen encoders with a ridge probe

**Purpose.** Fix each encoder's input options and rank the frozen representations. It does not
decide the replacement, because frozen features trail fine-tuning.

### 6.1 Protocol

Each encoder is frozen with its released weights, and its window embeddings are cached
(`scripts/mil_cache.py --pretrained`). Two readouts of a window are compared:

$$
h^{\text{pooled}}_{r,k} = \operatorname{LN}\Big(\tfrac{1}{CP} \sum_{c,p} t_{r,k,c,p}\Big) \in \mathbb{R}^{D},
\qquad
h^{\text{channel}}_{r,k} = \Big[\tfrac{1}{P} \sum_{p} t_{r,k,c,p}\Big]_{c = 1}^{C} \in \mathbb{R}^{CD}
\tag{6.1}
$$

where $t_{r,k,c,p} \in \mathbb{R}^{D}$ is the encoder's token of patch $p$, channel $c$, window $k$
of recording $r$, $C = 19$, $P$ the patches per window, and LN a layer normalisation. The pooled
readout is the feature the fine-tune head reads; the per-channel readout keeps the topography, as
the flattened-token probes of the encoders' papers do. The recording feature is the mean over its
$K_r$ cached windows,

$$
f_r = \tfrac{1}{K_r} \sum_{k=1}^{K_r} h_{r,k}
\tag{6.2}
$$

(`mean`), or that mean concatenated with the standard deviation over windows (`mean_std`). The
probe is a ridge regression of the z-scored age,

$$
\hat{w}, \hat{b} = \arg\min_{w, b} \sum_{r \in \text{train}}
\Big(\tfrac{a_r - \mu}{\sigma} - w^\top \tilde{f}_r - b\Big)^2 + \alpha \lVert w \rVert^2,
\qquad
\hat{a}_r = \sigma\,(\hat{w}^\top \tilde{f}_r + \hat{b}) + \mu
\tag{6.3}
$$

where $a_r$ is the age in years, $\mu$ and $\sigma$ the mean and population standard deviation of
the training ages, $\tilde{f}_r$ the features standardised with the training recordings, and
$\alpha$ the penalty chosen by leave-one-out on the training recordings among 25 values from
$10^{-1}$ to $10^{5}$ (`scripts/fm_probe.py::fit_probe`). The tables report

$$
\operatorname{MAE} = \tfrac{1}{n} \sum_{r} \lvert \hat{a}_r - a_r \rvert,
\qquad
R^2 = 1 - \frac{\sum_r (\hat{a}_r - a_r)^2}{\sum_r (a_r - \bar{a})^2}
\tag{6.4}
$$

over the $n$ recordings of the evaluated set, with $\bar{a}$ their mean age, and the paired
difference between two encoders on the same recordings,

$$
\Delta = \operatorname{MAE}_a - \operatorname{MAE}_b
\tag{6.5}
$$

with a 95 % percentile interval from 2,000 bootstrap resamples of recordings
(`labram/eval/cv_compare.py::bootstrap_delta_mae`).

Recordings follow P4's fold 0: fold 0 of `cv_split.json` is the validation set, the other folds
the training set (`scripts/fm_probe.py sample`, seeded, proportional per source dataset). Windows
are P4's: 60 s trimmed at both ends, 19 channels.

### 6.2 Screening of the input options

2,000 training and 1,000 validation recordings of fold 0, the first 10 selected windows of each
(up to 30 for recordings of the cohort's own validation split). Predicting the training mean age
gives a validation MAE of 16.29 years. The same probe on the features of the fine-tuned E2L model
(2,000 training recordings, the cohort's validation split) gives 6.98 years, so the probe itself
is sound.

Variant names: `LB` LaBraM-base, `CB` CBraMod, `RV` REVE-base; `car` / `nocar` with or without the
common average reference; `hp` 0.3 Hz high-pass; `clip` ±100 µV clip; `zw` z-score per window;
`zr` z-score per recording and channel (REVE's pretraining normalisation, on the cached windows);
`attn` REVE's pretrained attention pooling instead of the mean.

### Table 6.1. Screening with the pooled readout

Validation MAE and $R^2$ (Eq. (6.4)) of the ridge probe (Eq. (6.3)) on the pooled readout
(Eq. (6.1)), $n = 1{,}000$ recordings; Δ (Eq. (6.5)) is against `LB_mean` with the `mean`
features, negative when the variant is better. `LB_mean` averages the patch tokens, `LB_cls` reads
the class token.

| Variant | Dim | MAE, mean | MAE, mean_std | $R^2$, mean | Δ vs `LB_mean` [95 % interval] |
|---|---|---|---|---|---|
| `LB_mean` | 200 | 13.92 | 14.39 | 0.22 | reference |
| `LB_cls` | 200 | 14.34 | 14.67 | 0.17 | +0.42 [−0.00, +0.84] |
| `CB_car` | 200 | 14.22 | 14.37 | 0.20 | +0.30 [−0.15, +0.74] |
| `CB_nocar` | 200 | 14.60 | 14.84 | 0.17 | +0.67 [+0.21, +1.18] |
| `CB_car_hp` | 200 | 13.56 | 13.65 | 0.25 | −0.36 [−0.85, +0.12] |
| `CB_car_clip` | 200 | 13.88 | 13.98 | 0.18 | −0.04 [−0.61, +0.53] |
| `CB_car_hp_clip` | 200 | 13.79 | 13.41 | 0.17 | −0.13 [−0.70, +0.49] |
| `RV_car` | 512 | 12.93 | 13.13 | 0.28 | −0.99 [−1.57, −0.42] |
| `RV_nocar` | 512 | 13.07 | 13.28 | 0.31 | −0.85 [−1.33, −0.39] |
| `RV_zw_car` | 512 | 12.97 | 13.09 | 0.31 | −0.95 [−1.46, −0.42] |
| `RV_zw_nocar` | 512 | 13.15 | 13.49 | 0.30 | −0.77 [−1.28, −0.23] |
| `RV_zr` | 512 | 13.99 | 14.43 | 0.22 | +0.06 [−0.43, +0.60] |
| `RV_car_attn` | 512 | 13.19 | 13.49 | 0.26 | −0.73 [−1.21, −0.21] |

### Table 6.2. Screening with the per-channel readout

As Table 6.1 with the per-channel readout of Eq. (6.1). **Bold**: the variant selected for each
encoder.

| Variant | Dim | MAE, mean | MAE, mean_std | $R^2$, mean | Δ vs `LB_mean` [95 % interval] |
|---|---|---|---|---|---|
| **`LB_mean`** | 3,800 | 12.39 | 12.60 | 0.36 | reference |
| `LB_nocar` | 3,800 | 13.43 | 13.63 | 0.26 | +1.04 [+0.67, +1.40] |
| `CB_car` | 3,800 | 12.99 | 13.36 | 0.27 | +0.60 [+0.22, +1.02] |
| `CB_nocar` | 3,800 | 13.81 | 14.06 | 0.21 | +1.41 [+0.99, +1.85] |
| `CB_car_hp` | 3,800 | 12.30 | 12.39 | 0.33 | −0.10 [−0.51, +0.36] |
| `CB_car_clip` | 3,800 | 12.60 | 12.44 | 0.33 | +0.21 [−0.19, +0.62] |
| **`CB_car_hp_clip`** | 3,800 | 12.09 | 11.93 | 0.36 | −0.31 [−0.74, +0.15] |
| `RV_car` | 9,728 | 12.17 | 12.06 | 0.32 | −0.22 [−0.68, +0.31] |
| `RV_nocar` | 9,728 | 12.40 | 12.32 | 0.31 | +0.01 [−0.44, +0.49] |
| **`RV_zw_car`** | 9,728 | 11.55 | 11.63 | 0.44 | −0.84 [−1.27, −0.40] |
| `RV_zw_nocar` | 9,728 | 12.01 | 11.88 | 0.38 | −0.38 [−0.81, +0.09] |
| `RV_zr` | 9,728 | 13.02 | 13.18 | 0.30 | +0.62 [+0.19, +1.07] |

What the screening shows:

- **Frozen features are weak.** No frozen encoder comes within 4.5 years of the fine-tuned
  features. The best explains 44 % of the age variance.
- **The per-channel readout is better for every encoder**, by 0.6 to 1.5 years (Table 6.2 against
  Table 6.1). Averaging over channels discards topography.
- **The common average reference helps every encoder**: by 1.0 years for LaBraM, 0.8 for CBraMod
  and 0.2 to 0.5 for REVE.
- **CBraMod needs its pretraining band and amplitude range.** The high-pass and the clip together
  bring it from 12.99 to 12.09 years, level with LaBraM.
- **REVE prefers a z-score per window.** It is 0.6 years better than µV / 100. The z-score per
  recording, the closest to REVE's pretraining, is the worst variant: with clinical artefacts the
  recording statistics are dominated by a few large windows.
- **Ranking of the frozen encoders:** REVE ahead of LaBraM by 0.8 years (interval excludes 0);
  CBraMod and LaBraM not separable.

### 6.3 Full probe on test and external cohorts

The three encoders with their selected input and the per-channel readout, on 20,000 training and
5,000 validation recordings of fold 0 (30 windows each), the 7,612 test recordings and the five
external cohorts (`scripts/external_runs/fm_probe.sh all`). Results are added to this section when
the run finishes.

## 7. Stage B: fine-tuning on P4's fold 0

One run per encoder, with P4's split, bags, phases and evaluation. Only the encoder, its released
weights and the input options of Table 6.2 change (`scripts/mil_runs/submit_grid.sh`).

- **`CB4_L1`** = `P4_L1` + `model.model=cbramod_base_patch200`, the CBraMod weights,
  `labram_plus.band_hz=[0.3,99.0]`, `model.input_clip_uv=100`. Learning rate 2.8e-4 and layer
  decay 0.65 as in P4.
- **`RV4_L1`** = `P4_L1` + `model.model=reve_base_patch200`, the REVE weights,
  `model.encoder.input_norm=zscore_window`, layer decay 0.8 (Eq. (5.1)).

### Table 7.1. Training and inference throughput on one A10G

Windows of 19 channels × 10 s per second, mixed precision, compute only (no data loading), with
the released weights and the selected input options. "Train" is forward, backward and optimiser
step at the stated batch; "relative" is against LaBraM at batch 256.

| Encoder | Train, batch 256 | Train, batch 128 | Peak memory | Inference | Relative training speed |
|---|---|---|---|---|---|
| LaBraM-base | 817 | 778 | 6.4 GB (batch 256) | 2,726 | 1.00 |
| CBraMod | 638 | 731 | 5.7 GB (batch 256) | 3,329 | 0.78 |
| REVE-base | out of memory | 233 | 15.4 GB (batch 128) | 820 | 0.29 |

CBraMod costs about 1.3 times a P4 fold. REVE costs about 3.5 times, and P4's 256 windows per
step do not fit a 24 GB GPU, so its job needs the batch split over two accumulation steps or over
GPUs.

**Decision rule.** Paired comparison with P4_L1 fold 0 on the same 7,612 test recordings
(`scripts/paired_bag_mae.py`) and on the external cohorts (`scripts/eval_age_by_cohort.py`). A
difference counts when its bootstrap interval excludes 0 or it exceeds 0.38 years, the
single-pair threshold of `labram/eval/research_summary.py`. An encoder replaces LaBraM only if it
wins on Harvard test and does not lose on LEMON and Dortmund; 5-fold cross-validation and the
ensemble follow as a separate step.

**Pre-flight of `CB4_L1`.**
- The P4 recipe ran end to end on a GPU with the released CBraMod and REVE weights under mixed
  precision (synthetic recordings).
- `ONLY=CB4_L1 FOLD=0 scripts/mil_runs/submit_grid.sh --dry_run` stages the weights as the
  `pretrained` channel from `s3://eeg-data-public/models/cbramod/pretrained_weights.pth`.

## 8. How to run

```bash
# input checks (Section 4)
python scripts/fm_input_check.py parity --encoder cbramod --weights ./checkpoints/cbramod-base.pth \
    --upstream <CBraMod>/models
python scripts/fm_input_check.py recon --weights ./checkpoints/cbramod-base.pth \
    --config <P4 fold 0>/run_config.yaml --cohort lemon=<LEMON cohort>::data.eval_minutes=0

# stage A (Section 6): sample, screen, cache, external, fit
scripts/external_runs/fm_probe.sh sample
scripts/external_runs/fm_probe.sh screen
scripts/external_runs/fm_probe.sh all

# stage B (Section 7)
ONLY=CB4_L1 FOLD=0 SPOT=false scripts/mil_runs/submit_grid.sh --dry_run
ONLY=CB4_L1 FOLD=0 SPOT=false scripts/mil_runs/submit_grid.sh

# any fine-tune with another encoder
python -m labram.runs.run_finetune --config labram/configs/defaults/finetune_tuab_age.json \
    --set model.model=cbramod_base_patch200 finetune_checkpoint.finetune=./checkpoints/cbramod-base.pth \
          'data.drop_channels=["A1","A2","T1","T2"]'
```

Weights: `./checkpoints/cbramod-base.pth` (Hugging Face `weighting666/CBraMod`,
`pretrained_weights.pth`, sha256 `0792cb80…7178`) and `./checkpoints/reve-base.safetensors`
(`brain-bzh/reve-base`, `model.safetensors`, sha256 `8ecc6506…e8fe`); both are mirrored under
`s3://eeg-data-public/models/`.

## 9. Licences

- **CBraMod:** code MIT, weights Apache-2.0.
- **REVE:** code MIT. The weights are under the REVE Responsible Use License v1.0: research and
  commercial use are allowed; clinical use is allowed on a lawful basis, but the model is "not
  intended, by itself, for diagnosis"; identifying a person from EEG representations is forbidden;
  a model derived from the weights must carry the licence. The licence text is stored next to the
  weights on S3. The weights are not gated on Hugging Face.

## 10. Out of scope

Re-conversion of the cohorts from raw; LaBraM's own pre-training phases; EEGPT, LUNA and FEMBA;
hybrids with handcrafted features; 30 s windows (HEEDB recordings are spliced where flat windows
were dropped, and the compact training cohort stores listed windows only); 5-fold
cross-validation of a new encoder.
