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
- **Frozen encoders are weak for age.** A ridge probe on 20,000 training recordings reaches 9.3
  to 9.7 years of test MAE with a frozen encoder (Table 6.3), against 6.06 for the fine-tuned P4
  model. The replacement question is therefore decided by fine-tuning, not by frozen features.
- **Frozen ranking** (Table 6.4). REVE is the best in-domain: 0.41 years better than LaBraM on
  Harvard test. CBraMod is the best on the two external decision cohorts: 2.0 years better than
  LaBraM on LEMON and 0.7 on Dortmund. Both are 4 to 5 years better than frozen LaBraM on CHBP.
- **Input per encoder, selected on validation** (Table 6.2): common average reference for all
  three; CBraMod with a 0.3 Hz high-pass and a ±100 µV clip; REVE with a per-window z-score.
- **Fine-tune (stage B): both runs are in progress and go through P4's full 17 epochs.** No
  final result yet. At the same epoch, REVE is 0.6 to 0.9 years ahead of P4 and CBraMod about
  2 years behind (Table 7.3). The CBraMod run was stopped once from outside in epoch 5 and
  continued from its checkpoint.
- **Next candidate: REVE with only its last 6 layers trained.** It runs 2.1 times faster and fits
  P4's batch (Table 7.4). The job is prepared and not started.

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
its dataset appendix was not read verbatim. LaBraM's row is the review's reading of its paper.

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

The three encoders with their selected input (Table 6.2) and the per-channel readout, fitted on
20,000 training recordings of fold 0 with up to 30 windows each (`mean` features, Eq. (6.2)), and
scored on 5,000 validation recordings, the 7,612 test recordings and the five external cohorts
(`scripts/external_runs/fm_probe.sh all`). Windows of the external cohorts follow their usual
scoring: every window, and no trims for the cohorts of short recordings (Dortmund, TDBRAIN, SRM).
External cohorts are restricted to adults (18 years and older) and scored per recording.

### Table 6.3. Frozen encoders with a ridge probe: MAE and $R^2$ per cohort

MAE in years and $R^2$ (Eq. (6.4)) of the ridge probe (Eq. (6.3)); $n$ recordings. Feature width:
3,800 for LaBraM and CBraMod, 9,728 for REVE. **Bold**: the lowest MAE of the row. "Overlap" marks
a cohort an encoder saw during pretraining (Table 2.1): C = CBraMod, R = REVE, L = LaBraM.

| Cohort | $n$ | Overlap | LaBraM MAE | CBraMod MAE | REVE MAE | LaBraM $R^2$ | CBraMod $R^2$ | REVE $R^2$ |
|---|---|---|---|---|---|---|---|---|
| Validation (fold 0) | 5,000 | L, C, R on its Temple part | 9.82 | 9.92 | **9.56** | 0.57 | 0.56 | 0.59 |
| Test, all | 7,612 | L, C, R on its Temple part | 9.72 | 9.59 | **9.30** | 0.62 | 0.63 | 0.65 |
| Test, Harvard | 7,338 | none | 9.75 | 9.61 | **9.34** | 0.62 | 0.63 | 0.65 |
| Test, Temple | 274 | L, C, R | 8.79 | 8.89 | **8.32** | 0.59 | 0.61 | 0.65 |
| LEMON | 214 | none | 10.85 | **8.84** | 9.61 | 0.58 | 0.71 | 0.64 |
| Dortmund | 3,262 | none | 9.46 | **8.72** | 9.05 | 0.34 | 0.44 | 0.41 |
| TDBRAIN | 2,222 | R | 9.88 | **9.57** | 9.80 | 0.37 | 0.42 | 0.38 |
| SRM | 152 | R | 10.14 | **9.96** | 11.32 | 0.08 | 0.11 | −0.12 |
| CHBP | 250 | none | 15.32 | 11.56 | **10.40** | −3.09 | −1.71 | −1.11 |

### Table 6.4. Frozen CBraMod and REVE against frozen LaBraM, paired

Δ (Eq. (6.5)) on the recordings of Table 6.3, with its 95 % interval; negative when the encoder
is better than LaBraM. **Bold**: the interval excludes 0.

| Cohort | CBraMod − LaBraM | REVE − LaBraM |
|---|---|---|
| Validation (fold 0) | +0.11 [−0.07, +0.28] | **−0.25 [−0.43, −0.07]** |
| Test, all | −0.13 [−0.27, +0.01] | **−0.42 [−0.55, −0.28]** |
| Test, Harvard | −0.14 [−0.28, +0.00] | **−0.41 [−0.55, −0.28]** |
| Test, Temple | +0.10 [−0.64, +0.87] | −0.47 [−1.24, +0.30] |
| LEMON | **−2.01 [−2.84, −1.20]** | **−1.24 [−2.11, −0.32]** |
| Dortmund | **−0.74 [−0.98, −0.49]** | **−0.41 [−0.68, −0.15]** |
| TDBRAIN | −0.31 [−0.63, +0.01] | −0.09 [−0.40, +0.25] |
| SRM | −0.19 [−1.53, +1.14] | +1.18 [−0.18, +2.58] |
| CHBP | **−3.76 [−5.15, −2.25]** | **−4.92 [−6.61, −3.27]** |

What the full probe shows:

- **Frozen features stay far from fine-tuning.** With ten times the training recordings of the
  screening, the validation MAE improves from 11.6–12.4 years to 9.6–9.9. On test the probes
  reach 9.3–9.7 years, against 6.06 for P4.
- **In-domain, REVE is ahead.** On Harvard test it is 0.41 years better than LaBraM; CBraMod is
  0.14 better, with an interval that touches 0 (Table 6.4).
- **On the external decision cohorts, CBraMod is ahead.** It is 2.0 years better than LaBraM on
  LEMON and 0.7 on Dortmund; REVE is 1.2 and 0.4 better.
- **Device shift.** On CHBP frozen LaBraM fails (15.3 years), while CBraMod and REVE reach 11.6
  and 10.4. All three still have a negative $R^2$ there.
- **Pretraining overlap did not help REVE.** On TDBRAIN it equals LaBraM, and on SRM it is the
  worst of the three, although both cohorts were in its pretraining data.

One caveat on reading Table 6.4 as "encoder against encoder": each encoder runs with its own
selected input. CBraMod's clip and REVE's z-score bound the amplitude, which also makes an input
more robust across devices; LaBraM was screened with and without the average reference only. Part
of the external gain may therefore come from the input options and not from the encoder.

Both candidates are at least as good as LaBraM when frozen, and better outside the training
domain. That is a reason to run the fine-tunes of Section 7, not a result about them.

## 7. Stage B: fine-tuning on P4's fold 0

One run per encoder, with P4's split, bags, phases and evaluation. Only the encoder, its released
weights and the input options of Table 6.2 change (`scripts/mil_runs/submit_grid.sh`).

- **`CB4_L1`** = `P4_L1` + `model.model=cbramod_base_patch200`, the CBraMod weights,
  `labram_plus.band_hz=[0.3,99.0]`, `model.input_clip_uv=100`. Learning rate 2.8e-4 and layer
  decay 0.65 as in P4.
- **`RV4_L1`** = `P4_L1` + `model.model=reve_base_patch200`, the REVE weights,
  `model.encoder.input_norm=zscore_window`, layer decay 0.8 (Eq. (5.1)), and each update as two
  accumulation steps of 16 recordings × 8 windows (`trainer.batch_size=16`,
  `trainer.update_freq=2`, `mil.window_batch_size=128`), so the windows per update stay P4's 256.

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
step do not fit a 24 GB GPU. At 128 windows per accumulation step REVE takes 15.9 GB and runs at
235 windows per second, in the window phase and in the bag phase.

**Decision rule.** Paired comparison with P4_L1 fold 0 on the same 7,612 test recordings
(`scripts/paired_bag_mae.py`) and on the external cohorts (`scripts/eval_age_by_cohort.py`). A
difference counts when its bootstrap interval excludes 0 or it exceeds 0.38 years, the
single-pair threshold of `labram/eval/research_summary.py`. An encoder replaces LaBraM only if it
wins on Harvard test and does not lose on LEMON and Dortmund; 5-fold cross-validation and the
ensemble follow as a separate step.

**Pre-flight.**
- The P4 recipe ran end to end on a GPU with the released CBraMod and REVE weights under mixed
  precision (synthetic recordings).
- The `CB4_L1` and `RV4_L1` configurations each ran locally on the real cohort and fold 0 in debug
  mode (a window epoch, a joint bag epoch, validation and test scoring), `RV4_L1` with its two
  accumulation steps.
- The dry runs stage the weights as the `pretrained` channel from
  `s3://eeg-data-public/models/cbramod/pretrained_weights.pth` and
  `s3://eeg-data-public/models/reve/reve-base/model.safetensors`.

### Table 7.2. Stage B runs

Both started on 2026-11-24 from commit `d3da8b4` and run P4's schedule of 17 epochs (8 window
epochs, 1 epoch that trains the pool only, 8 joint epochs). P4_L1 fold 0 took 12.8 hours on
`ml.g5.4xlarge`.

| Run | Where | Identifier | Duration |
|---|---|---|---|
| `CB4_L1` fold 0 | local A10G, `scripts/mil_runs/run_local_fm.sh` | `/data/runs/heedb/fm_encoders/heedb_mil_CB4_L1_fold0_local_cv5/fold_0`; ClearML [epochs 0–4](https://app.clear.ml/projects/41a5515610ed449d9831ac7c2c8020c7/experiments/13a62dc1bc8049669b79af59923e0cfb), [epochs 5–16](https://app.clear.ml/projects/41a5515610ed449d9831ac7c2c8020c7/experiments/9ebe43f858b64e478889373a45648ce8) | 59 min per window epoch; expected end 25 Nov, about 20:00 UTC |
| `RV4_L1` fold 0 | SageMaker, on-demand `ml.g5.4xlarge` | `labram-mil-rv4-l1-f0-fold-0-2026-11-24-22-58-36-692` | 3.2 h per window epoch; expected end 26 Nov, about 18:00 UTC (cap 72 h) |

The CBraMod run was stopped from outside on 25 Nov at 04:19 UTC, in epoch 5: the nightly Claude
Code update restarted the service the run had been started from, and systemd stopped the run
with it. It was continued at 11:07 UTC from the epoch-4 checkpoint (`output.auto_resume=true`,
optimiser and schedule restored) as its own systemd unit, which that update does not touch.

### Table 7.3. Interim results at the same epoch

Recording-level MAE in years (Eq. (6.4)) as each run logs it after an epoch (`val_mae`,
`test_mae`: fold-0 validation recordings and the 7,612 test recordings, first 5 minutes; LEMON
and TDBRAIN as the runs' standing test sets). All epochs shown are window-level epochs of the
first phase. **These are not final results:** P4 reached its 6.59 / 6.06 only at the end of the
joint phase. "–" marks an epoch a run has not reached or a set it did not score at that epoch.

| Epoch | P4 val / test | CBraMod val / test | REVE val / test | CBraMod LEMON, TDBRAIN | REVE LEMON, TDBRAIN |
|---|---|---|---|---|---|
| 0 | 9.52 / 9.48 | 13.69 / 13.86 | 8.85 / 8.68 | 14.77, 10.77 | 8.29, 7.83 |
| 1 | 8.85 / 8.84 | 11.56 / 11.73 | 9.00 / 8.97 | 13.99, 9.74 | – |
| 2 | 8.60 / 8.50 | 10.88 / 11.07 | 8.02 / 7.61 | 13.23, 9.04 | 6.75, 6.45 |
| 3 | 8.59 / 8.54 | 11.12 / 11.38 | – | – | – |
| 4 | 8.67 / 8.64 | 10.42 / 10.51 | – | 11.48, 8.26 | – |

Reading Table 7.3, with the caution that the runs are a third of the way or less:

- **REVE is ahead of LaBraM at the same epoch**, by 0.6 years on validation and 0.9 on test after
  epoch 2.
- **CBraMod is about 2 years behind**, and after five epochs still worse than its own frozen
  ridge probe (9.92 on validation, Table 6.3). The P4 recipe as it stands does not suit it.
  Candidate reasons, none tested yet: the readout averages over channels (the per-channel readout
  was 0.6 to 1.5 years better in Table 6.2), the ±100 µV clip removes amplitude a fine-tuned model
  could use, and CBraMod's own recipe uses a learning rate of 1e-4 without layer decay.

### 7.1 Training only the last layers of REVE

REVE-base has 12 times LaBraM's parameters (Table 1.1), so training only its top layers is the
natural way to cut its cost. CBraMod is smaller than LaBraM and fits too slowly already, so the
option is not planned for it.

### Table 7.4. REVE-base: speed and memory against the number of layers trained

Windows per second for forward, backward and optimiser step at 256 windows of 19 channels × 10 s
per step, mixed precision, one A10G, compute only. The layers below the trained ones and the
input stem keep the released weights and need no backward pass. "All 22" does not fit at 256
windows and is measured at 128.

| Layers trained | Trainable parameters | Windows per second | GPU memory | Relative to all 22 |
|---|---|---|---|---|
| all 22 | 69.2 M | 234 | 15.9 GB (128 per step) | 1.0 |
| last 8 | 25.1 M | 444 | 11.4 GB | 1.9 |
| last 6 | 18.8 M | 503 | 8.7 GB | 2.1 |
| last 4 | 12.6 M | 579 | 6.2 GB | 2.5 |
| last 2 | 6.3 M | 683 | 3.6 GB | 2.9 |
| none (head and readout norm only) | 1,537 | 819 | 1.6 GB | 3.5 |

- **Gain.** With the last 6 layers a fold takes about 20 hours instead of 42 and runs at P4's
  256 windows per step without accumulation (Table 7.4), which makes a 5-fold study affordable.
- **Risk.** Frozen features are weak for age (Table 6.3), and an earlier LaBraM run on TUAB with
  only the last block trained reached 13.6 years on test against 8.3 for the full fine-tune
  (`docs/age_training_scenario_D.md`, Section 7; other settings differed too). The full REVE run
  is already a soft version of the same idea: by Eq. (5.1) with $\gamma = 0.8$, the layers below
  the top 8 train at under 17 % of the learning rate.
- **Test.** `RV4T6_L1` = `RV4_L1` with `model.trainable_prefixes` set to the last 6 layers
  (`backbone.net.transformer.layers.16.` to `.21.`), the readout norm, the head and the pool, and
  P4's batch without accumulation. It is compared paired with the full `RV4_L1` run on fold 0;
  within 0.38 years, the partial fine-tune is used for the 5 folds. The job is defined in
  `scripts/mil_runs/submit_grid.sh`; its dry run and a local debug run on the real fold pass
  (18.9 M trainable parameters). It is not started.

**Status.** `CB4_L1` and `RV4_L1` are in progress (Table 7.2), with interim results in Table 7.3;
`RV4T6_L1` is prepared.

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

# stage B (Section 7): on SageMaker, or on the local GPU
ONLY=RV4_L1 FOLD=0 SPOT=false scripts/mil_runs/submit_grid.sh --dry_run
ONLY=RV4_L1 FOLD=0 SPOT=false scripts/mil_runs/submit_grid.sh
scripts/mil_runs/run_local_fm.sh                             # CBraMod, fold 0
scripts/mil_runs/run_local_fm.sh output.auto_resume=true     # continue after an interruption
# from a Claude Code session, start a long local run as its own unit (see the script's header):
systemd-run --user --unit=labram-cb4-l1-f0 --collect --working-directory=$PWD /bin/bash -c \
    'scripts/mil_runs/run_local_fm.sh >> /data/runs/heedb/fm_encoders/CB4_L1_fold0.log 2>&1'
ONLY=RV4T6_L1 FOLD=0 SPOT=false INSTANCE=ml.g5.2xlarge WORKERS=10 scripts/mil_runs/submit_grid.sh --dry_run

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
