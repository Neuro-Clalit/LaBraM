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
- **Fine-tune (stage B): finished, six runs on P4's fold 0** (Tables 7.12 and 7.14 give every
  run per cohort). Result in one line: REVE with its authors' fine-tuning protocol is the first
  encoder that passes the decision rule against LaBraM; CBraMod does not.
  - **REVE with its authors' protocol (`RV5_L1`) passes the decision rule on this fold.** Test
    MAE 5.81 years against 6.06 for P4, paired −0.24 years (95 % interval −0.33 to −0.16), all
    of it on the Harvard recordings. It is level with P4 on LEMON (+0.09), Dortmund (+0.04),
    TDBRAIN (−0.11) and CHBP (+0.48), each with an interval that includes 0, and 1.25 years worse
    on SRM (Table 7.18).
  - **REVE with our recipe and stochastic depth (`RV4DP_L1`) is level with P4 in the training
    domain and behind outside.** Test −0.11 years; Dortmund +0.55, TDBRAIN +0.21, SRM +1.97
    (Table 7.15). Trained without stochastic depth it over-fitted early and was stopped
    (`RV4_L1`); with only its last 6 layers trained it is 0.57 years behind P4 (`RV4T6_L1`,
    Table 7.10).
  - **CBraMod does not replace LaBraM with either recipe.** With the P4 recipe it under-fits
    (test 8.19, `CB4_L1`, Table 7.5); with its authors' settings it reaches 6.72, still 0.66
    years behind P4 and behind on every external cohort (`CB5_L1`, Table 7.13).
  - **As one more member, a REVE model improves the P4 ensemble.** With equal weight per model,
    `RV5_L1` lowers the MAE of the 5-fold P4 ensemble on every cohort: by 0.12 years on test,
    0.08 on LEMON, 0.15 on Dortmund, 0.17 on TDBRAIN, 0.25 on SRM and 0.40 on CHBP
    (Table 7.17). A second LaBraM fold adds less (Table 7.16).
  - **Cost.** A REVE fold takes about 39 hours on one A10G against 13 for P4, and the model has
    69.3 M weights against 5.8 M.
  - **Not shown yet.** One fold only: whether REVE is better on the other folds, and how a
    5-fold REVE ensemble compares with the 5-fold P4 ensemble, needs the other four folds
    (Section 7.9, next steps).

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

### 1.1 Published downstream results against LaBraM

Both papers evaluate by fine-tuning on short-window classification tasks; neither has an age
task or any recording-level target. The CBraMod paper covers 12 datasets in 10 task types and one
more motor-imagery set in its appendix; the REVE paper covers 9 of the same datasets with the
same splits (Table 1.2).

### Table 1.2. Results of the two papers against LaBraM-base

Balanced accuracy on the test split after fine-tuning, mean over the authors' runs; for the one
regression task, Pearson correlation / $R^2$. Read from the papers (CBraMod: Tables 2, 3, 7–15;
REVE: Table 2). "–": not evaluated.

| Task | Dataset | LaBraM-base | CBraMod | REVE-base |
|---|---|---|---|---|
| Abnormal EEG detection | TUAB | 0.8140 | 0.8289 | 0.8315 |
| Event type classification | TUEV | 0.6409 | 0.6671 | 0.6759 |
| Motor imagery | PhysioNet-MI | 0.6173 | 0.6417 | 0.6480 |
| Motor imagery | BCIC-IV-2a | 0.4869 | 0.5138 | 0.6396 |
| Motor imagery | SHU-MI | 0.6166 | 0.6370 | – |
| Emotion recognition | FACED | 0.5273 | 0.5509 | 0.5646 |
| Emotion recognition | SEED-V | 0.3976 | 0.4091 | – |
| Sleep staging | ISRUC | 0.7633 | 0.7865 | 0.7819 |
| Seizure detection | CHB-MIT | 0.7075 | 0.7398 | – |
| Depression diagnosis | Mumtaz2016 | 0.9409 | 0.9560 | 0.9644 |
| Mental stress detection | MentalArithmetic | 0.6909 | 0.7256 | 0.7660 |
| Imagined speech | BCIC2020-3 | 0.5060 | 0.5373 | 0.5635 |
| Vigilance estimation (regression) | SEED-VIG | 0.593 / 0.243 | 0.646 / 0.337 | – |
| Mean of REVE's 9 datasets | | 0.6653 | 0.6898 | 0.7150 |

Reading Table 1.2:

- **In their own papers both models beat LaBraM-base on every dataset.** CBraMod gains 1.2 to 3.5
  points of balanced accuracy, REVE 1.8 to 15 points (BCIC-IV-2a); REVE is above CBraMod on 8 of
  9 datasets and 0.5 points below it on ISRUC, where its preprocessing differs.
- **The larger LaBraM models close most of the gap on the two Temple tasks.** LaBraM-huge
  (369 M parameters) reaches 0.8258 on TUAB and 0.6616 on TUEV, within 0.6 points of CBraMod and
  1.4 of REVE.
- **The comparison is not independent.** Apart from TUAB and TUEV, every LaBraM number in
  Table 1.2 comes from the CBraMod authors' own fine-tuning of the released LaBraM-base weights,
  and the REVE paper copies those numbers.
- **Frozen encoders.** On PhysioNet-MI with the encoder frozen, REVE-base reaches 0.537, against
  0.385 for CBraMod and 0.372 for LaBraM (REVE Table 3; CBraMod Table 18).

Independent benchmarks from 2025 and 2026 do not confirm a uniform win. Their table values below
were read from the papers' web pages by an automated reader and are **not checked cell by cell**;
check them against the source before citing.

- **EEG-FM-Bench** (arXiv:2508.17742, 14 datasets, one code base). With full fine-tuning per
  dataset, CBraMod is below LaBraM on PhysioNet-MI and TUAB (78.3 against 79.5), and REVE is below
  it on PhysioNet-MI (27.9 against 57.3) and one more dataset; on the other datasets both are
  above. With a frozen encoder, CBraMod is below LaBraM on 10 of 14 datasets and REVE on 3,
  among them TUAB (63.8 against 75.9).
- **OmniEEG-Bench** (arXiv:2606.00815, 54 datasets). Under full fine-tuning CBraMod has the best
  average rank (4.51) and LaBraM the second (4.88); REVE is not in the top three and is below
  LaBraM on 19 datasets, CBraMod on 18. With a frozen encoder, CBraMod and REVE rank second and
  third. On age groups of LEMON (4 classes), REVE is above LaBraM when frozen (0.50 against
  0.44) and below it when fine-tuned (0.46 against 0.52).
- **NeuroAtlas** (arXiv:2605.14698) has the only age regression: frozen encoders with a ridge
  probe on 10 cohorts, most of them sleep studies. CBraMod has a higher MAE than LaBraM on all 10
  cohorts; REVE has a lower MAE on 8 of 10 (for example 6.70 against 7.00 years on SHHS). A
  model on handcrafted sleep features is better than every foundation model (4.72 years on
  SHHS). The authors conclude that "pretrained models perform largely on par, with only narrow
  advantages for a few".

For brain age this agrees with our own results: frozen, REVE is slightly better than LaBraM and
CBraMod is level with it (Table 6.3); fine-tuned, neither has beaten LaBraM so far (Tables 7.5
and 7.10).

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
  `model.encoder` holds the options only these models read: `readout` (`mean`, `attn` for
  REVE's pretrained pooling, or `flatten` for CBraMod's all-token readout, Eq. (7.3)),
  `input_norm`, `clip_sd`, `drop_path` (stochastic depth for REVE, Eq. (7.1)), and the options of
  the authors' recipes (Section 7.5): `readout_dropout`, `lr_scale` (Eq. (7.2)), `probe_epochs`
  and `position_noise`. All are off by default.
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
loading, input options, factory, and the options of the authors' recipes) and
`tests/test_fm_finetune.py` (window-level run and the P4 recipe end to end with both encoders,
with our settings and with the authors', then re-scoring the run from its folder; a head-only
epoch leaves the encoder at the released weights).

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
- **`RV4T6_L1`** = `RV4_L1` with only the last 6 of the 22 layers, the readout norm, the head and
  the pool trained (`model.trainable_prefixes`), at P4's batch without accumulation (Section 7.2).
- **`RV4DP_L1`** = `RV4_L1` with stochastic depth, `model.encoder.drop_path=0.1` (Section 7.3).
- **`CB5_L1`** = `P4_L1` with CBraMod trained the way its authors fine-tune it, and
  **`RV5_L1`** = `P4_L1` with REVE and its authors' two-stage protocol (Section 7.5, Table 7.9).

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

All on fold 0 with P4's schedule of 17 epochs (8 window epochs, 1 epoch that trains the pool only,
8 joint epochs). P4_L1 fold 0 took 12.8 hours on `ml.g5.4xlarge`.

| Run | Where | Identifier | State |
|---|---|---|---|
| `CB4_L1` | local A10G, `scripts/mil_runs/run_local_fm.sh` | `/data/runs/heedb/fm_encoders/heedb_mil_CB4_L1_fold0_local_cv5/fold_0`; ClearML [epochs 0–4](https://app.clear.ml/projects/41a5515610ed449d9831ac7c2c8020c7/experiments/13a62dc1bc8049669b79af59923e0cfb), [epochs 5–16](https://app.clear.ml/projects/41a5515610ed449d9831ac7c2c8020c7/experiments/9ebe43f858b64e478889373a45648ce8) | finished 25 Nov, 18:44 UTC (59 min per window epoch, 33 min per joint epoch) |
| `RV4_L1` | SageMaker, on-demand `ml.g5.4xlarge` | `labram-mil-rv4-l1-f0-fold-0-2026-11-24-22-58-36-692` | stopped on request on 25 Nov in epoch 6, after 21.3 h (3.2 h per window epoch); epochs 0–5 are scored |
| `RV4T6_L1` | SageMaker, on-demand `ml.g5.4xlarge` | `labram-mil-rv4t6-l1-f0-fold-0-2026-11-25-21-19-50-438`; results in `/data/runs/heedb/fm_encoders/RV4T6_L1/fold_0` | ended 26 Nov, 18:29 UTC after 20.7 h, killed for host memory while scoring epoch 15 of 0–16; best epoch 10 (Table 7.10) |
| `RV4DP_L1` | local A10G, systemd unit `labram-rv4dp-l1-f0` | `/data/runs/heedb/fm_encoders/heedb_mil_RV4DP_L1_fold0_local_cv5/fold_0`; [ClearML](https://app.clear.ml/projects/e76a506be53e426aa0d95e90ad79855c/experiments/b2b8c98e12c044a59710f37372e5815c) | finished 27 Nov, 14:15 UTC after 40.8 h (2.9 h per window epoch, 1.9 h per joint epoch); best epoch 12 (Tables 7.7 and 7.12) |
| `CB5_L1` | SageMaker, on-demand `ml.g5.4xlarge`, 24 loader workers | `labram-mil-cb5-l1-f0-fold-0-2026-11-26-19-26-48-628`; ClearML task `heedb_mil_CB5_L1_fold0` | finished: all 17 epochs trained and scored by 27 Nov, 08:30 UTC (59 min per window epoch, 33 min per joint epoch); the job then spent 2.7 h in SageMaker's final upload, 15.7 h billed; results in `/data/runs/heedb/fm_encoders/CB5_L1/fold_0` (Table 7.13). A first job on `ml.g5.2xlarge` (`…-12-51-19-517`) was killed for host memory after 1.3 h, before the end of epoch 0 |
| `RV5_L1` | SageMaker, on-demand `ml.g5.8xlarge`, for epochs 1–6; then local A10G, systemd unit `labram-rv5-l1-f0` | SageMaker `labram-mil-rv5-l1-f0-fold-0-2026-11-26-19-05-37-271` (ClearML task `heedb_mil_RV5_L1_fold0`); local `/data/runs/heedb/fm_encoders/heedb_mil_RV5_L1_fold0_local_cv5/fold_0` (ClearML task `heedb_mil_RV5_L1_fold0_local`), log `/data/runs/heedb/fm_encoders/RV5_L1_fold0.log` | finished 28 Nov, 08:34 UTC; best epoch 10 (Table 7.18). About 39 h of training in total: the head-only epoch 0 on `ml.g5.2xlarge` (1.6 h), epochs 1–6 on `ml.g5.8xlarge` (19.4 h billed), epochs 7–16 on the local GPU (18.1 h; moved at the start of epoch 7 from the SageMaker job's epoch-6 checkpoint with `scripts/mil_runs/resume_local_from_sagemaker.sh`, same step time on both). The first job on `ml.g5.2xlarge` (`…-12-51-36-446`) was killed for host memory in epoch 1 |

- The CBraMod run was stopped from outside on 25 Nov at 04:19 UTC, in epoch 5: the nightly Claude
  Code update restarted the service the run had been started from, and systemd stopped the run
  with it. It was continued at 11:07 UTC from the epoch-4 checkpoint (`output.auto_resume=true`,
  optimiser and schedule restored) as its own systemd unit, which that update does not touch.
- **Three SageMaker jobs were killed for host memory on 26 Nov** (`Killed` in the log, failure
  reason "use an instance type with more memory"): the first `CB5_L1` and `RV5_L1` jobs on
  `ml.g5.2xlarge` (32 GB, 10 loader workers) after 1.3 and 4.1 hours of training, and `RV4T6_L1`
  on `ml.g5.4xlarge` (64 GB, 32 workers) after 20.7 hours. Reported host memory rose from 80 % to
  100 % within an epoch on the 32 GB instance and fell back at the epoch's end. What holds the
  memory inside the container is not established; the same run uses 16 GB on the local machine
  with 10 workers. P4 fold 0 (12.8 hours) and the stopped `RV4_L1` (21.3 hours) ran on 64 GB
  with 32 workers without it. Consequences: `ml.g5.2xlarge` is not used for this cohort
  any more; `RV5_L1` continues on 128 GB with 16 workers, `CB5_L1` on 64 GB with 24 workers. A
  killed job loses the epoch in progress only: its checkpoints stay on S3 and
  `RESUME=<checkpoint folder>` continues it.
- The stopped `RV4_L1` job keeps its checkpoints under
  `s3://sagemaker-us-east-1-574441342949/labram/checkpoints/labram-mil-rv4-l1-f0-fold-0-20261124T225836/`
  and can be continued with `RESUME=<that folder> ONLY=RV4_L1 FOLD=0 scripts/mil_runs/submit_grid.sh`.

### Table 7.3. Results at the same epoch

Recording-level MAE in years (Eq. (6.4)) as each run logs it after an epoch (`val_mae`,
`test_mae`: fold-0 validation recordings and the 7,612 test recordings, first 5 minutes; LEMON
and TDBRAIN as the runs' standing test sets, scored at some epochs only). Epochs 0–7 are
window-level epochs, epoch 8 trains the pool only, epochs 9–16 are joint. P4 and CBraMod are
complete; the REVE column is the full fine-tune `RV4_L1`, which was stopped after epoch 5, so
**it has no final result**. "–" marks an epoch a run did not reach or a set it did not score at
that epoch.

| Epoch | P4 val / test | CBraMod val / test | REVE val / test | CBraMod LEMON, TDBRAIN | REVE LEMON, TDBRAIN |
|---|---|---|---|---|---|
| 0 | 9.52 / 9.48 | 13.69 / 13.86 | 8.85 / 8.68 | 14.77, 10.77 | 8.29, 7.83 |
| 1 | 8.85 / 8.84 | 11.56 / 11.73 | 9.00 / 8.97 | 13.99, 9.74 | – |
| 2 | 8.60 / 8.50 | 10.88 / 11.07 | 8.02 / 7.61 | 13.23, 9.04 | 6.75, 6.45 |
| 3 | 8.59 / 8.54 | 11.12 / 11.38 | 7.72 / 7.40 | – | 6.68, 6.28 |
| 4 | 8.67 / 8.64 | 10.42 / 10.51 | 8.25 / 8.05 | 11.48, 8.26 | 8.73, 6.68 |
| 5 | 7.92 / 7.73 | 9.61 / 9.52 | 7.87 / 7.57 | 9.51, 7.93 | – |
| 6 | 7.79 / 7.56 | 9.90 / 9.85 | – | – | – |
| 7 | 7.69 / 7.49 | 10.05 / 10.12 | – | – | – |
| 8 | 7.05 / 6.67 | 9.33 / 9.07 | – | 10.49, 7.89 | – |
| 9 | 6.71 / 6.22 | 8.73 / 8.28 | – | 8.18, 7.26 | – |
| 10 | 6.69 / 6.19 | **8.56 / 8.19** | – | 7.39, 7.03 | – |
| 11 | 6.68 / 6.22 | 8.88 / 8.45 | – | – | – |
| 12 | 6.65 / 6.12 | 8.80 / 8.45 | – | – | – |
| 13 | 6.63 / 6.09 | 8.74 / 8.34 | – | – | – |
| 14 | 6.61 / 6.07 | 8.90 / 8.54 | – | 8.87, 7.15 | – |
| 15 | **6.59 / 6.06** | 8.87 / 8.49 | – | – | – |
| 16 | 6.60 / 6.07 | 8.85 / 8.47 | – | 8.71, 7.13 | – |

**Bold**: the best validation epoch of a finished run, the one its result is taken from.

Reading Table 7.3:

- **REVE was ahead of LaBraM at every epoch it reached**: by 0.4 to 0.9 years on validation at
  epochs 2 to 4, and by 0.05 at epoch 5, where P4 made its largest step. Its curve was noisy at
  the peak learning rate, and it never left the window phase, so its final gain is not known.
- **CBraMod stays about 2 years behind at every epoch**, in the window phase and in the joint
  phase.

### 7.1 Result: CBraMod with the P4 recipe

### Table 7.5. CBraMod against P4 on fold 0

Each run at its best validation epoch (CBraMod epoch 10, P4 epoch 15). MAE in years (Eq. (6.4))
per recording; Δ = CBraMod − P4 on the same recordings (Eq. (6.5)) with its 95 % interval from
2,000 bootstrap resamples, positive when CBraMod is worse. Test: the first 5 minutes of the 7,612
held-out recordings (P4 re-scored with `scripts/eval_age_by_cohort.py`, CBraMod from its test
bag-weight file). External cohorts: adults, every window, no trims for Dortmund, TDBRAIN and SRM,
both runs scored with `scripts/eval_age_by_cohort.py`. Validation is each run's logged value.

| Cohort | $n$ | P4 MAE | CBraMod MAE | Δ [95 % interval] |
|---|---|---|---|---|
| Validation (fold 0) | 19,071 | 6.59 | 8.56 | +1.97 |
| Test, all | 7,612 | 6.06 | 8.19 | +2.14 [+2.01, +2.27] |
| Test, Harvard | 7,338 | 6.05 | 8.20 | +2.15 [+2.03, +2.29] |
| Test, Temple | 274 | 6.32 | 8.06 | +1.74 [+1.08, +2.38] |
| LEMON | 214 | 4.73 | 7.44 | +2.71 [+2.05, +3.39] |
| Dortmund | 3,262 | 5.29 | 6.96 | +1.68 [+1.50, +1.86] |
| TDBRAIN | 2,222 | 5.48 | 7.03 | +1.55 [+1.31, +1.79] |
| SRM | 152 | 5.01 | 6.39 | +1.38 [+0.50, +2.21] |
| CHBP | 250 | 7.45 | 10.65 | +3.20 [+2.10, +4.33] |

- **Verdict.** By the decision rule CBraMod with the P4 recipe does not replace LaBraM: it is worse
  on Harvard test, on LEMON and on Dortmund, and on every other cohort, with every interval far
  from 0 (Table 7.5).
- **It under-fits.** At the last epoch its MAE on the scored sample of training recordings is 8.32
  years, against 8.85 on validation. A model that cannot fit its training data is limited by what
  it is given or how it is trained, not by generalisation. The frozen probe pointed the other way
  for the external cohorts (Table 6.4), so the encoder is not shown to be worse; this recipe is.
- **Untested candidates for the cause:** the readout averages over channels (the per-channel
  readout was 0.6 to 1.5 years better in Table 6.2); the ±100 µV clip removes amplitude a
  fine-tuned model could use; CBraMod's own recipe uses a learning rate of 1e-4 without layer
  decay. A second run with these changes is not started.

### 7.2 Training only the last layers of REVE

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
  (18.9 M trainable parameters). It was started on 25 Nov (Table 7.2). With the full run stopped,
  it is compared with P4 directly, and with `RV4_L1` at epochs 0 to 5.

### 7.3 Over-fitting and under-fitting: why the two encoders need opposite changes

Every fourth epoch a run scores a fixed sample of its training recordings like the validation
set (`train_eval_mae`: the first 5 minutes, no mixup). The difference to the validation MAE shows
whether a model memorises its training data or cannot fit it.

### Table 7.6. Error on training recordings against validation

MAE in years (Eq. (6.4)) on the scored sample of training recordings and on the fold-0 validation
recordings, at the same epoch of a run; "gap" is validation minus training.

| Run | Epoch | Training recordings | Validation | Gap |
|---|---|---|---|---|
| P4 (LaBraM) | 3 | 7.56 | 8.59 | 1.0 |
| `RV4_L1` (REVE, all layers) | 3 | 5.53 | 7.72 | 2.2 |
| `RV4DP_L1` (REVE, all layers + stochastic depth) | 3 | 5.86 | 7.67 | 1.8 |
| `RV5_L1` (REVE, authors' protocol) | 3 | 6.46 | 8.07 | 1.6 |
| `CB5_L1` (CBraMod, authors' settings) | 3 | 7.41 | 8.61 | 1.2 |
| P4 (LaBraM) | 7 (end of window phase) | 5.84 | 7.69 | 1.9 |
| `CB4_L1` (CBraMod) | 7 (end of window phase) | 9.01 | 10.05 | 1.0 |
| `CB5_L1` (CBraMod, authors' settings) | 7 (end of window phase) | 6.05 | 8.07 | 2.0 |
| `RV4T6_L1` (REVE, last 6 layers) | 7 (end of window phase) | 4.42 | 8.25 | 3.8 |
| `RV4DP_L1` (REVE, all layers + stochastic depth) | 7 (end of window phase) | 4.47 | 7.63 | 3.2 |
| `RV5_L1` (REVE, authors' protocol) | 7 (end of window phase) | 4.58 | 7.48 | 2.9 |
| P4 (LaBraM) | 11 | 4.27 | 6.68 | 2.4 |
| `CB5_L1` (CBraMod, authors' settings) | 11 | 4.64 | 7.31 | 2.7 |
| `RV4T6_L1` (REVE, last 6 layers) | 11 | 2.28 | 7.30 | 5.0 |
| `RV4DP_L1` (REVE, all layers + stochastic depth) | 11 | 2.79 | 6.61 | 3.8 |
| `RV5_L1` (REVE, authors' protocol) | 11 | 2.54 | 6.48 | 3.9 |
| P4 (LaBraM) | 15 | 3.72 | 6.59 | 2.9 |
| `RV4DP_L1` (REVE, all layers + stochastic depth) | 15 | 2.34 | 6.59 | 4.3 |
| `RV5_L1` (REVE, authors' protocol) | 15 | 1.84 | 6.50 | 4.7 |
| P4 (LaBraM) | 16 (end) | 3.71 | 6.60 | 2.9 |
| `CB4_L1` (CBraMod) | 16 (end) | 8.32 | 8.85 | 0.5 |
| `CB5_L1` (CBraMod, authors' settings) | 16 (end) | 4.16 | 7.26 | 3.1 |

- **REVE over-fits early.** After epoch 3 it fits the training recordings 2 years better than
  LaBraM does at the same epoch, and its gap is already 2.2 years (Table 7.6). With 12 times
  LaBraM's parameters and no dropout or stochastic depth in its released encoder, it has more
  room to memorise. The two runs in progress limit that in different ways:
  - `RV4T6_L1` trains only the last 6 layers (Section 7.2);
  - `RV4DP_L1` keeps all layers trainable and adds stochastic depth: during training each sample
    drops the attention or feed-forward branch of layer $l$ with probability

$$
p_l = 0.1 \cdot \frac{l - 1}{L - 1}, \qquad l = 1, \dots, L = 22
\tag{7.1}
$$

    the rule LaBraM's blocks use (`model.encoder.drop_path`,
    `labram/models/backbones/reve.py::TransformerBackbone`). It adds no weights, so the released
    file still loads strictly and the port stays identical to upstream when the option is off.
- **CBraMod under-fits.** It ends at 8.3 years on its own training recordings (Table 7.6), so
  freezing layers or adding regularisation would not help it. The candidate changes for it go
  the other way (Section 7.1).

### Table 7.7. REVE variants at the same epoch

Validation / test MAE in years (Eq. (6.4)) after each epoch, as in Table 7.3, and
the error on the scored training recordings at the epochs where a run scores them (Table 7.6).
All runs are finished (`RV4T6_L1` ended while scoring epoch 15; `RV4_L1` was stopped). "–": not reached or not scored.
`RV5_L1`'s epoch 0 trains the head only (Section 7.5), so its first row is not comparable.

| Epoch | P4 (LaBraM) | `RV4_L1`: all layers (stopped) | `RV4DP_L1`: all layers + stochastic depth | `RV4T6_L1`: last 6 layers | `RV5_L1`: authors' protocol |
|---|---|---|---|---|---|
| 0 | 9.52 / 9.48 | 8.85 / 8.68 | 9.05 / 8.92 | 10.33 / 10.44 | 13.04 / 13.37 (head only) |
| 1 | 8.85 / 8.84 | 9.00 / 8.97 | 9.10 / 9.06 | 9.21 / 9.10 | 8.51 / 8.30 |
| 2 | 8.60 / 8.50 | 8.02 / 7.61 | 7.76 / 7.45 | 8.95 / 8.87 | 8.54 / 8.47 |
| 3 | 8.59 / 8.54 | 7.72 / 7.40 | 7.67 / 7.39 | 8.55 / 8.40 | 8.07 / 7.95 |
| 4 | 8.67 / 8.64 | 8.25 / 8.05 | 8.08 / 7.89 | 8.74 / 8.61 | 8.33 / 8.20 |
| 5 | 7.92 / 7.73 | 7.87 / 7.57 | 7.59 / 7.31 | 8.61 / 8.38 | 8.61 / 8.56 |
| 6 | 7.79 / 7.56 | – | 7.43 / 7.10 | 8.26 / 8.00 | 7.60 / 7.34 |
| 7 (end of window phase) | 7.69 / 7.49 | – | 7.63 / 7.33 | 8.25 / 8.00 | 7.48 / 7.16 |
| 8 (pool only) | 7.05 / 6.67 | – | 6.91 / 6.44 | 7.58 / 7.12 | 6.87 / 6.37 |
| 9 (first joint epoch) | 6.71 / 6.22 | – | 6.68 / 6.11 | 7.34 / 6.83 | 6.53 / 5.97 |
| 10 | 6.69 / 6.19 | – | 6.57 / 5.96 | **7.17 / 6.63** (best) | **6.43 / 5.81** (best) |
| 11 | 6.68 / 6.22 | – | 6.61 / 6.02 | 7.30 / 6.73 | 6.48 / 5.87 |
| 12 | 6.65 / 6.12 | – | **6.55 / 5.94** (best) | 7.23 / 6.65 | 6.50 / 5.86 |
| 13 | 6.63 / 6.09 | – | 6.59 / 5.98 | 7.25 / 6.63 | 6.49 / 5.85 |
| 14 | 6.61 / 6.07 | – | 6.59 / 5.98 | 7.25 / 6.65 | 6.50 / 5.85 |
| 15 | **6.59 / 6.06** (best) | – | 6.59 / 5.99 | 7.25 / 6.67 | 6.50 / 5.85 |
| 16 (end) | 6.60 / 6.07 | – | 6.59 / 5.99 | not run | 6.51 / 5.86 |
| Training recordings, epoch 3 | 7.56 | 5.53 | 5.86 | 6.76 | 6.46 |
| Training recordings, epoch 7 | 5.84 | – | 4.47 | 4.42 | 4.58 |
| Training recordings, epoch 11 | 4.27 | – | 2.79 | 2.28 | 2.54 |
| Training recordings, epoch 15 | 3.72 | – | 2.34 | – | 1.84 |
| Training recordings, epoch 16 | 3.71 | – | 2.30 | – | 1.84 |

Reading Table 7.7:

- **Training only the last 6 layers is behind LaBraM and over-fits more.** At the end of the
  window phase `RV4T6_L1` is 0.56 years behind P4 on validation and 0.51 on test, while it fits
  its training recordings 1.4 years better (4.42 against 5.84). Its gap between training
  recordings and validation is 3.8 years, against 1.9 for P4. Freezing the lower 16 layers did
  not stop the memorising; it cost generalisation. The bag phase did not close the distance:
  the run stays 0.5 to 0.7 years behind P4 from epoch 8 to its end (Section 7.6).
- **Stochastic depth keeps the full fine-tune's accuracy and narrows the gap a little.** At
  epoch 3 `RV4DP_L1` equals the stopped run on validation and test (7.67 / 7.39 against 7.72 /
  7.40) and is 0.9 years ahead of P4 on validation. Its error on training recordings is 5.86
  against 5.53, so the gap falls from 2.2 to 1.8 years. At epoch 4, where every run has a worse
  epoch, it is 8.08 / 7.89 against 8.25 / 8.05 for the stopped run and 8.67 / 8.64 for P4. At
  epoch 5 it is 7.59 / 7.31, ahead of the stopped run (7.87 / 7.57) and of P4 (7.92 / 7.73).
- **In the bag phase the full fine-tune with stochastic depth is level with P4.** From epoch 8 to 12 `RV4DP_L1` is 0.03 to 0.14 years ahead of P4 on
  validation and 0.11 to 0.23 on test at the same epoch; from epoch 13 to its end it is flat at
  6.59 / 5.98–5.99. At its best epoch (12) it has 6.55 / 5.94 against 6.59 / 6.06 for P4's best epoch.
  Paired on the 7,612 test recordings (Eq. (6.5)) that is −0.11 years with a 95 % interval of
  −0.20 to −0.01 (Harvard −0.11, interval −0.20 to −0.01; Temple −0.15, interval −0.70 to
  +0.40), inside the 0.38-year threshold. On LEMON, as logged during training, it has 5.24
  years at epoch 12 against 4.87 for P4's best epoch. It memorises more: 2.34 years on training
  recordings against 6.59 on validation at epoch 15, where P4 has 3.72 against 6.59
  (Table 7.6).
- **The two models differ enough to help each other.** The mean of `RV4DP_L1`'s epoch-12
  predictions and P4's has a test MAE of 5.75 years, 0.31 years below P4 alone (interval −0.36
  to −0.26), with the two predictions correlated at 0.97. The last-6-layers model and CBraMod
  gave no such gain (Sections 7.5 and 7.6).
- **The authors' protocol starts slower (interim).** At epoch 3 `RV5_L1` has 8.07 / 7.95: 0.5
  years ahead of P4 and 0.4 to 0.6 behind the two full fine-tunes with our recipe. It spent
  epoch 0 on the head alone, so it has had one encoder epoch less; against `RV4DP_L1` at epoch 2
  (7.76 / 7.45) it is still 0.3 to 0.5 behind. Its gap between training recordings and
  validation is the smallest of the REVE runs (1.6 years, Table 7.6). On LEMON it logs 8.24 and
  8.81 years at epochs 1 and 3, where `RV4_L1` had 6.75 and 6.68 at epochs 2 and 3; one possible
  cause is its input, µV / 100 without the per-window z-score (not tested). At epochs 4 and 5,
  at its peak learning rate, its validation error rose to 8.33 and 8.61 while its training loss
  kept falling; at epoch 5 it is 0.7 years behind P4 and 1.0 behind `RV4DP_L1`. At epoch 6 it
  fell to 7.60 / 7.34, 0.2 years ahead of P4 and 0.2 behind `RV4DP_L1` at the same epoch.
- **From the end of the window phase the authors' protocol is ahead of both.** At
  epochs 7 to 10 `RV5_L1` is 0.04 to 0.15 years below `RV4DP_L1` on validation and 0.07 to 0.17
  on test at the same epoch, and 0.18 to 0.26 and 0.25 to 0.38 below P4. At epoch 10 it has
  6.43 / 5.81; paired on the 7,612 test recordings that is −0.24 years against P4's best epoch
  (95 % interval −0.33 to −0.16; Harvard −0.26, Temple +0.15 with an interval of −0.39 to
  +0.66) and −0.13 against `RV4DP_L1`'s best epoch (−0.21 to −0.05). Its LEMON error, high in
  the window phase, fell to 4.97 years at epoch 10 (P4: 4.99 at that epoch, 4.87 at its best),
  and on TDBRAIN adults it logs 5.37 years, below P4's 5.48. Epoch 10 stays its best: epochs
  11 to 16 are flat at 6.48 to 6.51 / 5.85 to 5.87. It memorises the training recordings more
  than any other run (1.84 years at epoch 15 against 6.50 on validation, Table 7.6) and still
  generalises best.

### 7.4 How the authors fine-tune their models, and what follows for ours

Read from the authors' code and papers: CBraMod (`github.com/wjq-learning/CBraMod`, commit
`b9e9610`: `finetune_main.py`, `finetune_trainer.py`, `models/model_for_*.py`; paper Appendices
D.4, E.5, H, J) and REVE (`github.com/elouayas/reve_eeg`, commit `06a7059`: `src/dt.py`,
`src/models/classifier.py`, `src/configs/`; paper Sections 3.3 and 4, Table 4). Both apply one
recipe to all their tasks, and both differ from P4's recipe in the same places.

### Table 7.8. The authors' fine-tuning recipes against the recipe of our runs

"Ours" is the P4 recipe as used in `CB4_L1` and `RV4_L1` (Section 7). For REVE the paper and the
released script disagree on one point: the paper adapts the encoder through LoRA, the script
unfreezes every weight; both are listed.

| | CBraMod authors | REVE authors | Ours |
|---|---|---|---|
| Stages | one: all weights | two: head alone with the encoder frozen (TUAB: up to 20 epochs), then the encoder too | one: all weights from the first step |
| Encoder update | every layer | paper: LoRA of rank 16 on the attention projections; script: every weight | every layer (`RV4T6_L1`: last 6) |
| Learning rate | 1e-4 on every encoder layer, 5e-4 on the head; cosine, no warm-up | head stage 5e-3; then 1e-4 on every layer; 5 warm-up epochs, halved on plateau, early stopping | peak 2.8e-4 with layer-wise decay (0.65 CBraMod, 0.8 REVE), 3 warm-up epochs, cosine |
| Gradient clipping | norm 1 | norm 2 | none |
| Readout | all channel × patch tokens flattened into a 3-layer MLP | all tokens flattened plus an attention-pooled token, or attention pooling with the pretrained query alone; RMSNorm; linear | mean of all tokens, LayerNorm, linear |
| Input | µV / 100; no clip, no z-score | one fixed scale per dataset (µV / 100); no clip, no z-score | CAR; CBraMod: 0.3 Hz high-pass and ±100 µV clip; REVE: z-score per window |
| Regularisation | dropout 0.1, label smoothing; no mixup | mixup with a uniform mixing weight; noise of 0.25 cm on the electrode positions; head dropout; paper: average of at least 5 models (+1.5 %) | mixup with α = 0.4 |
| Epochs, batch | 50, 64 | TUAB: up to 20 + 50, 16 | 17, 256 windows |
| Frozen encoder | "cannot currently serve as a fixed-parameter feature extractor" (Appendix J) | strong: TUAB balanced accuracy 0.810 frozen against 0.832 fine-tuned | – |
| Regression, recording level | one regression task (vigilance, SEED-VIG): same head, MSE loss, $R^2$ 0.34 against 0.24 for LaBraM-base; windows only | none | age; recordings pooled over windows |

What follows for our task:

- **CBraMod was under-trained by our recipe, as its authors' results predict.** With layer decay
  0.65 its input layers train at 1e-6 and its middle layers at 1.4e-5, against 1e-4 everywhere
  in the authors' recipe, and they report that a frozen CBraMod does not work. Our run under-fits
  (Table 7.6). Proposal `CB5_L1`: P4's schedule with the authors' settings: every encoder layer
  at 1e-4, head 5 times higher, gradient clip 1; a readout over all tokens (their two-layer form:
  38,000 → 200, feeding our pool); µV / 100 with the average reference and the 0.3 Hz high-pass
  but no clip; no mixup.
- **REVE's authors restrict how the encoder moves, not which layers move.** They first fit the
  head on the frozen encoder, then adapt all layers gently (1e-4, or LoRA), with gradient
  clipping. Our run adapts every layer from the first step at up to 2.2e-4 in the top layers, and
  it over-fits early; training only the last 6 layers over-fits more (Table 7.7). Proposal
  `RV5_L1`: a head-only first epoch, then every layer at 1e-4 without layer decay, gradient
  clip 2, attention pooling with the pretrained query (`model.encoder.readout=attn`), µV / 100
  input, position noise and uniform mixup. Proposal `RV5L_L1`: the same with LoRA of rank 16 on
  the attention projections in place of full updates.
- **Neither group averages the tokens.** Both keep the channel × patch structure in the readout,
  or use a learned pooling. Our frozen probe agrees: the per-channel readout was 0.6 to 1.5 years
  better than the mean (Tables 6.1 and 6.2).
- **Neither group clips or z-scores the fine-tuning input.** Both feed µV / 100, which keeps the
  absolute amplitude. CBraMod's pretraining removed large-amplitude segments and never clipped
  them, so our ±100 µV clip gives it flat-topped waveforms it has not seen.
- **Averaging models is part of REVE's reported result** (+1.5 % from at least 5 models). For us
  this is the 5-fold ensemble, and the average of a LaBraM and a REVE model.
- **Not covered by either group:** age or any recording-level target, pooling over windows, and a
  labelled set of our size (their largest is TUAB). These parts stay our own design.

`CB5_L1` and `RV5_L1` are implemented and running (Section 7.5). `RV5L_L1` needs LoRA for REVE's
layers, which does not exist yet; it is not started.

### 7.5 Runs with the authors' settings: `CB5_L1` and `RV5_L1`

Both runs keep everything of P4 that the authors' recipes do not cover: the split, the schedule
of 17 epochs with 3 warm-up epochs and a cosine, AdamW with weight decay 0.05, the loss, the bags
and the pool, and the evaluation. They take from the authors the parts of Table 7.8 that differ
from ours: the learning rate per layer, gradient clipping, the readout, the input scale and the
regularisation. The authors' epoch counts, batch sizes and early stopping are not taken over.

**Learning rate per layer** (`model.encoder.lr_scale`, `labram/optim_factory.py::layer_scale_values`).
With the peak rate $\eta$ (`optimizer.lr`), the encoder scale $\rho$ and the layer decay $\gamma$
of Eq. (5.1), a parameter of layer $i$ trains at the peak rate

$$
\eta_i =
\begin{cases}
\eta \, \rho \, \gamma^{\,L + 1 - i}, & i = 0, \dots, L \quad \text{(input stem and encoder layers)} \\
\eta, & i = L + 1 \quad \text{(readout, head, pool)}
\end{cases}
\tag{7.2}
$$

Both runs use $\eta = 5 \times 10^{-4}$, $\rho = 0.2$ and $\gamma = 1$: every encoder layer at
$10^{-4}$, the readout and the head at $5 \times 10^{-4}$, the authors' ratio. $\rho = 1$ gives
Eq. (5.1), so runs without the option are unchanged. In the bag phase the pool and the head are
multiplied by `mil.head_lr_scale` $= 2.8$, which gives P4's rate of
$2.8 \times 5 \times 10^{-4} = 1.4 \times 10^{-3}$.

**All-token readout** (`model.encoder.readout=flatten`, CBraMod). The encoder returns one token per
channel and second, $Z \in \mathbb{R}^{19 \times 10 \times 200}$. Instead of their mean, the window
feature is

$$
h = \mathrm{LN}\big(\mathrm{Dropout}_{0.1}\big(\mathrm{ELU}(W \, \mathrm{vec}(Z) + b)\big)\big),
\qquad W \in \mathbb{R}^{200 \times 38{,}000}
\tag{7.3}
$$

the first layer of the authors' head, followed by our readout norm $\mathrm{LN}$; $h$ then feeds
the head (window phase) and the pool (bag phase) as before. $W$ adds 7.6 M weights to CBraMod's
4.9 M. The tokens are always those of the full 19-channel layout in its fixed order, so the
feature does not depend on the channel order of the input, and a missing channel is a flat slot.

**Head-only first epoch** (`model.encoder.probe_epochs=1`, REVE). In epoch 0 the encoder is frozen
at the released weights; the pooling query, the readout norm and the head train. From epoch 1 all
weights train. A head-only step took 0.09 s against 0.22 s for a full step in the debug run
(16 windows per step, GPU shared with another job).

**Position noise** (`model.encoder.position_noise=0.0025`, REVE). In training mode each forward
adds one draw of Gaussian noise with a standard deviation of 0.25 cm to the electrode
coordinates, shared by the batch, as the authors' encoder does. Evaluation uses the exact
positions.

### Table 7.9. Settings of `CB5_L1` and `RV5_L1` against the first runs

Learning rates are peak values; all follow P4's warm-up and cosine. Layer $i$ as in Eq. (5.1).

| Setting | `CB4_L1` | `CB5_L1` | `RV4_L1` | `RV5_L1` |
|---|---|---|---|---|
| Encoder learning rate | $2.8 \times 10^{-4} \cdot 0.65^{13 - i}$: 1.0e-6 (stem) to 1.8e-4 (layer 12) | 1e-4 on every layer (Eq. (7.2)) | $2.8 \times 10^{-4} \cdot 0.8^{23 - i}$: 1.7e-6 (stem) to 2.2e-4 (layer 22) | 1e-4 on every layer (Eq. (7.2)) |
| Readout and head, window phase | 2.8e-4 | 5e-4 | 2.8e-4 | 5e-4 |
| Pool and head, bag phase | 1.4e-3 | 1.4e-3 | 1.4e-3 | 1.4e-3 |
| Epoch 0 | all weights | all weights | all weights | pooling query, readout norm and head only |
| Gradient clipping | none | norm 1 | none | norm 2 |
| Readout | mean of all tokens | all 190 tokens, linear + ELU + dropout 0.1 (Eq. (7.3)) | mean of all tokens | attention pooling with the pretrained query |
| Input after the average reference | 0.3 Hz high-pass, ±100 µV clip | 0.3 Hz high-pass, no clip | z-score per window | µV / 100, no z-score |
| Mixup | α = 0.4 | none | α = 0.4 | α = 1 (uniform mixing weight) |
| Noise on electrode positions | – | – | none | 0.25 cm |
| Weights trained | 4.9 M | 4.9 M + 7.6 M (readout) | 69.2 M | 69.2 M |
| Windows per update | 256 | 256 | 2 × 128 | 2 × 128 |

**Pre-flight.**
- Unit tests cover each option: the all-token readout is tied to the layout and not to the input's
  channel order; Eq. (7.2) reaches the optimiser's parameter groups; after a head-only epoch the
  saved encoder equals the released weights tensor by tensor (REVE: all but the pooling query),
  and it differs after the next epoch; position noise is applied in training mode only.
- Each configuration ran locally on the real cohort and fold 0 in debug mode with the released
  weights under mixed precision: `CB5_L1` through a window epoch and a joint bag epoch, `RV5_L1`
  through the head-only epoch, a released window epoch and a joint bag epoch, each with
  validation and test scoring.
- The dry runs stage the same weight files as `CB4_L1` and `RV4_L1`.

**Status.** `CB4_L1` is finished (Table 7.5). `RV4_L1` was stopped after epoch 5 (Table 7.3).
`RV4T6_L1` is finished (Section 7.6). `RV4DP_L1` is in progress (Tables 7.2 and 7.7). `CB5_L1`
and `RV5_L1` were restarted on 26 Nov after their first jobs were killed for host memory
(Table 7.2). `CB5_L1` is finished (Tables 7.11 and 7.13) and so is `RV5_L1` (Tables 7.7 and 7.18),
which ran on the local GPU from epoch 7 (Table 7.2). Neither showed a numerical problem under
mixed precision; reported host memory stayed at 74 to 83 % of 64 GB for `CB5_L1` and at 26 to
32 % of 128 GB for `RV5_L1`.

### Table 7.11. CBraMod with its authors' settings at the same epoch

Validation / test MAE in years (Eq. (6.4)) after each epoch, as in Table 7.3; LEMON and TDBRAIN
as `CB5_L1` logs them during training at the epochs where it scores them; the last rows are the
error on the scored training recordings (Table 7.6). All three runs are complete. "–": not
scored at that epoch.

| Epoch | P4 (LaBraM) | `CB4_L1`: P4 recipe | `CB5_L1`: authors' settings | `CB5_L1` LEMON, TDBRAIN |
|---|---|---|---|---|
| 0 | 9.52 / 9.48 | 13.69 / 13.86 | 9.82 / 9.71 | 9.04, 7.85 |
| 1 | 8.85 / 8.84 | 11.56 / 11.73 | 9.26 / 9.20 | 7.67, 7.13 |
| 2 | 8.60 / 8.50 | 10.88 / 11.07 | 8.51 / 8.20 | 7.69, 6.81 |
| 3 | 8.59 / 8.54 | 11.12 / 11.38 | 8.61 / 8.41 | – |
| 4 | 8.67 / 8.64 | 10.42 / 10.51 | 8.27 / 8.04 | 7.12, 6.64 |
| 5 | 7.92 / 7.73 | 9.61 / 9.52 | 8.10 / 7.87 | 6.57, 6.32 |
| 6 | 7.79 / 7.56 | 9.90 / 9.85 | 8.27 / 8.03 | – |
| 7 (end of window phase) | 7.69 / 7.49 | 10.05 / 10.12 | 8.07 / 7.75 | 6.50, 6.14 |
| 8 (pool only) | 7.05 / 6.67 | 9.33 / 9.07 | 7.52 / 7.09 | 5.71, 6.09 |
| 9 (first joint epoch) | 6.71 / 6.22 | 8.73 / 8.28 | 7.45 / 6.98 | 5.08, 6.02 |
| 10 | 6.69 / 6.19 | **8.56 / 8.19** | 7.29 / 6.83 | 5.39, 6.17 |
| 11 | 6.68 / 6.22 | 8.88 / 8.45 | 7.31 / 6.84 | – |
| 12 | 6.65 / 6.12 | 8.80 / 8.45 | **7.18 / 6.72** | 5.31, 5.90 |
| 13 | 6.63 / 6.09 | 8.74 / 8.34 | 7.22 / 6.68 | – |
| 14 | 6.61 / 6.07 | 8.90 / 8.54 | 7.27 / 6.75 | 5.24, 5.96 |
| 15 | **6.59 / 6.06** | 8.87 / 8.49 | 7.28 / 6.80 | – |
| 16 (end) | 6.60 / 6.07 | 8.85 / 8.47 | 7.26 / 6.76 | 5.28, 5.86 |
| Training recordings, epoch 3 | 7.56 | 10.13 | 7.41 | |
| Training recordings, epoch 7 | 5.84 | 9.01 | 6.05 | |
| Training recordings, epoch 11 | 4.27 | 8.50 | 4.64 | |
| Training recordings, epoch 15 | 3.72 | 8.32 | 4.20 | |

**Bold**: each run's best validation epoch.

Reading Table 7.11:

- **The authors' settings remove CBraMod's under-fitting.** `CB5_L1` is 1.3 to 3.9 years better
  than `CB4_L1` on validation at every epoch, and at epoch 11 it fits its training recordings at
  4.64 years against 8.50 (P4: 4.27). Which of the changes does it (learning rate on every
  layer, the all-token readout, no amplitude clip, no mixup) is not separated by this run.
- **It is still behind LaBraM.** In the bag phase it stays 0.4 to 0.8 years behind P4 at the
  same epoch on validation and on test (7.22 / 6.68 against 6.63 / 6.09 at epoch 13). It was
  ahead of P4 at epochs 2 and 4, before P4's step at epoch 5.
- **On LEMON it is close to LaBraM.** It logs 5.08 to 5.39 years from epoch 9 on; P4 logs 4.83
  to 5.33 at the same epochs. `CB4_L1` had 7.39 at its best epoch (Table 7.3).

### Table 7.13. `CB5_L1` against P4 on fold 0

Each run at its best validation epoch (`CB5_L1` epoch 12, P4 epoch 15). MAE in years (Eq. (6.4))
per recording; Δ = `CB5_L1` − P4 on the same recordings (Eq. (6.5)) with its 95 % interval from
2,000 bootstrap resamples, positive when `CB5_L1` is worse. Test: the first 5 minutes of the
7,612 held-out recordings (P4 re-scored with `scripts/eval_age_by_cohort.py`, `CB5_L1` from its
test bag-weight file). External cohorts as in Table 7.5: adults, every window, no trims for
Dortmund, TDBRAIN and SRM, all runs scored with `scripts/eval_age_by_cohort.py`
(`scripts/external_runs/fm_external.sh`). Validation is each run's logged value.

| Cohort | $n$ | P4 MAE | `CB5_L1` MAE | Δ [95 % interval] | `CB4_L1` MAE |
|---|---|---|---|---|---|
| Validation (fold 0) | 19,071 | 6.59 | 7.18 | +0.59 | 8.56 |
| Test, all | 7,612 | 6.06 | 6.72 | +0.66 [+0.56, +0.76] | 8.19 |
| Test, Harvard | 7,338 | 6.05 | 6.71 | +0.66 [+0.57, +0.76] | 8.20 |
| Test, Temple | 274 | 6.32 | 6.97 | +0.64 [+0.13, +1.15] | 8.06 |
| LEMON | 214 | 4.73 | 5.14 | +0.40 [−0.04, +0.89] | 7.44 |
| Dortmund | 3,262 | 5.29 | 5.75 | +0.46 [+0.32, +0.62] | 6.96 |
| TDBRAIN | 2,222 | 5.48 | 5.90 | +0.42 [+0.24, +0.60] | 7.03 |
| SRM | 152 | 5.01 | 6.20 | +1.20 [+0.32, +2.11] | 6.39 |
| CHBP | 250 | 7.45 | 10.38 | +2.93 [+2.00, +3.95] | 10.65 |

- **Verdict.** By the decision rule CBraMod with its authors' settings does not replace LaBraM:
  it is 0.66 years worse on Harvard test, above the 0.38-year threshold and with an interval
  far from 0, and it loses on Dortmund (+0.46). On LEMON the difference is +0.40 with an
  interval that includes 0 (Table 7.13).
- **The authors' settings are worth 1.5 years for CBraMod** on test (6.72 against 8.19), 2.3 on
  LEMON, 1.2 on Dortmund and 1.1 on TDBRAIN: with our recipe the encoder was under-trained.
- **CHBP stays out of reach for CBraMod.** It has 10.4 years with either recipe, against 7.45
  for P4; the device shift of that cohort (`docs/chbp_domain_shift.md`) is not helped by its
  training settings.
- **Averaging it with P4 gives nothing on test.** The mean of the two models' predictions has a
  test MAE of 6.12 years against 6.06 for P4 alone (Δ +0.06, interval +0.01 to +0.12);
  Table 7.16 has the external cohorts.

### 7.6 Result: REVE with only its last 6 layers trained

### Table 7.10. `RV4T6_L1` against P4 on fold 0

Each run at its best validation epoch (`RV4T6_L1` epoch 10, P4 epoch 15). MAE in years
(Eq. (6.4)) per recording; Δ = `RV4T6_L1` − P4 on the same recordings (Eq. (6.5)) with its 95 %
interval from 2,000 bootstrap resamples, positive when `RV4T6_L1` is worse. Test: the first
5 minutes of the 7,612 held-out recordings (P4 re-scored with `scripts/eval_age_by_cohort.py`,
`RV4T6_L1` from its test bag-weight file). Validation, LEMON (first 5 minutes, so it differs
from Table 7.5's all-window LEMON row) and TDBRAIN (adults, all windows) are each run's values
logged during training; P4's TDBRAIN value is computed from its saved predictions with the same
protocol (Table 7.12).

| Cohort | $n$ | P4 MAE | `RV4T6_L1` MAE | Δ [95 % interval] |
|---|---|---|---|---|
| Validation (fold 0) | 19,071 | 6.59 | 7.17 | +0.58 |
| Test, all | 7,612 | 6.06 | 6.63 | +0.57 [+0.48, +0.67] |
| Test, Harvard | 7,338 | 6.05 | 6.61 | +0.56 [+0.46, +0.67] |
| Test, Temple | 274 | 6.32 | 7.15 | +0.83 [+0.23, +1.41] |
| LEMON, first 5 minutes | 214 | 4.87 | 6.62 | +1.75 |
| TDBRAIN adults, all windows | 2,222 | 5.48 | 6.58 | +1.10 |

- **Verdict.** By the decision rule REVE with only its last 6 layers trained does not replace
  LaBraM: it is 0.56 years worse on Harvard test, above the 0.38-year threshold and with an
  interval far from 0, and it is worse on Temple test, on LEMON and on TDBRAIN (Table 7.10). It is far
  better than CBraMod with the same recipe (+2.14 years, Table 7.5).
- **It memorises the training recordings.** At epoch 11 its MAE on the scored training
  recordings is 2.28 years against 7.30 on validation, a gap of 5.0 years; P4 has 4.27 against
  6.68 at the same epoch (Table 7.6). Its validation MAE does not improve after epoch 10
  (7.23 to 7.30, Table 7.7).
- **Averaging it with P4 gives nothing.** The mean of the two models' predictions has a test MAE
  of 6.07 years against 6.06 for P4 alone (Δ +0.02, interval −0.03 to +0.07), although the two
  predictions correlate at 0.96.
- **The run ended one epoch early.** The job was killed for host memory while scoring epoch 15
  (Table 7.2), so the last epoch of the cosine was not run. With validation flat since epoch 10,
  that epoch would not change the result. The full external scoring of Table 7.5 was not run
  for this model: the test split already decides, and the best checkpoint stays on S3
  (`…/labram/checkpoints/labram-mil-rv4t6-l1-f0-fold-0-20261125T211950/cv/fold_0/checkpoint-best.pth`).

### 7.7 All runs per cohort

### Table 7.12. Stage B runs per cohort at their best validation epoch

Recording-level MAE in years (Eq. (6.4)) as each run logs it at its best validation epoch so far:
the fold-0 validation recordings and the 7,612 held-out test recordings (first 5 minutes), split
by source dataset (by report status: Table 7.14), and the two standing external test sets as scored
during training: LEMON with the run's window selection (60 s trims, first 5 minutes, so it
differs from the all-window LEMON row of Table 7.5), and TDBRAIN adults with every window and no
trims (the protocol of Table 7.5). P4 did not have TDBRAIN as a test set during training; its
value is computed from its saved per-recording predictions with the same protocol
(`/data/runs/heedb/external_eval/tdbrain_all_p34.json`, the 2,222 recordings of adults). The two
ways of scoring agree: `CB4_L1` has 7.03 years both ways. All runs are finished.

| Cohort | $n$ | P4 (LaBraM) | `CB4_L1` | `CB5_L1` | `RV4T6_L1` | `RV4DP_L1` | `RV5_L1` |
|---|---|---|---|---|---|---|---|
| Best validation epoch | | 15 | 10 | 12 | 10 | 12 | 10 |
| Validation, all | 19,071 | 6.59 | 8.56 | 7.18 | 7.17 | 6.55 | **6.43** |
| Validation, Harvard | | 6.14 | 8.40 | 6.74 | 6.73 | 6.07 | **5.92** |
| Validation, Temple | | 7.55 | 8.90 | 8.12 | 8.12 | 7.59 | **7.53** |
| Test, all | 7,612 | 6.06 | 8.19 | 6.72 | 6.63 | 5.94 | **5.81** |
| Test, Harvard | 7,338 | 6.05 | 8.20 | 6.71 | 6.61 | 5.94 | **5.79** |
| Test, Temple | 274 | 6.32 | 8.06 | 6.97 | 7.15 | **6.17** | 6.47 |
| Test, all: $R^2$ | 7,612 | 0.83 | 0.72 | 0.79 | 0.80 | 0.83 | 0.84 |
| LEMON, first 5 minutes | 214 | **4.87** | 7.39 | 5.31 | 6.62 | 5.24 | 4.97 |
| TDBRAIN adults, all windows | 2,222 | 5.48 | 7.03 | 5.90 | 6.58 | 5.68 | **5.37** |

**Bold**: the lowest MAE of a row.

Reading Table 7.12:

- **`RV5_L1` has the lowest error on validation, on test and on TDBRAIN.** It is 0.16 years
  below P4 on validation, 0.24 on test and 0.11 on TDBRAIN, and 0.10 above it on LEMON
  (first 5 minutes). Table 7.18 has the paired intervals.
- **`RV4DP_L1` is level with P4 in the training domain** (0.04 to 0.15 years below it) and 0.2
  to 0.4 years above it on LEMON and TDBRAIN.
- **`CB5_L1` and `RV4T6_L1` are 0.6 to 0.8 years behind P4 on test**, on the Harvard part and on
  the Temple part. `CB5_L1` is the better of the two outside the training data: 5.31 against 6.62
  years on LEMON, 5.90 against 6.58 on TDBRAIN.
- **On the Temple test recordings no REVE run is clearly ahead:** `RV4DP_L1` has 6.17 and
  `RV5_L1` 6.47 against 6.32 for P4, on 274 recordings.
- **The external cohorts with the protocol of Table 7.5** (all windows, Dortmund, SRM, CHBP,
  paired intervals) are in Table 7.13 for `CB5_L1`, Table 7.15 for `RV4DP_L1` and Table 7.18
  for `RV5_L1`. `RV4T6_L1` was not scored on them (Section 7.6).

### Table 7.14. Stage B runs by report status

The cohorts of Table 7.12 split by the report verdict of a recording: normal, abnormal, or
unlabelled (no verdict in the source data). Same runs, epochs, scoring and state as Table 7.12;
MAE in years (Eq. (6.4)). All 214 LEMON recordings are healthy volunteers. In TDBRAIN "normal"
are the 354 recordings of healthy participants and "unlabelled" all others; P4's TDBRAIN values
are computed from its saved predictions as in Table 7.12.

| Cohort | Report | $n$ | P4 (LaBraM) | `CB4_L1` | `CB5_L1` | `RV4T6_L1` | `RV4DP_L1` | `RV5_L1` |
|---|---|---|---|---|---|---|---|---|
| Validation | normal | 4,927 | 6.01 | 8.01 | 6.54 | 6.77 | 6.01 | **5.86** |
| Validation | abnormal | 7,320 | 7.42 | 9.15 | 7.99 | 7.84 | 7.40 | **7.29** |
| Validation | unlabelled | 6,824 | 6.12 | 8.33 | 6.77 | 6.74 | 6.03 | **5.93** |
| Validation, Harvard | normal | 2,632 | 5.54 | 8.16 | 6.11 | 6.29 | 5.50 | **5.37** |
| Validation, Harvard | abnormal | 4,000 | 6.77 | 8.81 | 7.38 | 7.27 | 6.77 | **6.56** |
| Validation, Harvard | unlabelled | 6,331 | 5.99 | 8.24 | 6.59 | 6.56 | 5.85 | **5.75** |
| Validation, Temple | normal | 2,295 | 6.55 | 7.83 | 7.03 | 7.33 | 6.59 | **6.43** |
| Validation, Temple | abnormal | 3,320 | 8.21 | 9.56 | 8.73 | 8.52 | **8.16** | 8.18 |
| Validation, Temple | unlabelled | 493 | **7.85** | 9.46 | 9.08 | 9.06 | 8.39 | 8.31 |
| Test | normal | 1,667 | 5.62 | 8.17 | 6.35 | 6.43 | 5.59 | **5.44** |
| Test | abnormal | 2,378 | 6.71 | 8.56 | 7.43 | 7.32 | 6.68 | **6.49** |
| Test | unlabelled | 3,567 | 5.82 | 7.96 | 6.42 | 6.26 | 5.62 | **5.53** |
| Test, Harvard | normal | 1,518 | 5.71 | 8.33 | 6.43 | 6.53 | 5.70 | **5.52** |
| Test, Harvard | abnormal | 2,253 | 6.62 | 8.48 | 7.36 | 7.21 | 6.59 | **6.37** |
| Test, Harvard | unlabelled | 3,567 | 5.82 | 7.96 | 6.42 | 6.26 | 5.62 | **5.53** |
| Test, Temple | normal | 149 | 4.65 | 6.49 | 5.53 | 5.36 | **4.39** | 4.65 |
| Test, Temple | abnormal | 125 | 8.31 | 9.93 | 8.68 | 9.28 | **8.28** | 8.64 |
| LEMON, first 5 minutes | normal | 214 | **4.87** | 7.39 | 5.31 | 6.62 | 5.24 | 4.97 |
| TDBRAIN adults, all windows | normal | 354 | 5.43 | 7.31 | 6.33 | 7.30 | 6.09 | **5.41** |
| TDBRAIN adults, all windows | unlabelled | 1,868 | 5.49 | 6.98 | 5.81 | 6.45 | 5.61 | **5.36** |

**Bold**: the lowest MAE of a row.

Reading Table 7.14:

- **`RV5_L1` is the best model in every report group of the Harvard recordings**, on validation
  and on test: 0.17 to 0.29 years below P4 (test: normal 5.52 against 5.71, abnormal 6.37
  against 6.62, unlabelled 5.53 against 5.82).
- **On Temple it gains nothing.** On the Temple test recordings it equals P4 on normal reports
  (4.65) and is 0.33 years worse on abnormal ones (8.64 against 8.31; 149 and 125 recordings).
  `RV4DP_L1` is the best there (4.39 and 8.28).
- **Abnormal recordings are harder for every model.** On test the abnormal group is 0.4 to 1.1
  years above the normal one in every run (P4 1.09, `RV5_L1` 1.05). The difference is largest
  on Temple test, 3.7 years for P4 (4.65 against 8.31).
- **On TDBRAIN `RV5_L1` is level with P4 in both groups** (healthy participants 5.41 against
  5.43, all others 5.36 against 5.49). The other foreign-encoder runs are 0.3 to 0.9 years worse
  on the healthy group than on the others.

### 7.8 Result: REVE fully trained with stochastic depth, alone and in an ensemble

### Table 7.15. `RV4DP_L1` against P4 on fold 0

Each run at its best validation epoch (`RV4DP_L1` epoch 12, P4 epoch 15). MAE in years
(Eq. (6.4)) per recording; Δ = `RV4DP_L1` − P4 on the same recordings (Eq. (6.5)) with its 95 %
interval from 2,000 bootstrap resamples, positive when `RV4DP_L1` is worse. Test: the first
5 minutes of the 7,612 held-out recordings (P4 re-scored with `scripts/eval_age_by_cohort.py`,
`RV4DP_L1` from its test bag-weight file). External cohorts as in Table 7.5: adults, every
window, no trims for Dortmund, TDBRAIN and SRM (`scripts/external_runs/fm_external.sh`,
`scripts/fm_external_compare.py`). Validation is each run's logged value.

| Cohort | $n$ | P4 MAE | `RV4DP_L1` MAE | Δ [95 % interval] |
|---|---|---|---|---|
| Validation (fold 0) | 19,071 | 6.59 | 6.55 | −0.04 |
| Test, all | 7,612 | 6.06 | 5.94 | −0.11 [−0.20, −0.01] |
| Test, Harvard | 7,338 | 6.05 | 5.94 | −0.11 [−0.20, −0.01] |
| Test, Temple | 274 | 6.32 | 6.17 | −0.15 [−0.70, +0.40] |
| LEMON | 214 | 4.73 | 5.05 | +0.31 [−0.11, +0.75] |
| Dortmund | 3,262 | 5.29 | 5.84 | +0.55 [+0.39, +0.71] |
| TDBRAIN | 2,222 | 5.48 | 5.68 | +0.21 [+0.02, +0.39] |
| SRM | 152 | 5.01 | 6.98 | +1.97 [+0.89, +3.02] |
| CHBP | 250 | 7.45 | 7.21 | −0.24 [−1.14, +0.62] |

- **Verdict.** By the decision rule REVE with stochastic depth does not replace LaBraM. It wins
  on Harvard test (−0.11 years, interval excludes 0) and does not lose on LEMON (+0.31, interval
  includes 0, below the 0.38-year threshold), but it loses on Dortmund: +0.55 years with an
  interval far from 0 (Table 7.15).
- **In the training domain it is level or slightly better; outside it is behind.** It is 0.11
  years better on the Harvard and Temple test recordings and 0.2 to 2.0 years worse on Dortmund,
  TDBRAIN and SRM. The largest loss is on SRM, the smallest cohort (152 recordings).
- **On CHBP it is level with P4** (7.21 against 7.45, interval −1.14 to +0.62), where CBraMod has
  10.4 years (Table 7.13).
- **It is the best foreign-encoder run on every cohort but SRM and Dortmund**, where `CB5_L1` is
  0.8 and 0.1 years better (Tables 7.13 and 7.15).

### Table 7.16. Averages of models against P4

MAE in years of the mean of the predictions of several models, on the cohorts of Table 7.15.
"Two models": P4 fold 0 averaged with one other model; Δ against P4 fold 0 alone. The other
model is another P4 fold (the control: what a second LaBraM model gives; the mean over the other
folds), `RV4DP_L1` or `CB5_L1`. "P4 ensemble": the mean of the 5 P4 folds, each at its best
validation epoch (fold 3 on test at epoch 15 instead of 16: the predictions of its last epoch
were not saved, and its test MAE is 6.12 at both); "+ run": that ensemble with the run as one
more member of equal weight, Δ against the ensemble.
Intervals: 95 %, 2,000 bootstrap resamples over recordings (`scripts/fm_external_compare.py`).

| Cohort | P4 fold 0 | + another P4 fold | + `RV4DP_L1` (Δ) | + `CB5_L1` (Δ) | P4 ensemble | + `RV4DP_L1` (Δ) | + `CB5_L1` (Δ) |
|---|---|---|---|---|---|---|---|
| Test, all | 6.06 | 5.99 | 5.75 (−0.31 [−0.36, −0.26]) | 6.12 (+0.06 [+0.01, +0.12]) | 5.94 | 5.82 (−0.13 [−0.14, −0.11]) | 5.94 (−0.00 [−0.02, +0.02]) |
| LEMON | 4.73 | 4.70 | 4.70 (−0.03 [−0.25, +0.20]) | 4.83 (+0.09 [−0.13, +0.34]) | 4.67 | 4.63 (−0.04 [−0.12, +0.04]) | 4.69 (+0.02 [−0.06, +0.11]) |
| Dortmund | 5.29 | 5.21 | 5.17 (−0.12 [−0.21, −0.03]) | 5.18 (−0.11 [−0.19, −0.03]) | 5.16 | 5.05 (−0.11 [−0.14, −0.08]) | 5.08 (−0.08 [−0.11, −0.05]) |
| TDBRAIN | 5.48 | 5.40 | 5.19 (−0.29 [−0.39, −0.18]) | 5.32 (−0.16 [−0.26, −0.06]) | 5.36 | 5.21 (−0.15 [−0.19, −0.12]) | 5.28 (−0.08 [−0.11, −0.05]) |
| SRM | 5.01 | 4.92 | 5.54 (+0.53 [−0.05, +1.07]) | 5.04 (+0.03 [−0.46, +0.51]) | 4.84 | 4.81 (−0.04 [−0.23, +0.15]) | 4.67 (−0.17 [−0.35, +0.01]) |
| CHBP | 7.45 | 7.72 | 6.67 (−0.78 [−1.27, −0.31]) | 8.11 (+0.66 [+0.14, +1.20]) | 7.95 | 7.52 (−0.44 [−0.60, −0.27]) | 7.81 (−0.14 [−0.35, +0.06]) |

Reading Table 7.16:

- **REVE adds more to P4 than a second LaBraM model does.** Averaged with P4 fold 0, `RV4DP_L1`
  lowers the test MAE by 0.31 years; another P4 fold lowers it by 0.07 (5.99). The same holds on
  TDBRAIN (0.29 against 0.08) and CHBP (0.78 against a rise of 0.27). Its predictions correlate
  with P4's at 0.97, two P4 folds at 0.99.
- **As a sixth member it improves the P4 ensemble on four of six cohorts and harms none.** With
  equal weight per model the MAE falls by 0.13 years on test, 0.11 on Dortmund, 0.15 on TDBRAIN
  and 0.44 on CHBP, each with an interval that excludes 0; LEMON and SRM do not change (−0.04,
  intervals include 0).
- **The weight matters on SRM.** With half the weight on the single REVE model, SRM gets worse
  (5.54 against 5.01 for P4 fold 0); with one sixth it does not (4.81 against 4.84).
- **CBraMod adds less.** As a sixth member `CB5_L1` lowers the MAE by 0.08 years on Dortmund and
  TDBRAIN and changes nothing on test.
- **What this does not show.** One REVE model on one fold: whether five REVE folds would add more
  than one, and how much of the gain a sixth LaBraM model with another seed would give, is not
  tested. The control uses a P4 model of another fold, which also saw other training recordings.

### Table 7.17. The P4 cross-validation ensemble with one more model

The short form of Table 7.16. "P4 ensemble (LaBraM CrossValid)": the mean of the predictions of
the 5 P4 models, one per cross-validation fold. "+ run": the same ensemble with that one model
as a sixth member of equal weight. MAE in years (Eq. (6.4)) per recording; Δ against the P4
ensemble with its 95 % bootstrap interval (Eq. (6.5)), negative when the added model helps. The
first row is the number of weights of all models of an ensemble: a P4 model has 5.8 M, a REVE
model 69.3 M, the CBraMod model 12.5 M (4.9 M encoder and 7.6 M all-token readout).

| Cohort | $n$ | P4 ensemble (LaBraM CrossValid) | + `RV5_L1` (REVE) | Δ [95 % interval] | + `RV4DP_L1` (REVE) | Δ [95 % interval] | + `CB5_L1` (CBraMod) | Δ [95 % interval] |
|---|---|---|---|---|---|---|---|---|
| Parameters | | 29.2 M | 98.5 M | +69.3 M | 98.5 M | +69.3 M | 41.7 M | +12.5 M |
| Test, all | 7,612 | 5.94 | 5.83 | −0.12 [−0.13, −0.10] | 5.82 | −0.13 [−0.14, −0.11] | 5.94 | −0.00 [−0.02, +0.02] |
| Test, Harvard | 7,338 | 5.94 | 5.82 | −0.12 [−0.13, −0.11] | 5.81 | −0.13 [−0.15, −0.11] | 5.94 | −0.00 [−0.02, +0.02] |
| Test, Temple | 274 | 6.02 | 5.95 | −0.07 [−0.16, +0.02] | 5.92 | −0.10 [−0.20, −0.01] | 6.00 | −0.02 [−0.11, +0.08] |
| LEMON | 214 | 4.67 | 4.59 | −0.08 [−0.16, +0.01] | 4.63 | −0.04 [−0.12, +0.04] | 4.69 | +0.02 [−0.06, +0.11] |
| Dortmund | 3,262 | 5.16 | 5.01 | −0.15 [−0.18, −0.12] | 5.05 | −0.11 [−0.14, −0.08] | 5.08 | −0.08 [−0.11, −0.05] |
| TDBRAIN | 2,222 | 5.36 | 5.19 | −0.17 [−0.20, −0.14] | 5.21 | −0.15 [−0.19, −0.12] | 5.28 | −0.08 [−0.11, −0.05] |
| SRM | 152 | 4.84 | 4.60 | −0.25 [−0.43, −0.07] | 4.81 | −0.04 [−0.23, +0.15] | 4.67 | −0.17 [−0.35, +0.01] |
| CHBP | 250 | 7.95 | 7.55 | −0.40 [−0.57, −0.25] | 7.52 | −0.44 [−0.60, −0.27] | 7.81 | −0.14 [−0.35, +0.06] |

- **Either REVE model lowers the ensemble's error and raises it nowhere.** `RV5_L1` helps on
  every row (the LEMON and Temple intervals just include 0); `RV4DP_L1` helps on six of eight.
  On test the two give the same gain (0.12 and 0.13 years); outside the training data `RV5_L1`
  gives more on LEMON, Dortmund, TDBRAIN and SRM.
- **SRM improves with `RV5_L1` in the ensemble** (−0.25 years), although that model alone is
  1.25 years worse than P4 there (Table 7.18): its errors on SRM differ from P4's.
- **The CBraMod model helps on two rows only.** It lowers the error by 0.08 years on Dortmund
  and on TDBRAIN and leaves the test recordings unchanged.
- **The cost is weights and compute.** With a REVE model the ensemble grows from 29.2 M to
  98.5 M weights, and that model takes 3.3 times a P4 model's inference time (Table 7.1) and
  about 40 hours to train against 13 for one P4 fold (Table 7.2). The CBraMod model adds 12.5 M
  weights and trains in 13 hours.

### 7.9 Result: REVE with its authors' protocol

### Table 7.18. `RV5_L1` against P4 on fold 0

Each run at its best validation epoch (`RV5_L1` epoch 10, P4 epoch 15). MAE in years
(Eq. (6.4)) per recording; Δ = `RV5_L1` − P4 on the same recordings (Eq. (6.5)) with its 95 %
interval from 2,000 bootstrap resamples, positive when `RV5_L1` is worse. Test and external
cohorts as in Table 7.15 (`scripts/external_runs/fm_external.sh`; the external cohorts were
scored from a copy of the best checkpoint taken while the run was still training, identical to
the final one). Rows by report status are each run's logged values (Table 7.14).

| Cohort | $n$ | P4 MAE | `RV5_L1` MAE | Δ [95 % interval] |
|---|---|---|---|---|
| Validation (fold 0) | 19,071 | 6.59 | 6.43 | −0.16 |
| Test, all | 7,612 | 6.06 | 5.81 | −0.24 [−0.33, −0.16] |
| Test, Harvard | 7,338 | 6.05 | 5.79 | −0.26 [−0.34, −0.17] |
| Test, Temple | 274 | 6.32 | 6.47 | +0.15 [−0.39, +0.66] |
| Test, normal report | 1,667 | 5.62 | 5.44 | −0.18 |
| Test, abnormal report | 2,378 | 6.71 | 6.49 | −0.22 |
| Test, unlabelled | 3,567 | 5.82 | 5.53 | −0.29 |
| LEMON | 214 | 4.73 | 4.82 | +0.09 [−0.38, +0.57] |
| Dortmund | 3,262 | 5.29 | 5.33 | +0.04 [−0.10, +0.17] |
| TDBRAIN | 2,222 | 5.48 | 5.37 | −0.11 [−0.27, +0.05] |
| SRM | 152 | 5.01 | 6.26 | +1.25 [+0.37, +2.11] |
| CHBP | 250 | 7.45 | 7.93 | +0.48 [−0.44, +1.37] |

Reading Table 7.18:

- **Verdict: by the decision rule REVE with its authors' protocol replaces LaBraM on fold 0.**
  It wins on Harvard test (−0.26 years, interval excludes 0) and does not lose on LEMON (+0.09)
  or on Dortmund (+0.04), both with intervals that include 0. No other run passed all three.
  The rule's next step, cross-validation and the ensemble, is not done.
- **It is level with P4 outside the training data, except on SRM.** TDBRAIN −0.11 and CHBP +0.48
  have intervals that include 0. On SRM it is 1.25 years worse (interval +0.37 to +2.11, 152
  recordings), as the other REVE run was (+1.97, Table 7.15).
- **The gain is on the Harvard recordings, in every report group** (normal −0.19, abnormal
  −0.25, unlabelled −0.29 years on Harvard test). On Temple test it is level on normal reports
  (4.65 in both) and 0.33 years worse on abnormal ones (8.64 against 8.31, 125 recordings).
- **Against the other REVE recipe** (`RV4DP_L1`, Table 7.15) it is better on every cohort but
  CHBP: 0.13 years on test, 0.23 on LEMON, 0.51 on Dortmund, 0.31 on TDBRAIN and 0.72 on SRM;
  on CHBP it has 7.93 against 7.21. The two recipes differ in several settings at once
  (Table 7.9: learning rate per layer, readout, input scale, head-only epoch, mixup, position
  noise, gradient clipping, stochastic depth), so this run does not say which of them matters.
- **It memorises most and generalises best.** Its error on training recordings is 1.84 years at
  the end against 6.50 on validation (Table 7.6); validation does not move after epoch 10.

### Table 7.19. Ensembles with both REVE models, and the REVE models alone

MAE in years (Eq. (6.4)) of the mean of the predictions; Δ against the 5-fold P4 ensemble with
its 95 % bootstrap interval (Eq. (6.5)). "+ both REVE models": the 5 P4 models, `RV5_L1` and
`RV4DP_L1`, equal weight per model. "The two REVE models alone": the mean of `RV5_L1` and
`RV4DP_L1`. Both REVE models were trained on fold 0 with different recipes, so this is not a
cross-validation ensemble of REVE.

| Cohort | $n$ | P4 ensemble (5 models) | + both REVE models (7 models) | Δ [95 % interval] | The two REVE models alone | Δ [95 % interval] |
|---|---|---|---|---|---|---|
| Parameters | | 29.2 M | 167.7 M | +138.5 M | 138.5 M | +109.3 M |
| Test, all | 7,612 | 5.94 | 5.74 | −0.20 [−0.23, −0.18] | 5.68 | −0.27 [−0.34, −0.19] |
| Test, Harvard | 7,338 | 5.94 | 5.73 | −0.21 [−0.23, −0.18] | 5.66 | −0.28 [−0.36, −0.20] |
| Test, Temple | 274 | 6.02 | 5.89 | −0.13 [−0.27, +0.02] | 6.11 | +0.09 [−0.37, +0.55] |
| LEMON | 214 | 4.67 | 4.57 | −0.10 [−0.22, +0.03] | 4.85 | +0.18 [−0.21, +0.58] |
| Dortmund | 3,262 | 5.16 | 4.96 | −0.20 [−0.24, −0.16] | 5.29 | +0.12 [−0.00, +0.25] |
| TDBRAIN | 2,222 | 5.36 | 5.10 | −0.26 [−0.31, −0.21] | 5.23 | −0.12 [−0.27, +0.03] |
| SRM | 152 | 4.84 | 4.71 | −0.13 [−0.42, +0.13] | 6.26 | +1.42 [+0.52, +2.33] |
| CHBP | 250 | 7.95 | 7.28 | −0.67 [−0.91, −0.43] | 7.14 | −0.81 [−1.60, −0.05] |

Reading Table 7.19:

- **With both REVE models added, the P4 ensemble improves on every cohort:** by 0.20 years on
  test, 0.20 on Dortmund, 0.26 on TDBRAIN and 0.67 on CHBP; the LEMON, SRM and Temple intervals
  include 0. It is the lowest error of all combinations on LEMON, Dortmund and TDBRAIN.
- **Two REVE models already beat five LaBraM models in the training domain** (5.68 against
  5.94 years on test) and on CHBP, are level on TDBRAIN, LEMON and Dortmund (intervals include
  0), and are 1.4 years worse on SRM. REVE alone is not a safe replacement outside the training
  domain yet; REVE next to LaBraM is.

### Table 7.20. `RV5_L1` against P4 by cohort and report status

Tables 7.17 and 7.18 split by the report verdict of a recording (normal, abnormal, or unlabelled
= no verdict in the source data; `scripts/fm_external_compare.py --by-report`). Single models:
P4 fold 0 and `RV5_L1` with Δ = `RV5_L1` − P4. Ensembles: the 5-fold P4 ensemble and that
ensemble with `RV5_L1` as a sixth member, with Δ against the P4 ensemble. MAE in years
(Eq. (6.4)); intervals 95 %, 2,000 bootstrap resamples (Eq. (6.5)); negative Δ = lower error
with REVE. Validation rows are each run's logged values, and have no ensemble: the other P4
folds trained on these recordings. LEMON, Dortmund, SRM and CHBP hold healthy participants only;
in TDBRAIN "normal" are the healthy participants.

| Cohort | Report | $n$ | P4 fold 0 | `RV5_L1` | Δ [95 % interval] | P4 ensemble | + `RV5_L1` | Δ [95 % interval] |
|---|---|---|---|---|---|---|---|---|
| Validation | normal | 4,927 | 6.01 | 5.86 | −0.15 | – | – | – |
| Validation | abnormal | 7,320 | 7.42 | 7.29 | −0.13 | – | – | – |
| Validation | unlabelled | 6,824 | 6.12 | 5.93 | −0.19 | – | – | – |
| Test | normal | 1,667 | 5.62 | 5.44 | −0.18 [−0.33, −0.02] | 5.59 | 5.50 | −0.10 [−0.12, −0.07] |
| Test | abnormal | 2,378 | 6.71 | 6.49 | −0.22 [−0.38, −0.05] | 6.56 | 6.45 | −0.11 [−0.14, −0.08] |
| Test | unlabelled | 3,567 | 5.82 | 5.53 | −0.29 [−0.41, −0.17] | 5.70 | 5.56 | −0.13 [−0.15, −0.11] |
| Test, Harvard | normal | 1,518 | 5.71 | 5.52 | −0.19 [−0.35, −0.02] | 5.70 | 5.60 | −0.09 [−0.12, −0.07] |
| Test, Harvard | abnormal | 2,253 | 6.62 | 6.37 | −0.25 [−0.42, −0.08] | 6.49 | 6.38 | −0.12 [−0.15, −0.09] |
| Test, Harvard | unlabelled | 3,567 | 5.82 | 5.53 | −0.29 [−0.41, −0.17] | 5.70 | 5.56 | −0.13 [−0.15, −0.11] |
| Test, Temple | normal | 149 | 4.65 | 4.65 | −0.00 [−0.55, +0.56] | 4.53 | 4.42 | −0.11 [−0.21, −0.01] |
| Test, Temple | abnormal | 125 | 8.31 | 8.64 | +0.32 [−0.60, +1.24] | 7.80 | 7.78 | −0.02 [−0.18, +0.15] |
| LEMON | normal | 214 | 4.73 | 4.82 | +0.09 [−0.38, +0.57] | 4.67 | 4.59 | −0.08 [−0.16, +0.01] |
| Dortmund | normal | 3,262 | 5.29 | 5.33 | +0.04 [−0.10, +0.17] | 5.16 | 5.01 | −0.15 [−0.18, −0.12] |
| TDBRAIN | normal | 354 | 5.43 | 5.41 | −0.02 [−0.49, +0.46] | 5.30 | 5.16 | −0.14 [−0.23, −0.06] |
| TDBRAIN | unlabelled | 1,868 | 5.49 | 5.36 | −0.13 [−0.31, +0.05] | 5.37 | 5.19 | −0.17 [−0.21, −0.14] |
| SRM | normal | 152 | 5.01 | 6.26 | +1.25 [+0.37, +2.11] | 4.84 | 4.60 | −0.25 [−0.43, −0.07] |
| CHBP | normal | 250 | 7.45 | 7.93 | +0.48 [−0.44, +1.37] | 7.95 | 7.55 | −0.40 [−0.57, −0.25] |

Reading Table 7.20:

- **On the test recordings REVE is better in every report group**, with intervals that exclude
  0: normal −0.18, abnormal −0.22, unlabelled −0.29 years. The same holds for the Harvard part
  alone.
- **On Temple test it is level.** Normal reports are equal (4.65), abnormal ones +0.32 years;
  both intervals are wide (149 and 125 recordings) and include 0.
- **Outside the training data only healthy participants and TDBRAIN's unlabelled group exist.**
  REVE is level with P4 on the healthy groups of LEMON, Dortmund and TDBRAIN and on TDBRAIN's
  other recordings; SRM (+1.25) is the exception.
- **In the ensemble REVE helps in every group.** As a sixth member it lowers the error in all
  14 rows; the interval excludes 0 in 12 of them (not for Temple abnormal and LEMON). Abnormal
  recordings gain as much as normal ones (0.11 against 0.10 years on test).

**Next steps (none is started; each needs a decision).**

1. **`RV5_L1` on folds 1 to 4.** It decides whether the result of Table 7.18 holds across folds,
   and gives the 5-fold REVE ensemble to compare with the 5-fold P4 ensemble and with both
   together. About 39 hours per fold on one A10G.
2. **A sixth LaBraM model as a control for Tables 7.16 and 7.17** (P4 on fold 0 with another
   seed, about 13 hours): how much of the ensemble gain any additional model would give.
3. **Why REVE is worse on SRM**, from the saved predictions (a constant offset or scattered
   error); no training needed.
4. **Later, if step 1 holds:** which settings of the authors' protocol carry the gain
   (Table 7.9), the protocol with stochastic depth added, and LoRA (`RV5L_L1`) as a cheaper
   way to adapt REVE.

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
ONLY=RV4T6_L1 FOLD=0 SPOT=false scripts/mil_runs/submit_grid.sh          # REVE, last 6 layers
JOB=RV4DP_L1 ENCODER=reve scripts/mil_runs/run_local_fm.sh model.encoder.drop_path=0.1   # REVE + stochastic depth, local
# the authors' settings (Section 7.5)
ONLY=CB5_L1 FOLD=0 SPOT=false WORKERS=24 scripts/mil_runs/submit_grid.sh             # ml.g5.4xlarge, 64 GB
ONLY=RV5_L1 FOLD=0 SPOT=false INSTANCE=ml.g5.8xlarge WORKERS=16 scripts/mil_runs/submit_grid.sh
# a finished run on the external cohorts, paired against P4 (Tables 7.13, 7.15, 7.16); next to a
# training run on the same GPU add GPU_FRACTION=0.17 BAG_CHUNK=128
scripts/external_runs/fm_external.sh score rv4dp RV4DP_L1_f0=<run dir>
scripts/external_runs/fm_external.sh table rv4dp RV4DP_L1_f0 <run dir>/analysis/test_bag_weights_epoch12.parquet
# move a SageMaker job to the local GPU at an epoch boundary: fetch its checkpoint folder, start the
# local run as its own unit, check that it trains, then stop the SageMaker job
AWS_PROFILE=mfa scripts/mil_runs/resume_local_from_sagemaker.sh fetch RV5_L1 \
    s3://sagemaker-us-east-1-574441342949/labram/checkpoints/<job folder>
systemd-run --user --unit=labram-rv5-l1-f0 --collect --working-directory=$PWD \
    --setenv=WEIGHTS=./checkpoints/reve-base.safetensors /bin/bash -c \
    'scripts/mil_runs/resume_local_from_sagemaker.sh run RV5_L1 >> /data/runs/heedb/fm_encoders/RV5_L1_fold0.log 2>&1'
# a job killed for host memory continues from its checkpoint folder (not on ml.g5.2xlarge: 32 GB is too little)
RESUME=s3://sagemaker-us-east-1-574441342949/labram/checkpoints/<job folder> ONLY=RV5_L1 FOLD=0 SPOT=false \
    INSTANCE=ml.g5.8xlarge WORKERS=16 scripts/mil_runs/submit_grid.sh

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
