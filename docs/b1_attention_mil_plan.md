# B1 attention MIL: implementation and experiment plan

Plan, 2026-11-08. This turns Proposal B1 of
[`window_aggregation_plan.md`](window_aggregation_plan.md) (section 4)
into concrete stages, files and runs. It takes into account what Proposal A
measured ([`window_confidence_results.md`](window_confidence_results.md)) and
the E2L recipe as it is today. In B1, a recording is the training unit: a bag
of $K$ windows goes through the LaBraM encoder, a learned gated attention
pools the window embeddings into one recording embedding, the head predicts
once, and the loss is taken once per recording.

## 1. What Proposal A tells B1

### Table 1.1. Findings that shape the design

| Finding (E2L, test) | Consequence for B1 |
|---|---|
| Mean pooling 7.00 y; A3 learned window weights 6.46 y, with spectrogram descriptors 6.38 y; all of the gain is on HEEDB (Tables 2.1, 5.1 of the results) | **The bar for B1**: it must beat 6.38 y (the best post-hoc pooling of the same model's windows), not just 7.00 |
| About 2/3 of the window-error variance is shared by the whole recording (Eq. (3.2)) | Re-weighting windows alone has a ceiling; B1's other lever, a recording-level *loss* that stops window-level memorization, is what can move the shared part |
| HEEDB windows in the first ~1.5 min after the 60 s trim have 13 y error vs 7.9 y later (Table 3.3); A3's largest weight is on the minute | Feed the window's **minute index** to the attention prior. Keep the E2L data unchanged so the comparison is clean; the trim fix is a separate arm (section 6) |
| Window descriptors that carry signal: across-channel power SD, weakest-channel SD, EMG ratio, TTA variance, wavelet burst peaks, aperiodic slope | The prior $q_i$ of Eq. (2.1): quality + STFT + wavelet descriptors, computed on the GPU per batch (`labram/eval/window_descriptors.py`), no cache needed |
| A3's own weights are a poor recording confidence ($\rho\approx-0.1$); a recording-level model on recording summaries reaches −0.31 (Table 5.3) | Do not expect attention entropy to be the confidence. Log it, but evaluate B1's confidence with the same learned-confidence protocol |
| E2L costs about 2 h 20 min per epoch (2.5 M windows, ~300 windows/s on one A10G), 37 GPU-hours in total | A from-scratch MIL ladder is expensive. Start with a frozen-encoder stage on cached embeddings (hours), then warm-start from E2L, and only then train from `labram-base` |

## 2. The model

With $x_i$ a 10 s window ($19\times2000$ samples, A1/A2/T1/T2 dropped as in
E2L), $f_\theta$ the LaBraM encoder up to its pooled feature (the CLS token
after the final norm in E2L, $D=200$), $q_i\in\mathbb R^{n_q}$ the window's
descriptors standardized with training statistics, and $B_c$ the bag of
recording $c$:

$$
\begin{aligned}
h_i&=f_\theta(x_i),\qquad
a_i=\mathbf w^{\top}\bigl(\tanh(\mathbf V h_i)\odot\sigma(\mathbf U h_i)\bigr)+\boldsymbol\beta^{\top}q_i,\\
\alpha_i&=\frac{\exp(a_i/\tau)}{\sum_{j\in B_c}\exp(a_j/\tau)},\qquad
z_c=\sum_{i\in B_c}\alpha_i h_i,\qquad \hat y_c=g(z_c),
\end{aligned}
\tag{2.1}
$$

$$
\mathcal L=\frac1{\lvert\mathcal B\rvert}\sum_{c\in\mathcal B}\operatorname{Huber}_{1}\bigl(\hat y_c-y_c\bigr)
+\lambda_H\frac1{\lvert\mathcal B\rvert}\sum_{c\in\mathcal B}\Bigl(\log\lvert B_c\rvert-H(\alpha_c)\Bigr),
\tag{2.2}
$$

where the symbols are:
- $\mathbf V,\mathbf U\in\mathbb R^{128\times D}$ and $\mathbf w\in\mathbb R^{128}$ are the gated-attention parameters (Ilse et al. 2018);
- $\boldsymbol\beta\in\mathbb R^{n_q}$ is the descriptor prior, set to 0 when `bag_pool_prior` is off;
- $\tau$ is a fixed temperature (default 1);
- $g$ is the existing head (`head_type=linear` in E2L);
- $y_c$ is the z-scored age;
- $H(\alpha_c)=-\sum_i\alpha_i\log\alpha_i$;
- $\lambda_H\in\{0, 0.01, 0.05\}$.

The pooled $z_c$ goes through the **same head** as today, so a
window-level checkpoint (E2L) loads into a bag model without any new head
weights. With $\alpha_i=1/\lvert B_c\rvert$ (`bag_pool=mean`), Eq. (2.1) is
the mean of the window *embeddings*, the matched control for the attention.

**Prior and descriptors.** $q_i$ is computed on the GPU from the raw window
before the forward pass:
- quality (5): EMG ratio and its channel max, max $\lvert x\rvert$, mean and minimum channel SD;
- STFT spectrogram (18) and Morlet wavelet (16);
- the minute index of the window inside the bag's recording.

That gives $n_q = 40$. The standardization mean and SD come from 200 training
batches at startup, are frozen, and are saved in the checkpoint, so eval uses
the same values.

## 3. Implementation

Design choice: **a bag is a dataset item.** A `RecordingBagDataset` wraps
the window-selected dataset, and its item $c$ is
`(x [K, C, L], y, stem, minutes [K], mask [K])`:
- **train:** $K$ windows drawn without replacement, re-drawn every epoch, padded and masked when a recording has fewer than $K$;
- **eval:** all of the recording's eval-budget windows (up to 30).

This keeps every downstream assumption intact: one target and one stem id per
item, `DistributedSampler` / `gather_sharded_eval` sharding by item (which is
now by recording), and the per-cohort and per-dataset metrics keyed by stem.
No custom batch sampler is needed.

### Table 3.1. Code changes

| # | Piece | Change | Files |
|---|---|---|---|
| 1 | Config | `data.bag_size` ($K$, 0 = off), `data.bag_size_choices` (train $K$ drawn per batch from, e.g., `[4, 8, 16]`), `model.bag_pool` = `none` / `mean` / `attention`, `model.bag_pool_prior` (bool), `model.bag_attention_dim` (128), `model.bag_temperature`, `loss.bag_entropy_weight` ($\lambda_H$); `validate()` rules: bag mode requires regression or multitask, and rejects `codebook_reg`, DeepSpeed and `head_type=attention` | `labram/configs/data_config.py`, `model_config.py` (also `TransformerArchConfig`), `loss_config.py`; round-trip in `tests/test_new_config_fields.py` |
| 2 | Bag dataset | `RecordingBagDataset(dataset, k, train, seed)`: groups window indices by stem (reusing `samplers._files` / `recording_stem`), keeps each window's minute index (`window_selection.window_index`), and `set_epoch` reseeds the draws; `collate_bags` pads to the batch's largest $K$ and builds the mask | new `labram/data/bags.py` |
| 3 | Loaders | in `run_finetune.main`: when `bag_size > 0`, wrap train/val/test/train_eval in `RecordingBagDataset` and skip `RecordingSubsetSampler` (a bag *is* the per-recording draw). `trainer.batch_size` counts **bags**. `num_training_steps_per_epoch` counts bags; `train_loop` calls `set_epoch` on the bag dataset | `labram/runs/run_finetune.py`, `labram/runs/finetune_setup.py` (`build_dataloaders` collate_fn) |
| 4 | Pooling module | `GatedAttentionPool(D, L, n_q, temperature)` and `MeanPool`, both `forward(h [B,K,D], mask, q=None) -> (z [B,D], alpha [B,K])` with masked softmax | new `labram/layers/bag_pool.py`, exported from `labram.layers` |
| 5 | Model | `NeuralTransformer.bag_pool` (built when `bag_pool != none`, so layer decay gives it the head's full LR; names keep `blocks.*` intact); `forward(..., features_only=True)` returns the pooled feature without the head; `forward_bag(x [B,K,C,A,T], mask, q, channel_indices, dropper=None) -> (out, alpha)` flattens to $B\cdot K$, encodes (through the channel dropper when one is configured), reshapes, pools, applies the head. Optional activation checkpointing for $K>16$ | `labram/models/neural_transformer.py`, `registry.py`; mirror the kwargs in `labram/eval/loading.py::build_finetune_model` |
| 6 | Channel drop | `ChannelDropper.forward(..., features_only=True)` passes it to the model and returns features, so the per-pattern grouping works on the flattened windows (E2L runs every forward through the dropper) | `labram/train/channel_drop.py` |
| 7 | Train step | bag branch in `train_one_epoch`: flatten for `augment_batch` (3-D) and the descriptors, `forward_bag`, criterion on the bag target `(B, 1)` + entropy term (Eq. (2.2)). Mixup acts on **bags**: one $\lambda$ per batch, a permutation of bags, mix the pooled $z_c$ and the targets (C-Mixup option kept). Logs the mean attention entropy and max weight | `labram/train/train_finetune.py` (`train_one_epoch`, `train_class_batch`) |
| 8 | Evaluation | bag branch in `evaluate`: one prediction row per recording, with the group id equal to the stem, so `aggregate_windows` sees one window per case and the case metrics, per-cohort and per-dataset breakdowns, `log_summary_metrics` and model selection all work unchanged. Window-level metrics are also kept by scoring the windows through the head (`window_*`). Writes `analysis/bag_weights.parquet` at the best epoch (stem, window, $\alpha_i$, window prediction) | `labram/train/train_finetune.py::evaluate`, `train_loop` |
| 9 | Freeze / LP-FT | `optimizer.head_warmup_epochs`: train only `bag_pool` + `head` (+ `fc_norm`) for the first epochs, then unfreeze (LP-FT); add `bag_pool` to the default `lora.train_prefixes` | `labram/runs/run_finetune.py`, `labram/train/train_finetune.py::train_loop`, `labram/layers/lora.py` |
| 10 | Offline tools | `collect_predictions` and `scripts/eval_age_by_cohort.py` score bag models; `window_confidence` diagnostics read `bag_weights.parquet` (α vs window error, vs descriptors, entropy) | `labram/eval/inference.py`, `scripts/eval_age_by_cohort.py`, `labram/eval/window_confidence.py` |

### Table 3.2. Tests

| Test | Checks |
|---|---|
| `tests/test_bag_pooling.py` | shapes; masked positions get $\alpha=0$; permutation invariance; `mean` equals the masked mean; the prior moves α as $\boldsymbol\beta$ says; the entropy term is 0 at the uniform α |
| `tests/test_bags.py` | one item per recording; $K$ draws without replacement and re-drawn per epoch; short recordings padded and masked; eval items are all eval windows in order; minute index |
| `tests/test_finetune.py` (extend) | `train_one_epoch` and `evaluate` in bag mode on a tiny model; weights change; bag mixup permutes bags; with $K=1$ and `bag_pool=mean`, bag mode equals window mode |
| `tests/test_channel_drop.py` (extend) | `features_only` through the dropper equals the per-sample forward |
| `tests/test_multi_gpu.py` (extend) | 2-process CPU run in bag mode; sharded eval equals single-process |
| `tests/test_samplers.py` (extend) | `run_finetune.main` end to end with `bag_size=4` on the npy fixture |

## 4. Stages and runs

All runs use E2L's data, split and eval protocol (`tuh_heedb_v1`, 19 channels,
60 s trims, first 5 min for val/test, CAR). Every stage reports val and test
recording MAE overall, per dataset (TUH / HEEDB) and per dataset × cohort,
next to E2L mean pooling (7.25 / 7.00) and A3 + spectrogram (6.75 / 6.38).

### Stage 1. Frozen-encoder MIL on cached E2L embeddings (about 2 GPU-hours)

Cache E2L's $h_i$ (CLS feature, 200-d, float16) and window prediction for the
val + test windows (543 k; ~10 min) and the training windows (2.5 M; ~1 h,
~1 GB). Then fit the pooling + a fresh linear head on recordings, with the
encoder frozen. Each variant trains in minutes on CPU or GPU.

### Table 4.1. Stage 1 variants

| Run | Pooling | Fitted on | Question |
|---|---|---|---|
| S1-mean | `mean` | val, cross-fitted by subject (as A3) | does pooling *embeddings* beat pooling predictions (7.25 / 7.00)? |
| S1-attn | `attention`, $\lambda_H=0.01$ | val, cross-fitted | does learned attention beat A3 + spectrogram (6.75 / 6.38)? |
| S1-attn-q | + descriptor prior | val, cross-fitted | does the prior add to content attention? |
| S1-attn-train | `attention` | training windows | is the memorized-train-embedding problem real? Expect worse transfer than fitting on val |

**Go / no-go.** If S1-attn does not beat A3 + spectrogram on val, attention
on E2L embeddings adds nothing over descriptor weights, and Stage 3 has to
earn its keep through the loss alone (B1-mean vs control). Either way,
Stage 1 gives the first α-vs-error diagnostics at almost no cost.

### Stage 2. Implementation (Table 3.1, Table 3.2)

About 2–3 days of work, in the order 1 → 2 → 4 → 5 → 6 → 3 → 7 → 8 → 9 → 10.
The tests are written with each piece, and a 2-epoch smoke run on a 2 % subset
precedes Stage 3.

### Stage 3. Warm start from E2L (about 10 GPU-hours)

Each run starts from E2L `checkpoint-best.pth` and trains 4 epochs: lr 1e-4,
1 warm-up epoch, the pool and head only for the first half epoch (LP-FT),
layer decay 0.65. One epoch is every training recording once with $K=8$
(84,891 bags, 0.68 M windows, ~40 min); batch 32 bags = 256 windows.

### Table 4.2. Stage 3 runs

| Run | Config | Question |
|---|---|---|
| W0 control | window-level loss, E2L recipe, `train_minutes_per_epoch=1.333` (8 windows per recording per epoch), same 4 epochs | what does more training alone do? |
| W1 B1-mean | `bag_size=8`, `bag_pool=mean` | does a recording-level loss help over W0? |
| W2 B1-attn | `bag_pool=attention`, $\lambda_H=0.01$ | do learned weights beat the mean inside training? |
| W3 B1-attn-q | + `bag_pool_prior` | do the descriptors help the trained attention? |

### Stage 4. From `labram-base` (about 20–25 GPU-hours)

Runs the best Stage 3 variant and the control from the pretrained
checkpoint, as E2L did: 12 epochs, $K=8$, $K$ drawn from $\{4, 8, 16\}$ in
training. This is the real test of the memorization question (the train-eval
MAE at the selected epoch vs val). Two seeds if the difference is below 0.3 y.

### Table 4.3. Stage 4 runs

| Run | Config | Question |
|---|---|---|
| F0 control | window-level, `train_minutes_per_epoch=1.333`, 12 epochs | matched window budget |
| F1 best B1 | bag mode, 12 epochs | lower val MAE **and** a smaller train–val gap than F0? |

### Stage 5. Optional

* **Bag-size sweep:** train $K\in\{4,16\}$; evaluate with 5 min and with the whole recording (all windows, encoded in chunks).
* **Multitask head:** status + pathology get recording-level pooling for free through `MultiTaskHead`.
* **B2 (precision pooling) and B3 (recording transformer):** only if B1 pays.

## 5. How runs are judged

* **Selection** is on val recording MAE; test is reported, never selected on.
  A change counts when it is at least 0.3 y on val, consistent on test, and
  the train–val gap is no larger (the rules of the `analyze-experiments` skill;
  noise floor about 0.15 y). Report paired bootstrap CIs on test, as in
  Table 2.2 of the results.
* **Diagnostics every B1 run reports** (from `bag_weights.parquet`, with the
  `window_confidence` functions):
  1. $\rho_{\text{within}}$ of $\alpha_i$ vs the window's own error, overall and per dataset × cohort, compared with A3's −0.16 to −0.18;
  2. Spearman of $\alpha_i$ with EMG, the minute index and the across-channel SD (does attention find what A3 found?);
  3. per-recording attention entropy and max weight (collapse check);
  4. the learned recording confidence of Eq. (5.2) of the results refitted on the B1 outputs (does $\rho$ improve beyond −0.31?), and risk-coverage curves;
  5. the normal / abnormal and TUH / HEEDB splits.

## 6. Decisions and risks

### Table 6.1. Decisions to confirm before Stage 1

| Decision | Recommendation | Alternative |
|---|---|---|
| Baseline | E2L (`tuh_heedb_v1`), the latest successful run; all A results are on it | TUAB-age M1 (the plan's original split): cheaper runs, but no HEEDB |
| HEEDB first-minutes effect | keep E2L's data; let the minute descriptor absorb it; run the trim fix as its own control later (needs a per-corpus `trim_start_sec`, not supported today) | fix the trim first. Cleaner, but it delays B1 and needs new code too |
| Stage 1 fitting set | val, cross-fitted by subject (train embeddings are memorized: E2L train MAE ~4 y vs 7 y val) | train embeddings: more data, likely optimistic attention |
| GPU budget | Stage 1 + 3 ≈ 12 GPU-hours, Stage 4 ≈ 25 | skip Stage 3 and go straight to Stage 4 |

### Table 6.2. Risks

| Risk | Mitigation |
|---|---|
| Attention collapses on one window (window memorization through the back door) | entropy term (Eq. (2.2)); log entropy and max α per epoch; `mean` as the control |
| Train $K$ (8) differs from eval $K$ (30) | draw train $K$ from $\{4,8,16\}$; evaluate at several budgets (Stage 5) |
| Channel-drop path runs the head per window | `features_only` through the dropper (Table 3.1 #6) |
| Mixup semantics | mix pooled $z_c$ of bags, never windows across recordings |
| Fewer optimizer steps per epoch | count epochs in bags; keep steps comparable to the control by the matched window budget |
| New fields forgotten in the second model builder | mirror them in `labram/eval/loading.py::build_finetune_model`; test rebuild from a saved bag checkpoint |
| Memory with large eval bags | encode in chunks under `no_grad`; attention depends only on $h_i$, so chunking is exact |
| Descriptor standardization drifts between train and eval | stats frozen at startup and stored in the checkpoint |
