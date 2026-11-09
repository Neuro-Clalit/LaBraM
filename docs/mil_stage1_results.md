# MIL Stage 1: attention pooling on frozen E2L / E2 window features

Results note, 2026-11-09. This is Stage 1 of
[`b1_attention_mil_plan.md`](b1_attention_mil_plan.md). The bag pooling
layers M0–M4 are trained on **cached, frozen** window features of E2L (the
run under test) and E2 (the benchmark). The comparison is against mean
pooling, and it includes analytics of what the attention weights follow.
Stages 3–4 (end-to-end bag training on SageMaker) are listed in section 5.

Code:
- `labram/mil/cache.py` + `scripts/mil_cache.py`: the feature cache.
- `labram/layers/bag_pool.py`: M0–M4.
- `labram/mil/stage1.py` + `scripts/mil_stage1.py`: training, metrics and analytics.
- `scripts/mil_stage1_report.py`: these tables and figures.

Tables are in `docs/data/mil_stage1/`. The cache (3.0 M windows per run) is at
`/data/runs/heedb/mil_cache/` and mirrored to the private
`s3://eeg-data-public/HEEDB/v4.1/derived/mil_cache/`.

## 1. Setup

**Cache.** Every window of a recording's eval budget (the first 5 min after
the 60 s trim, up to 30 windows) is stored for val (10,614 recordings), test
(7,612) and train (84,891, the same 30-window budget). Per window, the cache
holds:
- $h_i$: E2L's head-input feature, the CLS token after the final norm (200-d), computed through the run's channel-drop path exactly as in training;
- the run's window prediction $\hat y_i$;
- 39 descriptors: 5 quality, 18 STFT spectrogram and 16 wavelet;
- the minute of the window within the 5-minute budget.

**Model.** A pool of [`b1_attention_mil_plan.md`](b1_attention_mil_plan.md)
Eq. (2.1) maps the bag to $z_c$; the linear head, **initialised from the
run's own head**, predicts the z-scored age. The loss is Huber plus
$0.01\,(\log n - H(\alpha))$ for the softmax pools (Eq. (2.2) of the plan).
Training uses AdamW with a cosine schedule: pool lr 1e-3, head lr 1e-4,
dropout 0.1 on $h_i$.

**Protocols.**
- `val_cf`: fitted on val, 5-fold cross-fitted with folds grouped by subject (out-of-fold val predictions); refitted on all of val for test, 30 epochs.
- `train`: fitted on the training bags, 15 epochs, epoch chosen on val (so val is optimistic and test is the honest number).

**Benchmarks.**
- `M0-run`: the run's own reported mean pooling, i.e. the mean of $\hat y_i$, nothing fitted.
- `M0`: mean pooling of $h_i$ with the head refitted, the like-for-like control of every attention variant.

### Table 1.1. Variants

| name | pool | attention reads | notes |
|---|---|---|---|
| M1-h / M1-q / M1-hq | gated attention, $L = 128$ (M1) | $h_i$ / $q_i$ / both | softmax over the bag |
| M1-hq-time | M1-hq | both | + sinusoidal minute encoding added to $h_i$ |
| M1-hq-noent | M1-hq | both | no entropy term |
| M2-h / M2-hq | 4 gated branches, $L = 64$ (M2) | $h_i$ / both | top-window masking $p = 0.3$ |
| M3-h / M3-q / M3-hq | sigmoid gates, $L = 64$ (M3) | $h_i$ / $q_i$ / both | non-competing gates |
| M4-hq / M4-hq-sa | learned query, 4 heads (M4) | both | M4-hq-sa: + 1 self-attention block |

## 2. Recording MAE

### Table 2.1. `val_cf`: fitted on val (out-of-fold), test MAE in years

The Δ columns are the paired change of test MAE against `M0`, with 95 %
bootstrap CIs over recordings (1,000 resamples).

| variant | E2L val | E2L test | E2L test TUH | E2L test HEEDB | E2 val | E2 test | Δ test vs M0, E2L | Δ test vs M0, E2 |
|---|---|---|---|---|---|---|---|---|
| M0-run | 7.25 | 7.00 | 6.55 | 7.02 | 7.43 | 7.25 | +0.47 [+0.41, +0.52] | +0.69 [+0.62, +0.75] |
| **M0** (benchmark) | 6.90 | 6.54 | 6.36 | 6.54 | 6.88 | 6.56 | — | — |
| M1-h | 6.68 | 6.29 | 6.22 | 6.30 | 6.61 | 6.26 | −0.24 [−0.31, −0.18] | −0.30 [−0.36, −0.24] |
| **M1-q** | **6.64** | **6.22** | 6.43 | **6.22** | 6.61 | **6.23** | **−0.31** [−0.36, −0.26] | **−0.33** [−0.38, −0.28] |
| M1-hq | 6.72 | 6.32 | 6.42 | 6.32 | 6.64 | 6.28 | −0.21 [−0.28, −0.15] | −0.28 [−0.34, −0.21] |
| M1-hq-time | 6.66 | 6.27 | 6.33 | 6.27 | 6.58 | 6.23 | −0.26 [−0.32, −0.20] | −0.33 [−0.39, −0.28] |
| M1-hq-noent | 6.77 | 6.39 | 6.54 | 6.38 | 6.67 | 6.36 | −0.14 [−0.22, −0.07] | −0.20 [−0.28, −0.13] |
| M2-h | 6.66 | 6.29 | 6.29 | 6.29 | 6.60 | 6.25 | −0.24 [−0.30, −0.19] | −0.31 [−0.37, −0.26] |
| M2-hq | 6.67 | 6.25 | 6.36 | 6.24 | 6.60 | 6.24 | −0.29 [−0.34, −0.23] | −0.32 [−0.37, −0.26] |
| M3-h | 6.69 | 6.32 | 6.49 | 6.31 | 6.59 | 6.27 | −0.22 [−0.28, −0.15] | −0.29 [−0.36, −0.23] |
| **M3-q** | 6.65 | 6.23 | 6.49 | 6.22 | 6.62 | 6.25 | **−0.31** [−0.36, −0.26] | −0.31 [−0.36, −0.26] |
| M3-hq | 6.69 | 6.34 | 6.53 | 6.34 | 6.60 | 6.28 | −0.19 [−0.25, −0.12] | −0.28 [−0.35, −0.22] |
| M4-hq | 6.81 | 6.32 | 6.46 | 6.31 | 6.75 | 6.31 | −0.22 [−0.28, −0.16] | −0.25 [−0.31, −0.18] |
| M4-hq-sa | 7.82 | 7.31 | 7.42 | 7.31 | 7.75 | 7.17 | +0.78 [+0.66, +0.89] | +0.60 [+0.50, +0.72] |

### Table 2.2. `train`: fitted on the training bags (epoch chosen on val), test MAE in years

| variant | E2L val | E2L test | E2L test TUH | E2L test HEEDB | E2 val | E2 test | Δ test vs M0, E2L | Δ test vs M0, E2 |
|---|---|---|---|---|---|---|---|---|
| M0-run | 7.25 | 7.00 | 6.55 | 7.02 | 7.43 | 7.25 | +0.50 [+0.45, +0.55] | +0.72 [+0.66, +0.79] |
| **M0** (benchmark) | 6.86 | 6.50 | 6.23 | 6.51 | 6.83 | 6.53 | — | — |
| M1-h | 6.59 | 6.20 | 6.23 | 6.20 | 6.54 | 6.20 | −0.30 [−0.35, −0.26] | −0.33 [−0.38, −0.28] |
| M1-q | 6.60 | 6.20 | 6.30 | 6.19 | 6.56 | 6.21 | −0.30 [−0.35, −0.26] | −0.32 [−0.36, −0.27] |
| M1-hq | 6.59 | 6.19 | 6.21 | 6.19 | 6.52 | 6.20 | −0.31 [−0.36, −0.27] | −0.34 [−0.39, −0.28] |
| M1-hq-time | 6.62 | 6.22 | 6.23 | 6.22 | 6.55 | 6.22 | −0.28 [−0.32, −0.24] | −0.31 [−0.36, −0.26] |
| M1-hq-noent | 6.63 | 6.23 | 6.29 | 6.23 | 6.53 | 6.23 | −0.27 [−0.32, −0.22] | −0.30 [−0.36, −0.24] |
| M2-h | 6.60 | 6.21 | 6.18 | 6.21 | 6.54 | 6.21 | −0.30 [−0.34, −0.25] | −0.32 [−0.37, −0.27] |
| M2-hq | 6.59 | 6.19 | **6.15** | 6.19 | 6.53 | 6.20 | −0.31 [−0.35, −0.27] | −0.33 [−0.38, −0.29] |
| M3-h | 6.61 | 6.21 | 6.30 | 6.21 | 6.54 | 6.21 | −0.29 [−0.34, −0.24] | −0.32 [−0.37, −0.26] |
| M3-q | 6.60 | 6.19 | 6.34 | 6.18 | 6.55 | 6.20 | −0.31 [−0.36, −0.26] | −0.33 [−0.38, −0.28] |
| M3-hq | 6.61 | 6.21 | 6.30 | 6.21 | 6.53 | 6.20 | −0.29 [−0.34, −0.24] | −0.33 [−0.38, −0.28] |
| **M4-hq** | 6.57 | **6.18** | 6.28 | **6.17** | 6.52 | **6.18** | **−0.33** [−0.37, −0.28] | **−0.36** [−0.40, −0.31] |
| M4-hq-sa | 6.59 | 6.17 | 6.23 | 6.17 | 6.54 | 6.19 | −0.33 [−0.38, −0.28] | −0.34 [−0.40, −0.29] |

### Table 2.3. E2L test MAE per dataset and report cohort (`train` protocol)

| variant | TUH normal (149) | TUH abnormal (125) | HEEDB normal (1,518) | HEEDB abnormal (2,253) | HEEDB unlabelled (3,567) |
|---|---|---|---|---|---|
| M0-run | 4.91 | 8.50 | 7.12 | 7.56 | 6.64 |
| M0 | 4.50 | 8.29 | 6.42 | 7.09 | 6.19 |
| M1-h | 4.63 | 8.14 | 5.84 | 6.80 | 5.97 |
| M1-q | 4.59 | 8.34 | 5.86 | 6.76 | 5.98 |
| M3-q | 4.64 | 8.37 | 5.82 | 6.75 | 5.98 |
| M2-hq | 4.55 | 8.06 | 5.85 | 6.80 | 5.96 |
| M4-hq | 4.59 | 8.31 | 5.83 | 6.76 | 5.95 |

What Table 2.1, Table 2.2 and Table 2.3 show:

1. **Refitting the head on recordings is the first big step.** `M0`, mean
   pooling of embeddings with the head refitted, beats the run's own mean of
   predictions by 0.47–0.72 y. With a linear head, pooling embeddings or
   predictions is the same thing, so the gain comes from the **refit**: a
   head trained per window with mixup, and read out by averaging, is
   miscalibrated at the recording level. This is a recording-level
   recalibration (it also includes the bias the window training leaves). It
   has to be the reference, or attention gets credit for it.
2. **Attention adds another 0.24–0.36 y on top of `M0`**, with every CI clear
   of zero. That is a total of about 0.8 y below E2L's reported 7.00 (test
   6.18–6.22).
3. **With little fitting data (`val_cf`, about 8,500 bags per fold), the
   descriptors carry the attention.** M1-q and M3-q are best. Adding $h_i$ to
   the scores (hq) is worse than either input alone, and removing the entropy
   term costs 0.1 y. The self-attention variant (M4-hq-sa) overfits badly.
4. **With the full training set (`train`, 84,891 bags) every variant ends up
   in the same place**: 6.17–6.23 on test, within the noise of each other.
   M4 and M4-sa, the largest pools, are now slightly best. The choice of pool
   and of attention input matters only when data is scarce.
5. **The gain is mostly on HEEDB** (normal 6.42 → about 5.84, abnormal
   7.09 → about 6.77). TUH abnormal moves a little, and TUH normal not at all
   (4.50 for `M0` vs 4.55–4.64). Same pattern as Proposal A.
6. **E2 shows the same ranking with slightly larger gains**, so nothing here
   is specific to the longer-trained E2L.

## 3. What the attention weights follow

With $w_i$ the pooling weight, $\hat y_i = g(h_i)$ the window's prediction
through the fitted head, and $e_i = |\hat y_i - y_c|$:
- $\rho_{\text{within}}$ is the mean over recordings of $\operatorname{Spearman}_{i\in W_c}(w_i, e_i)$ (Eq. (3.1) of [`window_confidence_results.md`](window_confidence_results.md));
- the "Δ low-w" and "Δ high-w" columns are the excess error of the lowest and highest within-recording weight fifths;
- $\rho(w, \cdot)$ are within-recording Spearman correlations with descriptors;
- "ρ(conc, rec err)" and "ρ(wSD, rec err)" correlate the weight concentration $1-\text{ESS}/n$ and the weighted SD of the window predictions with the recording's error;
- "MAE@50" is the error left after deferring the less confident half of recordings by wSD.

### Table 3.1. E2L attention analytics, test, `val_cf`

| variant | ρ within | share ρ<0 | Δ low-w (y) | Δ high-w (y) | ρ(w, minute) | ρ(w, ch. spread) | ρ(w, EMG) | mean max w | ρ(conc, rec err) | ρ(wSD, rec err) | MAE@50 by wSD | max \|Σw ŷ − ŷ\| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1-h | −0.11 | 0.62 | 2.52 | −0.77 | 0.17 | −0.17 | 0.05 | 0.13 | 0.22 | 0.17 | 5.62 | 0.00 |
| M1-q | −0.12 | 0.65 | 2.67 | −0.99 | 0.21 | −0.37 | 0.10 | 0.07 | 0.08 | 0.16 | 5.63 | 0.00 |
| M1-hq | −0.10 | 0.62 | 2.36 | −0.70 | 0.17 | −0.19 | 0.09 | 0.15 | 0.23 | 0.16 | 5.73 | 0.00 |
| M1-hq-noent | −0.13 | 0.64 | 2.74 | −0.91 | 0.22 | −0.31 | 0.11 | 0.24 | 0.19 | 0.17 | 5.64 | 0.00 |
| M2-hq | −0.06 | 0.57 | 1.79 | −0.49 | 0.12 | −0.17 | 0.06 | 0.13 | 0.21 | 0.12 | 5.88 | 0.00 |
| M3-h | **−0.16** | 0.64 | **2.99** | **−1.03** | 0.22 | −0.25 | 0.05 | 0.15 | 0.15 | **0.20** | **5.54** | 0.04 |
| M3-q | −0.13 | **0.67** | 2.69 | −0.98 | **0.29** | **−0.41** | 0.11 | 0.07 | 0.09 | 0.19 | 5.57 | 0.00 |
| M4-hq | 0.00 | 0.48 | 0.47 | 0.41 | −0.07 | −0.04 | −0.00 | 0.08 | 0.13 | 0.01 | 6.39 | 16.39 |

(The full table for both runs and both protocols is in
`docs/data/mil_stage1/*_analytics.csv`.)

### Figure 3.1. Excess window error by attention-weight decile, test, `val_cf`

The y axis is the mean $e_i$ minus the recording's mean $e$ (years), by
within-recording weight decile (1 = lowest weight), for 7,612 test
recordings. (a) E2L, (b) E2.

![Figure 3.1](figures/mil_stage1_weight_rank_val_cf.png)

What Table 3.1 and Figure 3.1 show:

* **The weights track window error, as strongly as A3 did.**
  $\rho_{\text{within}}$ is −0.11 to −0.16 (A3: −0.16 to −0.18), and in about
  65 % of recordings the lower-weight windows are the worse ones. The
  bottom-weight decile is 3.7–4 y worse than the recording average, and the
  top deciles about 1 y better (Figure 3.1).
* **They learned what A3 learned.** Weights rise with the window's minute
  (ρ +0.17 to +0.29; +0.55 for M1-q/M3-q in the `train` protocol: the bad
  first minutes of HEEDB) and fall with the across-channel power spread
  (ρ −0.17 to −0.41: one deviant channel). The EMG ratio is again slightly
  *positive*.
* **The weighted-sum identity holds:** for M0–M3 the recording prediction
  equals $\sum_i w_i \hat y_i$ (max error 0.00–0.04 y), so the weights are
  exactly the share each window's own prediction contributes. It does not
  hold for M4 or for the time-encoding variants. There the pooled value is a
  projection of $h_i$ (M4) or $h_i$ plus a time code, so $w_i$ is not the
  weight of $g(h_i)$. M4's flat decile curve is the result of that, not
  evidence that M4 ignores bad windows.
* **The recording confidence from the weights is a usable signal.** The
  weighted SD of the window predictions correlates with the recording error
  at ρ = 0.16–0.20. Deferring the less confident half by it leaves MAE
  5.5–5.6 y (vs 6.2–6.3 overall). Weight concentration works for $h$-based
  attention (ρ 0.15–0.23), not for descriptor-only attention.

## 4. Conclusions for Stages 3–4

1. **The bar for end-to-end MIL** is `M0` + attention on frozen E2L features:
   6.18–6.22 test (from 7.00). A trained bag model must beat about 6.2 to
   justify end-to-end training. Its recording-level loss must buy something
   beyond what a frozen-feature pooling plus a head refit already gives.
2. **Head refit and attention are separate gains** (about 0.5 and 0.3 y).
   Stage 3 reports both: `case_mean_mae` (mean of window predictions in the
   same model) next to the attention MAE.
3. **Descriptor-scored attention (M1-q / M3-q) is the robust choice** when
   the pooling sees few recordings, and any pool works with the full
   training set. Stage 3 therefore runs M0, M1-q, M3-q, M1-h, M1-hq-time,
   M2-hq and M1-q with feature retention; Stage 4 runs M0, M1-q and
   M1-hq-time, each against its window-level control.

## 5. Stages 3–4: the grid (running)

Every run uses E2L's data, split, channels and protocol; the ClearML project
is `LaBraM/heedb_mil`.
- **Stage 3:** warm start from E2L, 4 epochs, lr 1e-4, encoder frozen in epoch 0, pool and head lr ×5.
- **Stage 4:** from `labram-base`, 12 epochs, lr 2.8e-4.
- **Batches:** bag runs use 32 recordings × 8 windows per step; window controls use 8 windows per recording per epoch, batch 256.
- **Launchers:** `scripts/mil_runs/submit_grid.sh` (SageMaker, S3 compact cohort, streamed) and `scripts/mil_runs/run_local_grid.sh` (local GPU, local npy).

**First attempt (2026-11-09 01:10 UTC).** Spot capacity for g5 was
unavailable, so 5 jobs moved to on-demand. Every `ml.g5.2xlarge` job with
24 loader workers then **ran out of host memory** (32 GB) after 0.5–2.7 h:
each worker gradually copies the dataset's ~17 M-name file list. W0
(16 workers) and W1 (`ml.g5.4xlarge`, 64 GB, 32 workers) survived. Streamed
reads also make `g5.2xlarge` slow: W0 needed about 2 h per epoch at 16
workers, vs about 20 min per epoch for W1 on `g5.4xlarge`/32 workers, and
about 15 min locally.

**Second attempt (05:25 UTC).**
- Stage 4 (F0, F2, F1, F3) runs on the **local GPU**, one job after another, about 4 h each.
- Stage 3 W2 runs on on-demand and W3 on spot `g5.4xlarge` (32 workers).
- W4–W7 run on on-demand `g5.2xlarge` with 10 workers.

### Table 5.1. Jobs and results so far (recording MAE in years; the train-eval − val gap measures memorization)

| job | config | where | status | val | test | test TUH / HEEDB | train-eval (gap) |
|---|---|---|---|---|---|---|---|
| E2L | reference, best epoch 11 | — | — | 7.24 | 7.00 | 6.55 / 7.02 | 4.63 (2.61) |
| W0 | window-level control, warm start | SM g5.2xlarge spot | epoch 0 of 4 done | 7.53 | 7.34 | 6.68 / 7.36 | 4.99 (2.54) |
| W1 | M0 bag (mean of embeddings), warm start | SM g5.4xlarge | **done**, epoch 3 | 6.82 | 6.44 | 6.19 / 6.45 | 3.93 (2.89) |
| L1 | ungated ABMIL on $h$ (section 6) | local | **done**, epoch 3 | **6.55** | **6.09** | 6.11 / 6.09 | 3.71 (2.84) |
| W2 | M1-q | SM g5.4xlarge | rerun 05:25 | | | | |
| W3 | M3-q | SM g5.4xlarge spot | rerun 05:25 | | | | |
| W4 | M1-h (gated) | SM g5.2xlarge | rerun 05:25 | | | | |
| W5 | M1-hq + time | SM g5.2xlarge | rerun 05:25 | | | | |
| W6 | M2-hq | SM g5.2xlarge | rerun 05:25 | | | | |
| W7 | M1-q + retention 0.1 | SM g5.2xlarge | rerun 05:25 | | | | |
| F0 | window-level control, from `labram-base` | local | running (1st) | | | | |
| F2 | M1-q, from `labram-base` | local | queued (2nd) | | | | |
| F1 | M0 bag, from `labram-base` | local | queued (3rd) | | | | |
| F3 | M1-hq + time, from `labram-base` | local | queued (4th) | | | | |

What W1 and L1 already show, against the matched M0 bag model W1:

* **End-to-end mean pooling (W1, 6.44 test) ≈ the frozen-feature `M0`
  with a refitted head (6.50).** The recording-level loss with mean pooling
  buys little beyond what a head refit already gives.
* **End-to-end attention (L1, 6.09) beats end-to-end mean pooling (W1, 6.44)
  by 0.35 y** under the same training, and beats the frozen-feature attention
  of Stage 1 (6.18–6.22) by about 0.1 y. The gain is on HEEDB (6.45 → 6.09);
  TUH stays about level (6.19 vs 6.11).
* **No reduction of memorization yet.** The train-eval − val gap is 2.84–2.89
  for both bag runs vs 2.61 for E2L. These are warm starts from a model that
  already memorizes; Stage 4 tests whether the recording-level loss changes
  that.
* The window-level control W0 is still running. After epoch 0 (the warm-up to
  lr 1e-4) it is *worse* than E2L (7.34 test), so it will show how much of
  W1 / L1 is merely the extra epochs.

## 6. End-to-end, local: L1 (simplest ABMIL)

L1 is Ilse et al.'s original, **ungated** attention on the encoder's window
features, $a_i=\mathbf w^\top\tanh(\mathbf V h_i)$ with $L=128$, trained end
to end on the local A10G (`scripts/mil_runs/run_local_abmil.sh`; ClearML
`LaBraM/heedb_mil/heedb_mil_L1_abmil_local`). The setup:
- warm start from E2L, with E2L's data, split, channels and evaluation;
- 4 epochs, the encoder frozen in epoch 0, lr 1e-4 (pool and head ×5);
- 32 recordings × 8 random windows per step, mixup across recordings, entropy term 0.01.

It took 1 h 9 min (epoch 0 about 6 min, the unfrozen epochs about 15 min, plus about 4 min of evaluation each).

### Table 6.1. L1 per epoch, recording MAE (y)

"Mean-pool" is the mean of the same model's window predictions (head on
each $h_i$). The train-eval set is 1,000 fixed training recordings, scored
like val.

| epoch | val | test | val mean-pool | test mean-pool | train-eval | train-eval − val gap |
|---|---|---|---|---|---|---|
| 0 (encoder frozen) | 6.69 | 6.31 | 6.98 | 6.69 | 4.22 | 2.47 |
| 1 | 6.69 | 6.28 | 7.02 | 6.71 | 4.03 | 2.66 |
| 2 | 6.58 | 6.15 | 6.89 | 6.58 | 3.86 | 2.72 |
| **3** | **6.55** | **6.09** | 6.84 | 6.49 | 3.71 | 2.84 |
| *E2L (best epoch 11)* | *7.24* | *7.00* | | | *4.63* | *2.61* |
| *Stage 1 best, frozen features (`train` protocol)* | *6.52–6.62* | *6.17–6.23* | | | | |

### Table 6.2. L1 (epoch 3) test MAE per dataset and cohort vs. E2L

| | TUH normal | TUH abnormal | HEEDB normal | HEEDB abnormal | HEEDB unlabelled | all |
|---|---|---|---|---|---|---|
| E2L | 4.91 | 8.50 | 7.12 | 7.56 | 6.64 | 7.00 |
| L1 | **4.41** | **8.13** | **5.71** | **6.79** | **5.80** | **6.09** |

What Table 6.1 and Table 6.2 show:

* **End-to-end ABMIL reaches 6.09 test / 6.55 val**: 0.91 y better than E2L
  and about 0.1 y better than the best frozen-feature pool of Stage 1. It
  improves in every cohort, TUH included, unlike all post-hoc pooling so far.
* **Attention matters inside the trained model.** Its own mean of window
  predictions is 6.49 on test, so attention removes another 0.40 y. The
  window-level MAE gets *worse* as training goes on (8.35 → 8.93), as
  expected: windows are no longer trained to predict the age on their own,
  only to contribute to the recording.
* **Still improving at epoch 3** (val 6.69 → 6.55), so longer training is
  likely to help.
* **Memorization is not reduced yet.** The train-eval − val gap is 2.84 vs
  2.61 for E2L. The recording loss lowers train and val error together, but
  4 epochs from an already-memorizing E2L cannot show whether it stops
  memorization. That question belongs to Stage 4 (from `labram-base`, F0 vs
  F1–F3).
* **Attention vs. window error (test, epoch 3):**
  - $\rho_{\text{within}}=-0.11$, in 60 % of recordings the lower-weight windows are the worse ones;
  - the lowest-weight decile is 4.3 y worse than the recording average, the middle deciles about 1 y better;
  - the weights rise with the window's minute (ρ +0.33);
  - the largest weight is 1.47× uniform on average, so the attention is soft, not collapsed;
  - the weighted SD of the window predictions gives a recording confidence with ρ = 0.20, and MAE 5.43 on the more confident half.

The SageMaker control W0 (window-level training for the same 4 epochs from
E2L) is the check that the gain is not simply 4 more epochs.
