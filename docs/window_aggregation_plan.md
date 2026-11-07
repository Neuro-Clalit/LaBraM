# Recording-level aggregation of EEG windows: beyond the mean

Research note and plan, status as of 2026-11-07. Question: for recording-level
targets (brain age, normal/abnormal, pathology groups) the model sees 10 s
windows, and the recording prediction is the plain mean of its window
predictions over the first 5 minutes. How do we (1) weight windows by a
confidence that reflects their quality, (2) learn those weights instead of
hand-tuning them, and (3) obtain a *global* representation of the whole
recording rather than a bag of local ones? Background:
[`age_regression.md`](age_regression.md), the experiment registry
[`age_overfitting_experiments.md`](age_overfitting_experiments.md) and the
handcrafted-feature study [`eeg_feature_extraction.md`](eeg_feature_extraction.md).

## 1. Where we are

Table 1.1 lists every place the code turns windows into a recording-level
quantity today; Table 1.2 collects what the project has already measured about
window selection and pooling, which is what the proposals below have to beat.

### Table 1.1. Current window-to-recording paths

| Stage | What happens | Where |
|---|---|---|
| Training | every 10 s window of the trimmed recording is a sample and inherits the recording label, Eq. (1.2); windows of one recording are scattered over the epoch (`RecordingSubsetSampler` caps them per recording) | `labram/data/tuh_datasets.py`, `labram/data/samplers.py` |
| Model input | one window = `[C, 2000]`, up to 60 s by concatenating pickles (`data.window_sec`); mean over the `C x 10` patch tokens, then the head | `labram/models/neural_transformer.py::forward_features` |
| Evaluation | the first `data.eval_minutes` (5 min = 30 windows) after the 60 s trim; window predictions pooled per recording by `evaluation.agg_windows` = `mean` (also `median`, `max`; `entropy`/`vote` for classification only), Eq. (1.1) | `labram/utils/eval_metrics.py::aggregate_windows`, `labram/train/train_finetune.py::_window_and_case_metrics` |
| Offline re-pooling | EMG-index rejection / smooth down-weighting of cached window predictions (Eqs. (14.1)-(14.5) of `age_regression.md`) | `scripts/age_artifact_pooling.py`, `labram/eval/age_analysis.py` |
| Fold ensemble | median over the 5 CV fold models per recording | `labram/eval/cv_compare.py` |
| Global handcrafted features | per-recording median over the clean windows of minutes 1-6 of ~280 spectral/time features per channel + filter-bank Riemann tangent vectors; fed to ridge, or to the transformer as extra tokens (`model.session_features=tokens`) | `labram/features/tuab.py`, `dataset_maker/make_session_features.py` |

With $W_c$ the evaluated windows of recording $c$ and $\hat y_i$ the window
prediction in years (de-normalized), the recording prediction and the training
loss are (`aggregate_windows`, mode `mean`; `losses/regression.py`, Huber
$\delta = 1$ on z-scored targets):

$$
\hat y_c=\frac{1}{\lvert W_c\rvert}\sum_{i\in W_c}\hat y_i ,
\tag{1.1}
$$

$$
\mathcal L_{\text{win}}=\frac1N\sum_{i=1}^{N}\operatorname{Huber}_\delta\bigl(\hat y_i-y_{c(i)}\bigr),
\tag{1.2}
$$

where $y_{c(i)}$ is the age of the recording window $i$ belongs to: every
window carries the whole recording's label.

### Table 1.2. What has been measured so far

Val / test are recording MAE in years on the shared TUAB age split (all
recordings) unless stated; sources are the registry tables.

| Question | Result | Source |
|---|---|---|
| Mean vs. median pooling | logged side by side every epoch (`case_mean_*`, `case_median_*`); differences within the 0.15 y noise floor | `train_finetune.py` |
| Whole recording vs. first 5 min at eval | B (5 min) 8.40 / 8.91 vs. A (whole) in the same family; the 5 min budget was adopted for cost | registry Table 3.2 |
| Longer input (30 s windows, run C) | +0.44 y on test vs. B; memorizes more (train MAE 3.3 at the selected epoch) | registry Table 9.2 |
| Attention head over the patch tokens (N2L) | no gain vs. M1 / L1 (7.63 / 8.49) | registry Table 10.2 |
| EMG-index rejection / weighting of windows | val selects *no rejection* for 5 of 6 runs; best test-selected gain 0.02-0.16 y; keeping the cleanest 50 % costs 1.1-1.5 y | `age_regression.md` Table 14.2 |
| Handcrafted *recording-level* features + ridge, TUAB normal | test 6.72 y (10 groups), TUAB + TUEG healthy 6.75 y | feature notebook, section 9 |
| Minutes of clean EEG for age (handcrafted) | 10 s to 5 min: 10.3 to 8.8 y; still improving at 5 min | feature notebook, finding 4 |
| LaBraM fine-tune, best (M1) | 7.46 / 8.45; normal 6.60, abnormal 10.65; train MAE falls to about 1 y by epoch 14 (memorization) | registry Tables 6.2, 13 |

Three facts in Table 1.2 shape the plan.

1. **The information is in the aggregate, not in the window.** Handcrafted
   features that are *medians over all clean windows* of a recording beat the
   fine-tuned foundation model on normal recordings (6.72 vs. 6.60-6.75 y, with
   a far simpler model), and age error keeps falling as more minutes are
   pooled. A 10 s window is a noisy sample of a recording-level trait.
2. **Re-weighting after the fact does not help**, because the artifact-laden
   recordings are artifact-laden throughout (Table 14.2 of
   `age_regression.md`): there is nothing cleaner to average. Any gain must
   come from *training* the model to produce a recording-level estimate that
   is robust to its own bad windows, not from post-hoc pooling.
3. **Inherited window labels drive memorization.** Eq. (1.2) asks the encoder
   to map every window of a 65-year-old to 65, including the ones carrying
   only muscle. The cheapest way to do that is to recognize the recording
   (*The Identity Trap*, arXiv 2606.06647). A loss defined on the recording
   removes that incentive and adds one gradient per recording per step
   instead of 120 correlated ones.

## 2. What the literature does

Table 2.1 summarizes the aggregation strategies used by comparable work. The
field is moving from "average the crops" to learned, permutation-invariant
pooling with a loss on the bag.

### Table 2.1. Aggregation strategies in related work

| Work | Task | Aggregation | Take-away for us |
|---|---|---|---|
| Gemein et al. 2024, *Brain age revisited* (TCN, TUAB/TUEG) | brain age | cropped decoding, crop predictions averaged per recording; 5-seed ensembles; 6.60 y on TUAB normal | the published best is still a plain mean, so pooling is not where their gain came from (normal-only training, ensembles and augmentation are) |
| Engemann et al. 2022, reusable brain-age benchmark | brain age | filter-bank covariances *averaged over epochs*, then Riemannian tangent space + ridge; 7.75 y | a global second-order statistic over the whole recording is a strong, artifact-robust recording descriptor |
| Ilse, Tomczak & Welling 2018, attention-based deep MIL | bag classification | gated attention weights per instance, softmax over the bag, loss on the bag only, weights are interpretable | the standard learned-weights operator; Eq. (4.2) |
| Ma et al. 2026, *BridgeMIL* (arXiv) | EEG disease diagnosis, one label per subject | stage 1: self-supervised encoder aligning nearby windows and within-subject sub-bags; stage 2: attention-MIL aggregator, supervision only at subject level, feature retention | the closest match to our problem: inherited window labels are shown to be unreliable, MIL aggregation wins on 14 of 15 dataset x backbone settings; subject scarcity matters more than window scarcity |
| Xiao et al. 2025, *EmotionMIL* | trial-level EEG labels | retentive self-attention over segments, pseudo-bag augmentation | sub-sampling bags from a recording is a cheap augmentation (our Eq. (4.1)) |
| Schmidt et al. 2023, attention Gaussian process; Castro-Macias et al. 2025, probabilistic smooth attention | medical MIL | distribution over the attention weights; uncertainty correlates with error | a confidence per window *and* per recording comes for free from the attention |
| Kendall & Gal 2017; Aviv et al. 2025 (*MoGU*) | heteroscedastic regression; uncertainty-gated mixture | each predictor outputs mean and variance; variance is the gating/weight | inverse-variance pooling, Eq. (4.4), is the probabilistic version of "weight by confidence" |
| Sibilano et al. 2023; LEAD (arXiv 2502.01678) | subject-level EEG diagnosis | majority vote over epochs | still common, but it discards the confidence that a regression output carries |
| BioSerenity-E1 fine-tune (arXiv 2505.21507) | whole-recording normal/abnormal | global average pooling over the recording's token sequence | pooling *features* rather than *predictions* is the usual choice when the recording fits in memory |

## 3. Proposal A: confidence-weighted pooling without retraining (days)

This is the cheap half of the question: a per-window confidence and a weighted
sum, on cached window predictions of existing runs
(`<run>/analysis/window_predictions.parquet`). It will not move the MAE much
(Table 1.2, row 5), but it gives the confidence signal the clinical use needs
(which recordings to trust) and a baseline the trained variants must beat.
All three estimators below plug into `scripts/age_artifact_pooling.py`.

**A1, inverse-variance pooling with test-time augmentation.** Score each window
$M$ times under label-preserving perturbations (random crop offset of up to
5 s, hemisphere flip, `labram/train/augment.py`) and use the spread as the
window's uncertainty:

$$
\bar y_i=\frac1M\sum_{m=1}^{M}\hat y_{im},\qquad
\sigma_i^2=\frac1M\sum_{m=1}^{M}\bigl(\hat y_{im}-\bar y_i\bigr)^2,\qquad
\hat y_c^{\text{iv}}=\frac{\sum_{i\in W_c}\bar y_i/(\sigma_i^2+\varepsilon)}{\sum_{i\in W_c}1/(\sigma_i^2+\varepsilon)} ,
\tag{3.1}
$$

with $\varepsilon = 1\ \text{y}^2$ to cap the weight of a window that happens
to be invariant. A window whose prediction flips under a 5 s shift is not
reading a trait.

**A2, robust location instead of the mean.** The Huber M-estimate of the
recording age, solved by iteratively re-weighted means
(`scipy.optimize` or three fixed-point steps from the median):

$$
\hat y_c^{\text{M}}=\arg\min_{\mu}\sum_{i\in W_c}\rho_\delta\bigl(\hat y_i-\mu\bigr),\qquad
w_i=\min\Bigl(1,\ \frac{\delta}{\lvert\hat y_i-\mu\rvert}\Bigr),\qquad \delta=5\ \text{y},
\tag{3.2}
$$

which is the mean when the windows agree and the median when a few are wild;
the 10 %-trimmed mean is the non-iterative stand-in.

**A3, learned pooling weights from window descriptors.** The smallest "learn
the weights" model: a softmax over a linear score of per-window descriptors
$q_i$ (EMG ratio of Eq. (14.2) in `age_regression.md`, `max_abs` and
`min_std` from `window_stats.npz`, the TTA variance of Eq. (3.1), the
distance $\lvert\hat y_i-\operatorname{median}_c\rvert$ and the minute index),

$$
w_i=\frac{\exp(\theta^{\top}q_i)}{\sum_{j\in W_c}\exp(\theta^{\top}q_j)},\qquad
\hat y_c^{\theta}=\sum_{i\in W_c}w_i\,\hat y_i,\qquad
\theta^\star=\arg\min_\theta\frac1{\lvert\mathcal V\rvert}\sum_{c\in\mathcal V}\bigl\lvert\hat y_c^{\theta}-y_c\bigr\rvert ,
\tag{3.3}
$$

fitted on the validation recordings $\mathcal V$ (six to eight parameters,
so cross-validate over val folds and report the test MAE of the val-selected
$\theta$). If $\theta^\star$ comes out near zero the descriptors carry no
pooling information and we stop here; if not, the same descriptors become the
*prior* term of Eq. (4.2).

Deliverable: one table per run (M1, M1c, MT1, the CV folds) with mean,
median, Eqs. (3.1)-(3.3), and a recording-level selective-prediction curve
(coverage vs. MAE when the least confident recordings are deferred), the
regression analogue of `labram/eval/aggregation.py::selective_accuracy`.

## 4. Proposal B: train on recordings, not windows (the main lever)

Make the recording the training unit. A bag of $K$ windows of one recording
goes through the shared encoder, a learned pooling produces one recording
embedding, the head predicts once and the loss is taken once, on the
recording label. Nothing is predicted for an individual window, so no window
is asked to "be 65". This is attention-based MIL (Ilse et al. 2018) with the
LaBraM encoder as the instance network, the design BridgeMIL found to beat
inherited labels across backbones.

**Bags.** Each epoch, every training recording contributes one bag of $K$
windows drawn without replacement from its selected windows (random crop
offsets as today), re-drawn every epoch, Eq. (4.1); at evaluation the bag is
*all* windows of the eval budget, or of the whole recording, processed in
chunks under `no_grad`:

$$
B_c=\{x_{i_1},\dots,x_{i_K}\}\subset W_c,\qquad
h_i=f_\theta(x_i)\in\mathbb R^{D},
\tag{4.1}
$$

where $f_\theta$ is the pretrained `NeuralTransformer` up to its pooled
feature (`fc_norm` output, $D = 200$ for `labram_base`). With $K = 8$ and 8
recordings per step the encoder processes 64 windows, exactly today's batch,
so the step cost is unchanged; the epoch is $K/120$ of today's, so use
$K \cdot$ `epochs` $\approx$ `train_minutes_per_epoch` to keep the number of
optimizer steps comparable.

**Pooling with learned, quality-aware weights.** Gated attention over the bag,
with an additive prior from the window-quality descriptors $q_i$ of
Eq. (3.3) so the pooling can learn "muscle windows count less" directly, a
temperature $\tau$, and an entropy term that stops the attention from
collapsing onto one window early in training:

$$
\begin{aligned}
a_i&=\mathbf w^{\top}\bigl(\tanh(\mathbf V h_i)\odot\sigma(\mathbf U h_i)\bigr)+\boldsymbol\beta^{\top}q_i,\\
\alpha_i&=\frac{\exp(a_i/\tau)}{\sum_{j\in B_c}\exp(a_j/\tau)},\qquad
z_c=\sum_{i\in B_c}\alpha_i\,h_i,
\end{aligned}
\tag{4.2}
$$

$$
\mathcal L_{\text{bag}}=\frac1{\lvert\mathcal B\rvert}\sum_{c\in\mathcal B}
\operatorname{Huber}_\delta\bigl(g(z_c)-y_c\bigr)
+\lambda_H\,\frac1{\lvert\mathcal B\rvert}\sum_{c\in\mathcal B}\Bigl(\log K-H(\alpha_c)\Bigr),
\tag{4.3}
$$

where $\mathbf V,\mathbf U\in\mathbb R^{L\times D}$ ($L = 128$),
$\mathbf w\in\mathbb R^{L}$, $\boldsymbol\beta$ has one entry per descriptor,
$g$ is the existing head (linear, MLP or multi-task), $\mathcal B$ is the
batch of recordings, $H(\alpha_c)=-\sum_i\alpha_i\log\alpha_i$ and
$\lambda_H\in[0, 0.1]$ (0 = plain MIL). $\alpha_i$ is the learned per-window
confidence the question asks for: it is logged per recording and must be
checked against the EMG index (section 6).

**Variant B2, heteroscedastic windows with inverse-variance pooling.** Keep a
window-level output but make it a distribution, $(\mu_i, s_i)$ with
$\sigma_i^2=\exp(s_i)$, and pool by precision; the recording loss is the
Gaussian negative log-likelihood of the pooled estimate:

$$
\hat y_c=\frac{\sum_{i\in B_c}\mu_i/\sigma_i^2}{\sum_{i\in B_c}1/\sigma_i^2},\qquad
\sigma_c^2=\Bigl(\sum_{i\in B_c}\sigma_i^{-2}\Bigr)^{-1},\qquad
\mathcal L_{\text{hs}}=\frac1{\lvert\mathcal B\rvert}\sum_{c\in\mathcal B}\Bigl[\frac{(\hat y_c-y_c)^2}{2\sigma_c^2}+\tfrac12\log\sigma_c^2\Bigr].
\tag{4.4}
$$

This is Eq. (3.1) learned end to end: the only way a bad window can avoid
hurting the recording loss is to declare a large $\sigma_i^2$, so the model
learns to flag its own unreliable windows (Kendall & Gal 2017), and
$\sigma_c$ is a calibrated recording-level confidence. B2 keeps the head
window-local, so it is the smaller change from the current code; B1 lets
windows interact through $z_c$ and is the more expressive one.

**Variant B3, a recording transformer.** Replace Eq. (4.2) by one or two
transformer blocks over the window embeddings with a learned recording token
and a sinusoidal embedding of the window's minute offset,

$$
[\,r_c;\ \tilde h_1;\dots;\tilde h_K\,]
=\operatorname{Block}^{(1\dots 2)}\bigl([\,r_0;\ h_1+p(t_1);\dots;h_K+p(t_K)\,]\bigr),
\qquad \hat y_c=g(r_c),
\tag{4.5}
$$

a hierarchical model (10 s encoder, then a recording-level encoder) in the
spirit of the sequence-over-epochs design of EVA-Net (arXiv 2511.15393).
It can represent trends over the session (drowsiness, eyes open/closed
segments), which a set pooling cannot, at the price of more parameters to
fit on 2,000 training recordings; try it after B1.

### Table 4.1. Implementation map for Proposal B

| Piece | Change | Files |
|---|---|---|
| Bag sampler and collate | `RecordingBagSampler` (groups window indices by stem like `RecordingSubsetSampler`, yields $K$ per recording, re-drawn per epoch) and a collate returning `x [B, K, C, T]`, `y [B]`, `q [B, K, n_q]` (from `window_stats.npz` + the EMG index) | new `labram/data/bags.py`; `labram/data/window_qc.py` for $q_i$ |
| Config | `data.bag_size` ($K$, 0 = off), `model.bag_pool` = `attention` / `invvar` / `transformer` / `mean`, `model.bag_pool_prior` (use $q_i$), `loss.bag_entropy_weight` ($\lambda_H$), `loss.bag_nll` (B2) | `labram/configs/data_config.py`, `model_config.py`, `loss_config.py` |
| Pooling modules | `AttentionMILPool` (Eq. (4.2)), `PrecisionPool` (Eq. (4.4)), `RecordingBlockPool` (Eq. (4.5)); all return `(z, weights)` so the weights can be logged | new `labram/layers/bag_pool.py`, exported from `labram.layers` |
| Model | `NeuralTransformer.forward_bag(x, q)`: flatten `[B*K]`, encode, reshape, pool, head; optional `torch.utils.checkpoint` over the encoder for $K > 16$ | `labram/models/neural_transformer.py` |
| Training loop | when `bag_size > 0`: bag batches, loss on recording targets; mixup acts on bags (mix two recordings' $z_c$ and labels, C-Mixup style); augmentation unchanged per window | `labram/train/train_finetune.py`, `labram/runs/finetune_setup.py` |
| Evaluation | bag = all eval-budget windows per recording in chunks; metrics go through `aggregate_windows` with mode `none` on one row per recording, so per-cohort / per-dataset breakdowns and `log_summary_metrics` work unchanged; weights written to `<run>/analysis/bag_weights.parquet` | `train_finetune.py::evaluate`, `labram/eval/loading.py` |
| Tests | synthetic bags: shapes, permutation invariance of Eq. (4.2), eval equals train pooling for $K$ = all, 2-process DDP | `tests/test_bag_pooling.py` |

Both B1 and B2 reuse the pretrained weights unchanged; only the pooling and
the head are new, so the LP-FT recipe (train pooling + head for one epoch
with the encoder frozen, then unfreeze) is natural and cheap.

## 5. Proposal C: global descriptors of the whole recording

Proposal B still builds the recording from 10 s pieces. Three ways to add a
descriptor that is global by construction; all are complementary to B and
enter through the existing `model.session_features` paths.

**C1, codebook histogram of the whole recording.** The frozen VQNSP tokenizer
turns every patch into one of 8,192 codes at negligible cost (no gradients,
one pass over the recording). The recording's bag of codes,

$$
\pi_c[k]=\frac{1}{\lvert W_c\rvert\,N_p}\sum_{i\in W_c}\sum_{p=1}^{N_p}\mathbf 1\{z_{ip}=k\},
\qquad s_c=\sqrt{\pi_c}\ \ (\text{or }\log(\pi_c+\epsilon)),
\tag{5.1}
$$

with $N_p = C\times 10$ patches per window, is permutation-invariant, uses
*every* second of the recording (not 5 minutes), and is robust by
construction: one artifact window moves $1/\lvert W_c\rvert$ of the mass.
`bag_of_codes` already exists as a window-level feature source of the
codebook-regularized classifier (`labram/models/codebook_classifier.py`);
this lifts it to the recording. First test: ridge on $s_c$ against the
handcrafted ridge (6.72 y); then as a `session_features=head` input next to
$z_c$.

**C2, fuse the learned embedding with the handcrafted global features.** The
`tokens` and `head` modes of `model.session_features` already inject the
per-recording medians and Riemann tangent vectors
(`dataset_maker/make_session_features.py`). With B1 in place the natural
fusion is at the recording level,

$$
\hat y_c=g\bigl([\,z_c;\ \mathbf P s_c\,]\bigr),\qquad \mathbf P\in\mathbb R^{32\times d_s},
\tag{5.2}
$$

i.e. `session_features=head` applied to $z_c$ instead of to a window's
pooled feature, which is where it belongs: the handcrafted vector describes
the recording, not the window. A stacking baseline (ridge on
$[\hat y_c^{\text{LaBraM}}, \hat y_c^{\text{ridge}}]$ fitted on val) tells
how much the two views disagree before any joint training.

**C3, recording-level self-supervision for the aggregator (BridgeMIL stage
1).** Pretrain the pooling on the 10x larger TUEG corpus without labels:
pull together two sub-bags of the same recording and two neighbouring
windows, with variance/covariance regularization (VICReg) to avoid collapse.
It makes $z_c$ a stable recording descriptor before any age label is seen.
Caveat from *The Identity Trap*: this objective *is* a subject-identity
objective. It is harmless for age (a trait of the subject, and the splits are
subject-disjoint) and useful for pathology only if the sub-bags come from
*different sessions* of the same subject when available, so that
session-state artifacts are not what gets aligned.

**Not proposed: longer model inputs.** 30 s windows already cost 0.44 y
(Table 1.2) and 60 s is the hard limit of the time embedding. Global context
should come from pooling embeddings, not from a longer attention window.

## 6. Experiment plan and how to judge it

All runs on the shared TUAB age split (`data.split_json` of M1), scenario D
settings + mixup 0.4, 4 epochs, 3 seeds each; selection on val recording MAE;
compare with the `analyze-experiments` rules (noise floor about 0.15 y, a
change counts above 0.3 y on val with the train-val gap not larger).
Table 6.1 lists the runs in order; each answers one question, and a stage is
dropped when its answer is "no".

### Table 6.1. Planned runs

| # | Run | Config | Question | Cost |
|---|---|---|---|---|
| 1 | A1-A3 offline | cached predictions of M1, M1c, MT1, CV folds | is there any pooling-side gain and does any descriptor predict window reliability? | CPU hours |
| 2 | B1-mean | `bag_size=8`, `bag_pool=mean` | does the *recording-level loss alone* reduce memorization (train MAE at the selected epoch) and the val MAE? | = M1 |
| 3 | B1-attn | + `bag_pool=attention`, $\lambda_H = 0.01$ | do learned weights beat the mean within the same training? | = M1 |
| 4 | B1-attn-q | + `bag_pool_prior=true` | do the quality descriptors help the weights? | = M1 |
| 5 | B2 | `bag_pool=invvar`, `loss.bag_nll=true` | does a calibrated per-window variance work as well, and is $\sigma_c$ a usable confidence? | = M1 |
| 6 | B1-K16 / K4 | bag size sweep | how many windows per step are needed; eval with all windows vs. 5 min | 0.5-2 x M1 |
| 7 | B3 | `bag_pool=transformer` (1 block) | does order / trend information add anything over the set pooling? | 1.2 x M1 |
| 8 | C1 ridge, C2 stacking | offline | how far do global descriptors alone get, and how complementary are they? | CPU hours |
| 9 | C2 joint | best B + `session_features=head` on $z_c$ | does fusion beat stacking? | = M1 |
| 10 | normal-only + TUEG | best of the above with `case_filter=normal` and the TUEG cohort | comparable to Gemein's 6.60 and the ridge 6.72 | 3-10 x M1 |

Diagnostics that every B run must report, beyond MAE: (i) the Spearman
correlation between $\alpha_i$ (or $\sigma_i^{-2}$) and the EMG index of
Eq. (14.2) of `age_regression.md`, so that "confidence" is shown to track
quality rather than something arbitrary; (ii) the entropy of $\alpha_c$ per
recording (collapse check); (iii) a recording-level selective-prediction
curve (MAE vs. coverage, deferring by $\sigma_c$ or by attention entropy);
(iv) the normal / abnormal split, since the abnormal cohort carries the error
(10.65 vs. 6.60 y) and is where artifact-robust pooling should show first.

## 7. Risks and what to do about them

* **The val noise floor.** Three seeds per run, paired comparison on the same
  split; no claim below 0.3 y.
* **Bag-size mismatch between training ($K$ = 8) and evaluation (30-120).**
  Softmax weights renormalize, but attention trained on small bags can be
  over-confident on large ones; train with $K$ drawn from $\{4, 8, 16\}$ and
  evaluate at several budgets (run 6).
* **Attention collapse** onto one "easy" window per recording, which would
  recreate the window-level memorization. The entropy term of Eq. (4.3) and
  the per-recording entropy log catch it; B2 does not have this failure mode.
* **Memory at evaluation** for whole-recording bags: chunk the encoder pass,
  pool once; attention weights depend only on $h_i$, so chunking is exact.
* **Identity leakage in C3.** Only with subject-disjoint splits
  (`split_by=subject`, already the default) and, for pathology, cross-session
  positives.
* **Interaction with the multi-task head**: $z_c$ feeds `MultiTaskHead`
  unchanged, so the normal/abnormal branch gets the recording-level pooling
  for free; its BCE then also stops inheriting labels per window.

## References

- Gemein, Schirrmeister, Boedecker, Ball. *Brain age revisited: investigating the state vs. trait hypotheses of EEG-derived brain-age dynamics with deep learning.* Imaging Neuroscience 2024 (arXiv:2310.07029).
- Engemann et al. *A reusable benchmark of brain-age prediction from M/EEG resting-state signals.* NeuroImage 2022.
- Ilse, Tomczak, Welling. *Attention-based deep multiple instance learning.* ICML 2018.
- Ma et al. *Rethinking EEG-based disease diagnosis: decoupling instance representation learning from subject-level supervision (BridgeMIL).* arXiv 2026.
- Xiao et al. *EmotionMIL: an end-to-end multiple instance learning framework for emotion recognition from EEG signals.* IEEE TAC 2025.
- Schmidt et al. *Probabilistic attention based on Gaussian processes for deep multiple instance learning.* IEEE TNNLS 2023; Castro-Macias et al. *Probabilistic smooth attention for deep MIL in medical imaging.* Pattern Recognition 2025.
- Kendall, Gal. *What uncertainties do we need in Bayesian deep learning for computer vision?* NeurIPS 2017. Aviv et al. *MoGU: mixture-of-Gaussians with uncertainty-based gating.* arXiv 2025.
- *The Identity Trap in EEG foundation models.* arXiv 2606.06647.
- EVA-Net, arXiv 2511.15393 (brain age as anomaly detection with a sequence model over EEG epochs); BioSerenity-E1 fine-tuning for whole-recording abnormality, arXiv 2505.21507; LEAD, arXiv 2502.01678.
