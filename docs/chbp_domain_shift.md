# Why the brain-age models degrade on CHBP: domain-shift analysis

Results note, 2026-11-11. On the Cuban Human Brain Mapping Project (CHBP,
`docs/chbp_onboarding.md`) our brain-age models are much worse than on the
Temple (TUH) and Harvard (HEEDB) data they were trained on, although CHBP is a
healthy cohort. This note asks why, using only signals and frozen models: **CHBP
is never used for training, tuning or fitting** (its ages are used only to score).

The analysis has four parts:
- the models on age-matched training-domain recordings (Section 3);
- the signal features in which CHBP differs (Section 4);
- which features the errors follow (Section 5), including two acquisition
  sub-cohorts inside CHBP (Section 6);
- causal tests that transform the signals in both directions (Section 7).

It ends with the literature (Section 8), conclusions and the comparison with
published CHBP results (Section 9).

Code:
- `scripts/domain_shift_analysis.py`: per-recording signal features and the CHBP
  vs Temple / Harvard comparison;
- `scripts/make_transformed_cohort.py`: label-free signal transforms of an npy
  cohort (gain, band limits, per-recording amplitude, global scale factor,
  spectrum matching) and the age-matched reference sample;
- `scripts/domain_shift_figures.py`: the figures and the transform-test table;
- `scripts/external_runs/chbp_domain_shift.sh`: the whole pipeline.

Outputs:
- `docs/data/external_eval/chbp/domain_shift_tests.csv`;
- per-recording features and scores in `/data/runs/heedb/domain_shift/`.

## 1. Summary

1. **The degradation is real, not an artefact of CHBP's narrow age range.** On
   healthy Temple and Harvard recordings weighted to CHBP's ages, W7 has MAE
   5.90 / 4.45 years and r 0.70 / 0.80. On CHBP it has 8.66 and r 0.47
   (Table 3.1).
2. **Two properties of CHBP's recording chain reproduce the whole degradation
   on training-domain recordings** (Table 7.1):
   - **Amplitude is about 0.6×** (channel std 5.8 vs 9.9 / 12.8 µV). It adds a
     pure offset: about +6 years of over-prediction with r unchanged.
   - **The 0.5–50 Hz recording band.** The 50 Hz low-pass alone drops r from
     0.66–0.71 to 0.40–0.44. The models use content above 50 Hz that CHBP does
     not have (its 45–70 Hz power is 4–5 % of the training data's).
   - Together, Temple / Harvard recordings degrade to MAE 11.5–13.7,
     r 0.36–0.37.
3. **CHBP holds two acquisition sub-cohorts.** The 64-channel cap (12-bit EDF,
   n = 161) reaches r 0.58. The 120-channel cap (16-bit EDF, n = 89) has
   r 0.27 and +9 years of bias, with a frequency-dependent spectral
   difference (Table 6.1).
4. **The population is not what drives the error.** CHBP is alert, eyes-closed
   and artifact-free: strong alpha, steeper 1/f slope, no artifact windows. In
   the literature these features read as *younger*, not older (Section 8).
5. **Label-free normalization of CHBP removes the bias but cannot restore the
   ranking** (Table 7.2).
   - Best: spectrum matching to the training data, MAE 7.01–7.89 (E2L 7.01, the
     only value under the 7.27 mean-age baseline); on the 64-channel cap 6.27.
   - r stays at 0.43–0.49, because the missing high-frequency content cannot be
     re-created.
   - The robust fix is on the training side and needs no CHBP data
     (Section 9.2).

## 2. Data and methods

**Sets.**
- *Temple* and *Harvard* are the recordings of the `tuh_heedb_v1` validation
  split with a normal report, aged 18–68: 1,198 (TUH) and 1,233 (HEEDB). None
  were used for training.
- *CHBP* is all 250 recordings of `chbp_v1`.
- For the transform tests (Section 7), 150 Temple and 150 Harvard recordings
  were drawn from those sets, weighted to CHBP's ages (2-year bins, seed 0;
  `make_transformed_cohort.py sample`).

**Signal segment.** Every feature uses what the models see: the first 5 minutes
after a 60 s trim, the 19 10-20 channels (A1, A2, T1, T2 dropped), common average
reference, µV at 200 Hz. In CHBP this segment lies inside the first eyes-closed
block.

**Models.** E2L, L1, W4 and W7, trained on `tuh_heedb_v1`; predictions are
de-normalized with the training age mean / std, 53.52 / 19.29 years.

**Age matching.** Recording $i$ of a reference set with age $a_i$ in 2-year bin
$b(i)$ gets the weight

$$ w_i = \frac{h_{b(i)}}{n_{b(i)}} \tag{2.1} $$

Here $h_b$ is the share of CHBP recordings in bin $b$ and $n_b$ the number of
reference recordings in it. Weighted statistics use $w_i / \sum_j w_j$.

**Metrics.** With $e_i = \hat y_i - y_i$ (predicted minus true age, years):

$$ \mathrm{MAE} = \frac{\sum_i w_i \lvert e_i\rvert}{\sum_i w_i}, \qquad
   \mathrm{bias} = \frac{\sum_i w_i e_i}{\sum_i w_i} \tag{2.2} $$

$w_i = 1$ for CHBP and in Section 7. $r$ is the (weighted) Pearson correlation
of $\hat y$ and $y$ (2.3).

## 3. Is it the age range?

CHBP is young and narrow (18–68, mean 31.7, SD 9.3), which caps any model's R²
(Section 9.3). The question here is whether the models are also poor on
*training-domain* recordings of the same ages.

### Table 3.1. Age-matched comparison, every window

MAE and bias (2.2), weighted to CHBP's ages (2.1); bias for ages 18–29; r (2.3).
Predictions over every window (the training-domain validation predictions are
scored that way).

| model | set | n | MAE | bias | bias 18–29 | r |
|---|---|---|---|---|---|---|
| E2L | Harvard normal | 1,233 | 4.87 | +2.79 | +4.44 | 0.77 |
| E2L | Temple normal | 1,198 | 6.47 | +4.81 | +6.96 | 0.68 |
| E2L | **CHBP** | 250 | **9.35** | +6.96 | +11.38 | 0.45 |
| L1 | Harvard normal | 1,233 | 4.38 | +1.34 | +1.75 | 0.80 |
| L1 | Temple normal | 1,198 | 5.79 | +3.32 | +4.73 | 0.69 |
| L1 | **CHBP** | 250 | **9.11** | +5.25 | +9.04 | 0.46 |
| W4 | Harvard normal | 1,233 | 4.37 | +1.50 | +2.04 | 0.80 |
| W4 | Temple normal | 1,198 | 5.86 | +3.62 | +5.08 | 0.70 |
| W4 | **CHBP** | 250 | **8.92** | +5.03 | +8.90 | 0.47 |
| W7 | Harvard normal | 1,233 | 4.45 | +1.34 | +1.78 | 0.80 |
| W7 | Temple normal | 1,198 | 5.90 | +3.51 | +4.80 | 0.70 |
| W7 | **CHBP** | 250 | **8.66** | +4.78 | +8.58 | 0.47 |

**What Table 3.1 shows:**
- At CHBP's ages the models are good on their own domain: r 0.70 (Temple) and
  0.80 (Harvard), MAE 4.4–6.5.
- On CHBP, r falls to 0.45–0.47 and the MAE rises by 3–5 years. The narrow age
  range is therefore not the reason.
- **Part of the over-prediction of young adults is generic.** Temple's healthy
  18–29-year-olds are over-predicted by +4.7 to +7.0 years too. CHBP adds about
  +4 years on top.

## 4. What differs in the signal

### Features

For recording $r$, let $x_{c,k}$ be channel $c$ in 10 s window $k$, after the
common average reference. Statistics are taken per window and then the median
over windows. Clinical recordings carry large settling and movement artifacts
(Harvard: 20 % of windows), which would otherwise dominate.

$$ \sigma_r = \operatorname{median}_k \operatorname{median}_c \operatorname{std}(x_{c,k}) \tag{4.1} $$

$P_r(f)$ is the channel-median Welch PSD (2 s Hann segments, median averaging),
in µV²/Hz. The absolute power of band $B$ is
$\tfrac{1}{\lvert B\rvert}\sum_{f\in B} P_r(f)$ (4.2). The relative power is
$\sum_{f\in B} P_r(f) / \sum_{1\le f<45} P_r(f)$ (4.3).

The aperiodic component is the OLS line

$$ \log_{10} P_r(f) = o - \chi \log_{10} f, \qquad f \in [2, 40]\ \text{Hz} \setminus [6, 14]\ \text{Hz} \tag{4.4} $$

with exponent $\chi$ and offset $o$. The alpha peak is the frequency
(6–14 Hz) and height of the largest O1/O2 log-PSD excess over line (4.4) (4.5).

A feature's separation between CHBP and a reference set is

$$ \mathrm{AUC} = P(X_{\mathrm{CHBP}} > X_{\mathrm{ref}}) + \tfrac12 P(X_{\mathrm{CHBP}} = X_{\mathrm{ref}}) \tag{4.6} $$

with the reference recordings weighted by (2.1). 0.5 is no difference; 0 or 1
is complete separation.

### Table 4.1. CHBP vs age-matched Temple and Harvard normals

Medians (reference medians weighted by (2.1)); AUC (4.6). Power in µV²/Hz;
"share" columns are fractions.

| feature | CHBP | Temple | Harvard | AUC vs Temple | AUC vs Harvard |
|---|---|---|---|---|---|
| channel std (4.1), µV | 5.84 | 9.95 | 12.81 | 0.13 | 0.05 |
| 99th percentile of \|x\|, µV | 20.2 | 42.7 | 70.6 | 0.17 | 0.11 |
| share of windows with a channel std > 100 µV | 0.000 | 0.033 | 0.200 | 0.31 | 0.08 |
| power 0.1–1 Hz (4.2) | 5.0 | 43.2 | 90.5 | **0.03** | **0.01** |
| power delta 1–4 Hz | 1.86 | 3.86 | 4.36 | 0.21 | 0.17 |
| power theta 4–8 Hz | 0.70 | 1.10 | 1.07 | 0.32 | 0.34 |
| power alpha 8–13 Hz | 1.98 | 1.05 | 0.92 | 0.65 | 0.69 |
| power beta 13–30 Hz | 0.159 | 0.266 | 0.244 | 0.32 | 0.35 |
| power 30–45 Hz | 0.017 | 0.069 | 0.056 | 0.15 | 0.15 |
| power 45–70 Hz | 0.001 | 0.036 | 0.028 | **0.03** | **0.02** |
| relative delta (4.3) | 0.255 | 0.401 | 0.479 | 0.24 | 0.15 |
| relative alpha | 0.438 | 0.173 | 0.152 | 0.82 | 0.87 |
| relative beta | 0.121 | 0.150 | 0.134 | 0.41 | 0.46 |
| aperiodic exponent $\chi$ (4.4) | 1.70 | 1.33 | 1.42 | 0.74 | 0.74 |
| aperiodic offset $o$ | 1.11 | 1.02 | 1.06 | 0.53 | 0.52 |
| alpha peak frequency (4.5), Hz | 10.0 | 10.0 | 10.0 | 0.53 | 0.49 |
| alpha peak height (4.5), log10 | 1.89 | 0.92 | 0.84 | 0.82 | 0.84 |
| posterior / frontal alpha | 1.91 | 1.46 | 1.47 | 0.68 | 0.68 |
| 95 % spectral edge, Hz | 20.5 | 28.5 | 24.5 | 0.25 | 0.34 |
| Hjorth complexity | 2.23 | 3.61 | 4.94 | 0.09 | 0.03 |
| mean \|correlation\| between channels | 0.46 | 0.41 | 0.46 | 0.67 | 0.48 |

### Figure 4.1. Spectra

![PSD](figures/chbp/chbp_shift_psd.png)

(a) Median PSD (µV²/Hz, log scale) over recordings and channels, common
average reference: Temple normal (n = 150), Harvard normal (n = 150), both
age-matched, and CHBP (n = 250). (b) CHBP's median PSD divided by each
reference's. The dashed line is the power response $\lvert H(f)\rvert^2$ of
CHBP's recording filters as specified: first-order high-pass 0.5 Hz, second-order
Butterworth low-pass 50 Hz (Valdes-Sosa et al. 2021).

### Table 4.2. EDF calibration headers

From the EDF headers of channel O1 (CHBP: all 250 recordings of release 1.0;
TUAB: 61 sampled normal recordings).

| source | files | signals per file | physical range (µV) | digital range | resolution (µV) |
|---|---|---|---|---|---|
| CHBP, 64-channel cap | 161 | 62 | ±246 | 12-bit, ±2047 | 0.116 |
| CHBP, 120-channel cap | 89 | 123 | ±5,063 | 16-bit, ±32767 | 0.155 |
| TUAB | 58 of 61 | — | ±5,000 | 16-bit | 0.153 |

**What Tables 4.1, 4.2 and Figure 4.1 show.** CHBP differs from the training
data in two kinds of property.

**Recording chain:**
- **Amplitude.** About 0.6× in channel std and 0.3–0.5× in the dynamic range
  actually used.
- **Band limits.** Very low frequencies are 0.06–0.12×, and 45–70 Hz is 0.04–0.05×.
  These two features separate CHBP almost completely (AUC 0.01–0.03).
- **The filters alone do not explain the high-frequency deficit.**
  Figure 4.1(b): from about 20 Hz upward the observed ratio lies far below the
  filter response. CHBP has about 3× (30 Hz) to 10× (45 Hz) less power than the
  filters account for. That fits recordings without muscle (EMG) activity under a
  technician watching for it, against clinical recordings.
- **The 64-channel files have only ±246 µV of range**, but clipping is negligible
  (in 40 files, a median of 0.002 % of samples sit at the per-channel maximum).

**Population and state:**
- An alert eyes-closed state: 2.5–2.9× relative alpha, 2× alpha peak height,
  a steeper 1/f slope (1.70 vs 1.33–1.42).
- No artifact windows (Harvard: 20 %).
- **The alpha peak frequency is the same** (10 Hz).

## 5. Which features the errors follow

### Partial correlation of error with features

Within each set, the Spearman correlation of the error $e_i$ with a feature,
both rank-residualized on age:

$$ \rho_{\mathrm{partial}} = \operatorname{corr}\big(R(e) - \widehat{R(e)}\,\vert\,R(y),\; R(z) - \widehat{R(z)}\,\vert\,R(y)\big) \tag{5.1} $$

$R(\cdot)$ is the rank, $z$ the feature, and $\widehat{\cdot}\,\vert\,R(y)$ the OLS
fit on the age rank. A negative value means the model predicts older when the
feature is lower.

### Table 5.1. Partial correlation (5.1) of the error with signal features, every window

| feature | W7 CHBP | W7 Temple | W7 Harvard | E2L CHBP | E2L Temple | E2L Harvard |
|---|---|---|---|---|---|---|
| channel std (4.1) | **−0.31** | 0.00 | −0.05 | **−0.31** | 0.01 | −0.02 |
| 99th percentile of \|x\| | −0.27 | −0.01 | −0.05 | −0.28 | −0.02 | −0.03 |
| power 0.1–1 Hz | −0.18 | −0.01 | −0.03 | −0.20 | 0.02 | −0.01 |
| power delta | **−0.33** | 0.04 | −0.07 | **−0.34** | 0.07 | −0.06 |
| power alpha | −0.31 | −0.08 | −0.02 | −0.30 | −0.10 | −0.03 |
| power 45–70 Hz | −0.23 | −0.01 | 0.05 | −0.26 | −0.05 | 0.06 |
| aperiodic offset | −0.24 | 0.03 | −0.07 | −0.24 | 0.07 | −0.07 |
| aperiodic exponent | −0.04 | 0.03 | −0.05 | −0.04 | 0.10 | −0.05 |
| alpha peak frequency | −0.04 | −0.13 | −0.16 | −0.03 | −0.15 | −0.21 |
| relative alpha | −0.15 | −0.16 | 0.00 | −0.13 | −0.19 | −0.03 |

### Figure 5.1. Amplitude and over-prediction

![amplitude](figures/chbp/chbp_shift_amplitude.png)

(a) Channel std (4.1, µV, log scale) against age for Temple normal
(n = 1,198), Harvard normal (n = 1,233) and CHBP (n = 250). (b) W7's mean error
(2.2), every window, per channel-std bin; bins with at least 8 recordings.

**What Table 5.1 and Figure 5.1 show:**
- **Within CHBP the error follows absolute amplitude** (ρ ≈ −0.3 for channel std
  and delta, alpha and 45–70 Hz power): the lower the amplitude, the older the
  prediction.
- **Within Temple and Harvard amplitude has no such effect** (|ρ| ≤ 0.08). There
  the error follows the alpha peak frequency, as physiology predicts.
- **CHBP lies at the edge of the training distribution.**
  - Half of CHBP is below 5.84 µV, a range that holds only 2.3 % of Harvard and
    7.1 % of Temple healthy adults (Figure 5.1(a)).
  - In the training data those low-amplitude recordings are mostly older people
    (mean age 46–49).
  - The few young ones are over-predicted by +6 to +12 years (n = 13).
  - So the models learned "low amplitude = old", and CHBP sits where that rule
    applies (Figure 5.1(b)).

## 6. Two acquisition sub-cohorts in CHBP

### Table 6.1. CHBP by cap

First 5 minutes. W7 / E2L: MAE and bias (2.2), r (2.3). The predict-the-mean
baseline uses each group's own mean. Bias by age is W7's, over every window.

| | 64-channel cap | 120-channel cap |
|---|---|---|
| recordings, EDF encoding | 161, 12-bit (±246 µV) | 89, 16-bit (±5,063 µV) |
| age, mean ± SD | 31.1 ± 9.1 | 32.8 ± 9.6 |
| channel std (4.1), µV | 5.90 | 5.53 |
| power ratio 120 / 64 at 4 / 10 / 20 / 40 / 45 Hz | — | 0.97 / 0.61 / 0.65 / 1.38 / 2.86 |
| predict the group mean, MAE | 7.19 | 7.32 |
| W7 MAE / bias / r | **7.26** / +2.47 / **0.58** | 12.45 / +9.09 / 0.27 |
| E2L MAE / bias / r | 7.58 / +4.98 / 0.55 | 13.10 / +10.80 / 0.27 |
| W7 bias at 18–29 / 30–39 / 40–68 | +6.4 / −0.4 / −0.4 | +13.1 / +9.7 / −2.9 |

**What Table 6.1 shows:**
- On the 64-channel cap the models behave much like on Temple: r 0.58, bias near
  zero from age 30.
- The 120-channel cap is the problem: r 0.27 and +9 to +11 years of bias.
- Its broadband amplitude is the same. Its spectrum differs in a
  frequency-dependent way: about 40 % less alpha and beta, more power at
  40–45 Hz. That is not a calibration factor.
- Topography is plausible in both caps. Alpha is largest at O1/O2, and the
  inter-channel correlation structures agree (r = 0.97 between caps). There is
  no sign of swapped labels.
- The electrode coordinates shipped with release 1.1 are the same template for
  both caps, so they cannot show real placement differences.
- The data paper itself is inconsistent about the amplifier gain (10,000 in the
  text, "20,000/1000" in its Table 7). The cause cannot be settled from the
  released data.

## 7. Causal tests with frozen models

Every transform is label-free (`make_transformed_cohort.py`), applied to whole
recordings $x$ (µV) before the standard pipeline:

$$ x' = k\,x \tag{7.1} $$

$$ x' = h_{\mathrm{LP}} \ast h_{\mathrm{HP}} \ast x \tag{7.2} $$

(7.2) is zero-phase Butterworth: high-pass order 2, low-pass order 4, applied
forwards and backwards.

$$ x' = \frac{\sigma^\ast}{\sigma_r}\,x \tag{7.3} $$

In (7.3), $\sigma_r$ is (4.1) over the whole recording and
$\sigma^\ast = 10.05$ µV, the median of the reference sample.

$$ G_r = \frac{1}{19\,\lvert F\rvert} \sum_{c} \sum_{f\in F} \log_{10} P_{r,c}(f),\quad F = [1, 19]\ \text{Hz}, \qquad x' = 10^{(G^\ast - G_r)/2}\,x \tag{7.4} $$

(7.4) is the global scale factor (GSF) of the Cuban qEEG norms (Hernández et al.
1994; HarMNqEEG code), with $G^\ast = -0.058$, the reference median.

$$ X'(f) = X(f)\,\operatorname{clip}\!\Big(\sqrt{\bar P_{\mathrm{ref}}(f) / \bar P_{\mathrm{CHBP}}(f)},\ 0.1,\ 10\Big) \tag{7.5} $$

In (7.5), $X$ is the Fourier transform of a channel and $\bar P$ the median PSD
over recordings and channels. The gain is one cohort-level curve for all of CHBP
(spectrum matching).

### Table 7.1. Forward tests: Temple + Harvard normals made CHBP-like

n = 300 (150 + 150, age-matched); first 5 minutes; MAE, bias (2.2), r (2.3).
All four models: `docs/data/external_eval/chbp/domain_shift_tests.csv`.

| transform of the training-domain recordings | W7 MAE | W7 bias | W7 r | E2L MAE | E2L bias | E2L r |
|---|---|---|---|---|---|---|
| none | 5.99 | +2.79 | 0.70 | 7.02 | +5.10 | 0.66 |
| amplitude × 0.62 (7.1) | 9.26 | +8.59 | 0.70 | 11.34 | +10.91 | 0.66 |
| high-pass 0.5 Hz (7.2) | 6.68 | +4.12 | 0.67 | 7.71 | +5.96 | 0.63 |
| low-pass 50 Hz (7.2) | 7.36 | +0.42 | **0.44** | 8.04 | +3.75 | **0.40** |
| band 0.5–50 Hz (7.2) | 8.54 | +3.75 | 0.40 | 9.75 | +6.79 | 0.38 |
| × 0.62 and band 0.5–50 Hz | **11.46** | +9.31 | 0.37 | **13.71** | +12.59 | 0.37 |
| per-recording amplitude (7.3) | 6.63 | +3.09 | 0.65 | 7.90 | +5.45 | 0.59 |
| per-recording GSF (7.4) | 6.44 | +2.61 | 0.67 | 7.33 | +4.88 | 0.62 |

### Table 7.2. Inverse tests: CHBP normalized towards the training data

CHBP, n = 250 as recorded, first 5 minutes; same columns. The last two columns
are the 64-channel cap only (n = 161; predict-the-mean 7.19).

| normalization of CHBP | W7 MAE | W7 bias | W7 r | E2L MAE | E2L bias | E2L r | 64-ch E2L MAE | 64-ch E2L r |
|---|---|---|---|---|---|---|---|---|
| none | 9.10 | +4.83 | 0.46 | 9.55 | +7.05 | 0.44 | 7.58 | 0.55 |
| amplitude × 1.615 (7.1) | 8.31 | −0.98 | 0.46 | 7.89 | +0.83 | 0.44 | 6.89 | 0.52 |
| per-recording amplitude (7.3) | 8.09 | −1.04 | 0.46 | 7.64 | +0.78 | 0.43 | 6.99 | 0.48 |
| per-recording GSF (7.4) | 8.50 | +2.87 | 0.46 | 8.86 | +5.11 | 0.44 | 7.56 | 0.49 |
| spectrum matching (7.5) | **7.89** | −2.60 | 0.49 | **7.01** | +0.48 | 0.48 | **6.27** | **0.56** |

### Figure 7.1. Transform tests, W7

![tests](figures/chbp/chbp_shift_tests.png)

(a) Mean error (2.2) and (b) MAE of W7 for every test: Temple + Harvard
(n = 300, blue) and CHBP (n = 250, aqua), first 5 minutes.

**What Tables 7.1, 7.2 and Figure 7.1 show:**
- **The amplitude shift is a pure offset.** × 0.62 leaves r unchanged and adds
  about +6 years to every model.
- **The band limit breaks the ranking.** The 50 Hz low-pass alone takes r from
  0.66–0.71 to 0.40–0.44 on the training domain; the high-pass costs much less. The
  models rely on content above 50 Hz, or at least have never seen EEG without
  it. Content there is mostly muscle and line-noise residue in clinical
  recordings.
- **Together, the two reproduce CHBP's degradation on Temple / Harvard
  recordings** (MAE 11.5–13.7, r 0.36–0.37), even worse than CHBP itself.
- **On CHBP, amplitude normalization removes the bias but not the ranking loss.**
  Per-recording amplitude does as well as a global gain. The GSF is weaker,
  because CHBP's strong alpha inflates its log-power mean, so it under-corrects.
- **Spectrum matching works best** (E2L 7.01; 64-channel cap 6.27, r 0.56). The
  content CHBP never recorded above 50 Hz cannot be put back, so r stays below
  0.5.
- **Per-recording amplitude or GSF normalization costs the frozen models only
  0.3–0.9 years** on the training domain (Table 7.1). Amplitude carries little
  age information there.

## 8. What the literature says

**Device, gain and harmonization**
- CHBP's own group removes a per-subject **global scale factor** before
  building qEEG norms: the geometric mean of power over channels and
  frequencies after average reference. It explains 42 % of the variance of
  age-corrected log spectra and is partly age-dependent (Hernández et al.
  1994).
- HarMNqEEG models device / site *batch* effects on top of it. CHBP is one of
  its batches, "Medicid-5 CHBMP" (Li et al. 2022, code).
- A successor normalizes power per electrode below 20 Hz, "to avoid the region
  where device-specific high-frequency roll-off differences become
  non-negligible". It warns that such normalization absorbs real amplitude
  effects (Bosch-Bayard et al. 2026).

**Age effects in the spectrum**
- With age, broadband power (aperiodic offset) and the 1/f exponent decrease,
  and alpha peak frequency slows (Voytek et al. 2015; Donoghue et al. 2020:
  exponent 1.43 vs 0.75, offset −11.1 vs −11.9, alpha 10.7 vs 9.6 Hz in 20–30
  vs 60–70-year-olds; Merkin et al. 2023; Tröndle et al. 2023).
- A gain loss therefore mimics ageing through the offset. CHBP's steep slope
  and strong alpha mimic youth.
- In simulations of a ridge brain-age model, a pure input gain or damping
  shifts the brain-age gap systematically (Gemein et al. 2026).

**Clinical vs research recordings**
- About half of routine adult clinical EEGs at MGH (a Harvard hospital) reach
  early sleep (Karakis et al. 2012).
- CHBP's technicians watched "to prevent drowsiness" (Valdes-Sosa et al. 2021).
- Muscle and other peripheral signals carry age information in TUAB-type data
  (Bomatter et al. 2024). CHBP lacks them.

**Band limits and cross-dataset transfer**
- The most important setting for cross-dataset brain-age generalization in a
  1,805-configuration study was a broadband 1–45 Hz input (Tveitstøl et al.
  2026).
- The CHBP benchmark itself used 0.1–49 Hz (Engemann et al. 2022).
- Models trained on TUEG with per-epoch z-scored 1–45 Hz input transfer to LEMON
  and TDBRAIN with R² 0.48–0.63 (Hajhassani et al. 2026).

**Label-free alignment**
- Riemannian re-centering raised TUAB → LEMON R² from about 0.26 to
  0.44 ± 0.06 (Mellot et al. 2023).
- It fails when the target's age distribution differs from the source's. On
  HarMNqEEG, which includes CHBP: R² −0.35, MAE 16.97, because the dataset mean
  covariance correlates with age (Mellot et al. 2024, GOPSA). CHBP is such a
  case: young and narrow.

**Bias correction**
- Corrections that use each test subject's age inflate R² (Butler et al. 2021).
- Only corrections fitted on training data may be applied to an external test
  (de Lange & Cole 2020; Beheshti et al. 2019).

## 9. Conclusions

### 9.1 Why the models degrade on CHBP, ranked by measured effect

1. **CHBP's 0.5–50 Hz recording band, above all the 50 Hz low-pass.**
   - Measured effect: r 0.66–0.71 → 0.40–0.44 on the training domain
     (Table 7.1).
   - CHBP's 45–70 Hz power is 4–5 % of the training data's, and below 1 Hz
     6–12 % (Table 4.1).
   - The models were trained on 0.1–75 Hz clinical EEG, where muscle activity
     fills that range.
2. **Amplitude about 0.6× the training data's.**
   - Measured effect: about +6 years of over-prediction, r unchanged
     (Table 7.1).
   - The models have no amplitude normalization and learned that low amplitude
     means old. CHBP sits where few young training recordings are (Figure 5.1).
   - The source of the low amplitude (gain, electrodes, export scale or
     population) cannot be separated with the released data. The paper's two
     gain values disagree.
3. **The 120-channel cap.**
   - A third of CHBP; a frequency-dependent difference from the 64-channel cap.
   - Bias +9 years, r 0.27, against r 0.58 on the 64-channel cap (Table 6.1).
4. **Not the cause, but part of the mismatch: population and state.**
   - Alert, clean, eyes-closed recordings with strong alpha and a steep slope.
   - These read as *younger*. The domain is still different from clinical
     routine EEG, where about half the recordings contain drowsiness.
5. **Not the cause: the narrow age range.** At CHBP's ages the models reach
   r 0.70–0.80 on their own domain (Table 3.1). The range does cap the
   attainable R² (Section 9.3).

### 9.2 How to bring CHBP-like data into the training domain

**At test time, with no labels:**
- **Spectrum matching** (7.5) to the training data's median spectrum is the
  best single step: bias about 0, E2L 7.01, 64-channel cap 6.27.
- **A global or per-recording amplitude normalization** (7.1, 7.3) removes the
  bias almost as well.
- None of them can restore the ranking (r ≤ 0.49).

**Robust fix, on the training side** (Temple / Harvard data only; CHBP stays an
external test). Each step is supported by Section 8:
1. **Train on a band every device records**, for example 0.5 (or 1) to 45 Hz,
   instead of 0.1–75 Hz. That removes the models' dependence on the content
   CHBP lacks; Table 7.1 shows that dependence is what breaks the ranking.
2. **Make the input amplitude-robust:** per-recording amplitude or GSF
   normalization (7.3, 7.4) during training, or random gain augmentation (for
   example × 0.5–2). On the frozen models this costs only 0.3–0.9 years
   in-domain (Table 7.1).
3. **Track the forward tests of Table 7.1** (gain × 0.62, band 0.5–50 Hz) on
   the Temple / Harvard validation split while training. They are a label-free
   robustness benchmark that needs no external data.
4. **Report CHBP per cap**, and as a separate check on top of the cohort mean,
   the age-matched training-domain reference of Table 3.1.

### 9.3 Comparison with published CHBP results

The only published CHBP age results are within-CHBP 10-fold CV (Engemann et
al. 2022, Table 1, n = 282, eyes open and closed pooled, 0.1–49 Hz):
- handcrafted features + random forest: MAE 6.48 ± 0.60, R² 0.18 ± 0.17;
- ShallowNet 6.65; Deep4Net 6.89; filterbank-riemann 7.17;
- predict the mean: 7.33 ± 0.83.

These models are trained on CHBP. Ours never are:
- **As recorded:** best 8.66 (W7, every window) and 9.10 (first 5 minutes).
- **After label-free spectrum matching:** 7.01 (E2L), and 6.27 on the
  64-channel cap.

For reference only, since they use CHBP ages: on the same kind of 10 subject
folds, a two-parameter recalibration of the frozen predictions gives 6.44–6.59,
and a ridge regression on frozen E2 embeddings gives 5.93
(`docs/data/external_eval/chbp/cv10_*.csv`). Neither is pursued, because CHBP is
kept as an external test.

## 10. References

- Beheshti I et al. 2019. Bias-adjustment in neuroimaging-based brain age frameworks. *NeuroImage: Clinical* 24:102063. doi:10.1016/j.nicl.2019.102063
- Bomatter P et al. 2024. Machine learning of brain-specific biomarkers from EEG. *EBioMedicine* 106:105259. doi:10.1016/j.ebiom.2024.105259
- Bosch-Bayard J et al. 2020. EEG source-spectra norms… *Front Neurosci* 14:555119. doi:10.3389/fnins.2020.555119
- Bosch-Bayard J et al. 2026. Back to the future of qEEG. *Brain Topogr* 39(5):87. doi:10.1007/s10548-026-01240-4
- Butler ER et al. 2021. Pitfalls in brain age analyses. *Hum Brain Mapp* 42:4092–4101. doi:10.1002/hbm.25533
- de Lange AMG, Cole JH. 2020. Commentary: correction procedures in brain-age prediction. *NeuroImage: Clinical* 26:102229. doi:10.1016/j.nicl.2020.102229
- Donoghue T et al. 2020. Parameterizing neural power spectra into periodic and aperiodic components. *Nat Neurosci* 23:1655–1665. doi:10.1038/s41593-020-00744-x
- Engemann DA et al. 2022. A reusable benchmark of brain-age prediction from M/EEG resting-state signals. *NeuroImage* 262:119521. doi:10.1016/j.neuroimage.2022.119521
- Gemein LAW et al. 2024. Brain age revisited. *Imaging Neuroscience*. doi:10.1162/imag_a_00210
- Gemein LAW et al. 2026. Clinically altered brain activity may not look like aged brain activity. *Alzheimer's Dement TRCI* 12(3):e70302. doi:10.1002/trc2.70302
- Hajhassani D et al. 2026. Automated artifact removal in EEG age prediction. arXiv:2609.31195
- Hernández JL et al. 1994. A global scale factor in brain topography. *Int J Neurosci* 76:267–278. doi:10.3109/00207459408986009
- Karakis I et al. 2012. Frequency and determinants of sleep in routine EEG. *J Clin Neurophysiol* 29:333. doi:10.1097/WNP.0b013e31826244ae
- Li M et al. 2022. Harmonized-Multinational qEEG norms (HarMNqEEG). *NeuroImage* 256:119190. doi:10.1016/j.neuroimage.2022.119190; code github.com/LMNonlinear/HarMNqEEG
- Mellot A et al. 2023. Harmonizing and aligning M/EEG datasets with covariance-based techniques. *Imaging Neuroscience* 1. doi:10.1162/imag_a_00040
- Mellot A, Collas A et al. 2024. Geodesic optimization for predictive shift adaptation on EEG data (GOPSA). NeurIPS 2024, arXiv:2407.03878
- Merkin A et al. 2023. Do age-related differences in aperiodic neural activity explain differences in resting EEG alpha? *Neurobiol Aging* 121:78–87. doi:10.1016/j.neurobiolaging.2022.09.003
- Tröndle M et al. 2023. Decomposing age effects in EEG alpha power. *Cortex* 161:116–144. doi:10.1016/j.cortex.2023.02.002
- Tveitstøl T et al. 2026. Assessing the robustness of deep learning based brain age prediction models across multiple EEG datasets. *IEEE TBME* 73(7):2582–2601. doi:10.1109/TBME.2025.3639477
- Valdes-Sosa PA et al. 2021. The Cuban Human Brain Mapping Project. *Sci Data* 8:45. doi:10.1038/s41597-021-00829-7
- Voytek B et al. 2015. Age-related changes in 1/f neural electrophysiological noise. *J Neurosci* 35:13257–13265. doi:10.1523/JNEUROSCI.2332-14.2015

Items read only as abstracts by the literature search (not full text):
Hernández 1994, Merkin 2023, Tröndle 2023, Karakis 2012, Tveitstøl 2026,
Li 2022 (its method was read from the authors' code).
