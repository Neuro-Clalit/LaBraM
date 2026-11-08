# Stimulus channels (photic / trigger / DC) in the TUH and HEEDB cohorts

Report, 2026-11-08. Questions:
1. Does a valid stimulus channel exist in the recordings of the TUEG
   and Harvard (HEEDB) cohorts, and how often?
2. Do we use it in training?

Script: `scripts/stim_channel_inventory.py`. Its outputs stay outside git
(`/data/runs/heedb/stim_channels/`), because the per-recording table lists
HEEDB sessions, which the HEEDB data use agreement keeps out of the repo.
Every HEEDB number below is an aggregate, and counts under 10 recordings are
shown as "<10" (`SMALL_CELL = 10`, as in `labram/eval/heedb_analysis.py`).

**Answer in short.**
- **TUH:** 65 % of the TUH recordings in the training cohort list a photic
  channel. About 55 % of those carry a real flash train, so about **36 %** of
  TUH recordings have a usable photic marker.
- **HEEDB:** no channel is labelled photic. The DC and TRIG inputs that do
  exist carry no flash train in a 160-session sample.
- **Training:** neither corpus feeds any stimulus channel to the model. The
  training npy files hold only the 23 EEG channels (section 3).

## 1. Header inventory: is a stimulus channel listed?

Every recording is checked from its EDF header, without reading any signal.
- **TUH (TUEG):** the channel labels come from the NEDC header dump
  (`tueg/v2.0.2/DOCS/headers.tar.gz`). A photic channel is labelled
  `PHOTIC-REF` (22,930 files) or `PHOTIC PH` (12,252 files). No TUEG file has
  a TRIG, STIM or EVENT label.
- **HEEDB:** the labels come from the HEEDB index (`heedb_index.parquet`,
  `labels`). It has no PHOTIC, STIM, STI or EVENT label at all. The
  candidates are the amplifier's DC inputs `DC1`–`DC16`, where a photic
  stimulator is often wired on Natus/XLTEK systems, and `TRIG`.

### Table 1.1. Recordings whose header lists a stimulus channel

The share of recordings per cohort and split, with $n$ the recordings in that
row. In the cohort rows, "TUH" and "HEEDB" are the two corpora inside the
joint cohort.

| cohort / part | split | $n$ | photic label | DC1–16 inputs | TRIG |
|---|---|---|---|---|---|
| TUEG, whole corpus | all | 70,830 | 49.7 % | — | 0 % |
| `all_both_v3` (TUH) | all | 29,309 | 69.9 % | — | 0 % |
| | train / val / test | 25,613 / 3,422 / 274 | 69.4 / 71.4 / 97.4 % | — | 0 % |
| `all_both_v3_anymiss` = TUH part of `tuh_heedb_v1` | all | 31,312 | 65.4 % | — | 0 % |
| | train / val / test | 27,616 / 3,422 / 274 | 64.4 / 71.4 / 97.4 % | — | 0 % |
| HEEDB, whole index | all | 341,084 | 0 % | 7.0 % | 0.8 % |
| `heedb_routine_v1` = HEEDB part of `tuh_heedb_v1` | all | 71,805 | 0 % | 33.2 % | 3.9 % |
| | train / val / test | 57,275 / 7,192 / 7,338 | 0 % | 33.3 / 33.7 / 32.3 % | 3.9 / 3.8 / 3.7 % |

### Table 1.2. HEEDB routine sessions with DC inputs, by site

The share of `heedb_routine_v1` sessions whose header lists at least one of
`DC1`–`DC16`.

| site | I0002 | I0003 | S0001 | S0002 |
|---|---|---|---|---|
| DC inputs | 95.7 % | 98.5 % | 26.9 % | 8.9 % |

Table 1.1 and Table 1.2 show the two corpora differ completely. TUH names a
photic channel in about two thirds of the cohort recordings, and in almost
every test recording. Whether HEEDB has DC inputs depends on the site, and
the label says nothing about what is connected to them.

## 2. Signal check: does the channel carry a flash train?

A random sample of EDFs is downloaded and the candidate channels are read,
in volts (MNE). Each channel $x$ of $T$ samples at $f_s$ Hz is normalized
between its 0.1st and 99.9th percentiles, $q_{0.1}$ and $q_{99.9}$, which
makes the test robust to a few glitch samples:

$$
v_t=\operatorname{clip}\Bigl(\frac{x_t-q_{0.1}}{q_{99.9}-q_{0.1}},\,0,\,1\Bigr),\qquad
L=\frac1T\sum_t \mathbf 1\{v_t<0.1\ \lor\ v_t>0.9\},
\tag{2.1}
$$

$$
r_t=\begin{cases}\mathbf 1\{v_t>0.5\}, & \frac1T\sum_t \mathbf 1\{v_t>0.5\}\le 0.5\\ \mathbf 1\{v_t\le 0.5\}, & \text{otherwise}\end{cases},\qquad
P=\#\{t: r_t=0,\ r_{t+1}=1\},
\tag{2.2}
$$

$$
S=\#\Bigl\{s: \text{second } s \text{ contains a pulse onset}\Bigr\},\qquad
F=\frac{S}{\lfloor T/f_s\rfloor},\qquad
R=\operatorname*{median}_{\text{active } s}\ \#\text{onsets in } s .
\tag{2.3}
$$

The symbols are:
- $L$: the share of samples at either extreme ("two-level" if $L\ge 0.98$).
- $r_t$: the rarer of the two levels, so either pulse polarity counts.
- $P$: the number of pulse onsets.
- $S$: the seconds that contain a pulse.
- $F$: the share of the recording's seconds that contain a pulse.
- $R$: the flash rate, in Hz.

Each channel is classified (`classify`) as:
- **flat**: $x$ is constant;
- **pulse train**: $L\ge 0.98$, $P\ge 20$ and $F<0.5$, i.e. a stimulator or trigger;
- **no stimulation**: $P<20$, with $L\ge0.98$ or the rarer level under 1 % of samples (a level with a few glitches);
- **analog**: anything else, i.e. a continuous signal, drift, or an unconnected input's noise.

A recording counts as "pulse train" if any of its candidate channels is one.

### Table 2.1. TUH: photic channel signal, 120 recordings

A random sample of 120 TUH recordings of `tuh_heedb_v1` whose header lists a
photic channel. The shares follow Eqs. (2.1)–(2.3); the pulse statistics are
medians over the pulse-train recordings.

| class | share | what it is |
|---|---|---|
| pulse train | 46.7 % | median $P$ = 1,210 flashes over $S$ = 120 s, $R$ = 9 Hz (range 9–11 Hz) |
| analog, but a flash train | 8.3 % | the same protocol (≈1,220 flashes in ≈126 s, 9 Hz) with non-square pulses, so $L<0.98$ |
| flat | 37.5 % | the channel exists, but no stimulation was recorded |
| no stimulation / glitches | 7.5 % | at most 11 isolated spikes |
| **usable flash marker** | **55.0 %** | 66 of 120; 95 % CI about 46–64 % |

The TUH protocol is strikingly uniform: about 2 minutes of stepped-frequency
flashes averaging 9–10 Hz. With 65.4 % of the cohort's TUH recordings
listing the channel (Table 1.1), about **36 %** (≈30–42 %) of TUH recordings
in `tuh_heedb_v1` have a usable photic marker.

### Table 2.2. HEEDB: DC and TRIG inputs, 160 sessions

A random sample of `heedb_routine_v1` sessions whose header lists DC or TRIG
inputs, 40 per site. 1,492 channels were read. "Peak-to-peak" is the median
range of a DC channel, in µV.

| site | sessions | DC: pulse train | DC: analog | DC: no stimulation | DC peak-to-peak (median) |
|---|---|---|---|---|---|
| I0002 | 40 | 0 | 100 % | 0 % | 4.2 mV |
| I0003 | 40 | 0 | 95 % | <10 sessions | 5.8 mV |
| S0001 | 40 | <10 sessions | 95 % | <10 sessions | 8.0 mV |
| S0002 | 40 | 0 | 100 % | 0 % | 6.7 mV |
| TRIG, all sites pooled | 13 | 0 | 0 % | 100 % (constant level) | — |

On HEEDB, Table 2.2 shows **no flash train on any DC or TRIG input** in the
sample:
- **DC inputs** carry a few-millivolt continuous signal, consistent with an
  unconnected amplifier input or another device. At S0001 a single input
  carried a slow 1 Hz train for about 3 minutes, which does not look like a
  photic protocol.
- **TRIG channels** are constant.

If HEEDB records photic stimulation at all, it is in the annotations: every
file has an EDF annotation track, and 14 % of routine sessions also have a
separate annotation file. That is not checked here.

## 3. Do we use these channels in training?

No:
- **npy files.** The npy builder (`dataset_maker/make_TUH_npy.py`, used for
  TUH and HEEDB alike) writes only the 23 EEG channels of
  `labram/data/tuh_metadata.py::TUH_CHANNELS` (FP1 … PZ, A1/A2, T1/T2). This
  is the `channels` list of every cohort manifest (`all_both_v3`,
  `all_both_v3_anymiss`, `heedb_routine_v1`, `tuh_heedb_v1`). E2/E2L drop
  A1/A2/T1/T2 on top of that (`data.drop_channels`), so they train on 19.
- **Features.** No session, condition or multitask feature reads a photic,
  trigger or DC channel (`dataset_maker/make_session_features.py`,
  `make_condition_features.py`, `make_multitask_targets.py`).
- **Analytics only.** The photic channel is read only by
  `labram/data/tuh_meta_features.py` (`photic_performed`, `photic_stim_sec`)
  for the meta-feature notebook.

The EEG's **response** to the flashes (photic driving) is still in the EEG
channels. When a flash train falls inside a recording's trimmed range, the
model trains on it, with no marker telling it so.

## 4. Fix made along the way

`tuh_meta_features.channel_inventory` recognised only `PHOTIC-REF` and missed
the 12,252 TUEG files that label the channel `PHOTIC PH`. `has_photic_channel`
in the existing `meta_features` tables therefore undercounts: 41 % instead of
70 % for `all_both_v3`. The function is fixed; rebuild those tables with
`dataset_maker/make_cohort_meta.py` / `make_TUAB_meta.py` to pick it up.

## 5. Possible next steps

* **Mark the photic windows (TUH).** Locate the flash train's time span in
  each TUH recording and flag the windows inside it. Then check whether the
  model errs more on them, the same question as the HEEDB first-minutes effect
  in `window_confidence_results.md` (Table 3.3). They could also serve as a
  window descriptor for A3 / B1.
* **HEEDB annotations.** Search the EDF annotation track and annotation files
  for "photic" / "PS" / "flash" events to get the stimulation periods without
  a channel.
* **Do not** add the stimulus channel as a model input. It would exist in
  about a third of TUH recordings and in no HEEDB recording, so the model
  could learn the corpus from it.

Reproduce (header inventory: about 1 min; signal sample: about 5 min with 8
workers, credentials for both buckets needed):

```bash
AWS_PROFILE=mfa PYTHONPATH=. python scripts/stim_channel_inventory.py \
  --out-dir /data/runs/heedb/stim_channels --n-tuh 120 --n-heedb 160 --workers 8
```
