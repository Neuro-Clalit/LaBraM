# Age regression (EEG brain age) on TUH data

Predicting a patient's age from their EEG — "brain age" — as a downstream
regression task, using the same pre-trained LaBraM encoder as the abnormal/normal
classification task. Off by default; opt in with `data.dataset=TUAB_AGE`.

## Where the age comes from

TUH **no longer distributes the clinical reports** ("We no longer distribute
reports with our corpora"), so the EDF header is the only source of patient
demographics. It is a complete one: the standard EDF *local patient
identification* field holds the age directly.

```
bytes   8:88   local patient identification
               'aaaaantl F 01-JAN-0000 aaaaantl Age:42'
                subject   sex dob      subject   age
bytes  88:168  local recording identification
               'Startdate 01-JAN-2012 aaaaantl_s001 XXX X'
                          session year
```

**MNE cannot give you this.** `read_raw_edf(...).info['subject_info']` returns
only `{'his_id', 'sex', 'last_name'}`: the date of birth is anonymised to
`01-JAN-0000`, so there is nothing to subtract a birth year from. The literal
`Age:` token has to be read from the raw header bytes, which is what
`labram/data/tuh_metadata.py` does — reading only the first 256 bytes per file, so
a full-corpus sweep takes seconds and needs neither MNE nor pandas.

### Validation

On TUAB v3.0.0 the parser finds an `Age:` field in **all 2,993 recordings (100%,
0 missing)**, and the parsed values reproduce the corpus `AAREADME.txt`
DEMOGRAPHICS tables *exactly* — every age decade bucket per split × label, and
every gender count (F/M × normal/abnormal × train/eval). That agreement is the
evidence the byte offsets and token positions above are right.

Usable ages 1–89: **2,978 recordings**, mean 49.1, median 49, std 17.4.

### Sentinel values

| Value | Count (TUAB) | Meaning | Handling |
|---|---|---|---|
| `Age:999` | 12 | TUH's redaction for patients aged 90+ (HIPAA requires ages over 89 to be aggregated). Confirmed: the per-split/per-label counts match the README's "90-100" row exactly (1/1/6/4). | Excluded |
| `Age:0` | 3 | Ambiguous — a genuine neonate or a missing value. | Excluded by the default 1–89 range |

Sentinels are dropped, never trained on. `RecordingMetadata.age` is `None` for
them while `raw_age` keeps the as-parsed value for auditing, so a sentinel can
never silently become a target. Widen the range with `--max_age` if you want the
90+ group in (their true ages are unrecoverable, so this is rarely a good idea).

## Joining age onto the existing windows

The window pickles `dataset_maker/make_TUAB.py` already produced are named
`<subject>_s<NNN>_t<NNN>_<windowIdx>.pkl`, and 100% of processed recording stems
map back to an EDF stem. So the age joins onto the **existing** windows by
filename via a sidecar — **no re-preprocessing**, which would otherwise be a
multi-hour MNE pass over ~409k windows. Only ~0.6% of windows are lost to
sentinels.

```
<corpus>/edf/processed/
├── age_metadata.json     # stem -> {age, sex, subject, session, token, year}
├── age_split.json        # subject-disjoint train/val/test window lists
├── train/  val/  test/   # the pickles make_TUAB.py wrote, untouched
```

`TUABAgeLoader` resolves `age_metadata.json` by searching its data root and
parents, so it can recover its labels from the root alone. That matters because
`cross_validation._build_split_dataset` rebuilds loaders positionally as
`type(src)(root, files, sampling_rate)` — there is no opportunity to pass a lookup
through.

Windows whose recording has no usable age are **filtered at construction time**,
not at access time: `TUHLoader.__getitem__` catches `KeyError` and substitutes a
different window, so a missing age would otherwise corrupt the targets invisibly.

## Splits: by subject, aggregated by recording

The shipped `processed/` split has **16 subjects in both train and val**.
`make_TUAB.py` shuffles and splits subjects independently within `normal/` and
`abnormal/`, and 54 TUAB train subjects appear as both — so a subject can land in
train via its abnormal recording and val via its normal one. It is also unseeded,
so the split is not reproducible. For age this leaks badly, because a subject's
age is near-constant (measured within-subject spread: mean 1.4 years; 194 of 443
multi-file subjects have spread 0).

`labram/data/age_splits.py` rebuilds train/val from the pooled windows **by
subject**, with a fixed seed, and asserts zero overlap on both save and load —
a leaking split raises rather than passing silently. TUAB's official eval set
(`processed/test`) is left untouched.

| split | subjects | recordings | windows | age mean ± std |
|---|---|---|---|---|
| train | 1,650 | 2,145 | 294,253 | 48.9 ± 17.8 |
| val   |   412 |   556 |  75,717 | 48.1 ± 15.8 |
| test  |   251 |   274 |  36,728 | 49.9 ± 17.8 |

A pooled split stores files as `<subdir>/<name>.pkl` so each split stays a
*single* loader over one root rather than a `ConcatDataset` (which
`enable_window_ids` does not recurse into). Group-id helpers therefore derive ids
from the basename.

**Split by subject, aggregate by recording.** The corpus README warns that a
subject "might be represented more than once (with different ages)" — age is a
property of the *session*, not the subject. So `cross_validation.split_by='subject'`
prevents leakage while `evaluation.agg_case_by='recording'` keeps the prediction
unit correct. These are independent knobs; do not collapse them to the same key.

## The regression task

`nb_classes=1` already builds `nn.Linear(embed_dim, 1)` — a scalar head — and
`finetune_setup.load_finetune_checkpoint` already drops a shape-mismatched
`head.weight`/`head.bias`. **No new model code is needed.**

But `nb_classes == 1` also means "binary classification" across ~20 call sites, so
it cannot distinguish the two. `FinetuneModelConfig.task`
(`"classification"` | `"regression"`, set from `DatasetBundle.task`) is what
selects:

| | classification | regression |
|---|---|---|
| criterion | BCE / cross-entropy | Huber (default), MSE or L1 — `loss.regression_loss` |
| output transform | `sigmoid` / `softmax` | none (a scalar, not a probability) |
| metrics | accuracy, ROC-AUC, PR-AUC, … | MAE, RMSE, R², Pearson/Spearman r |
| model selection | `accuracy`, higher is better | `mae`, **lower** is better |
| figures | confusion matrix, ROC/PR | predicted-vs-true scatter |
| window pooling | mean/median/max/vote/entropy | mean/median/max only |

Without that flag, `evaluate` would rebuild `BCEWithLogitsLoss` and score ages
with cross-entropy. `build_downstream_criterion` is the single dispatch point used
by both the train and eval paths.

**Target normalization.** The head is initialised with `init_scale=0.001`, so it
starts near zero and a raw target of ~49 would produce an enormous initial loss.
The loader z-scores the target with the **train split's** mean/std (carried on the
bundle as `target_stats`); `evaluate` de-normalizes once before computing metrics,
so every reported error is in **years**.

### Brain-age diagnostics

Age decoders regress toward the cohort mean: the old are predicted too young and
the young too old. Two extra metrics make that visible.

- `age_bias_slope` — least-squares slope of the residual `(pred - true)` against
  the true age. Near 0 is good; strongly negative means regression to the mean
  dominates.
- `mae_corrected` — MAE after removing that linear bias, i.e. the honest error
  once the effect is accounted for. Report this alongside raw MAE when the slope
  is far from 0.

A model that just predicts the training mean scores MAE ≈ 14 on TUAB. Published
EEG brain-age benchmarks land around **MAE 7–8 years**, so that is the range to
aim for; anything near 14 means the target plumbing is broken.

## Usage

```bash
TUAB=/path/to/TUH_Abnormal/v3.0.0/edf
```

Extract the demographics (prints a summary to cross-check against the corpus
AAREADME):

```bash
python dataset_maker/make_TUAB_age.py scan --root "$TUAB"
```

Build the subject-disjoint split:

```bash
python dataset_maker/make_TUAB_age.py split --root "$TUAB"
```

Fine-tune:

```bash
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 -m labram.runs.run_finetune \
  --config labram/configs/defaults/finetune_tuab_age.json \
  --set data.data_path="$TUAB" \
        finetune_checkpoint.finetune=./checkpoints/labram-base.pth
```

Cross-validated (group-disjoint by subject; see `docs/cross_validation.md`):

```bash
python -m labram.runs.finetune_cv \
  --config labram/configs/defaults/finetune_tuab_age.json \
  --set data.data_path="$TUAB" cross_validation.enabled=true
```

## Scaling to TUEG

The parser is corpus-agnostic — nothing in it is TUAB-specific. TUEG v2.0.2
(26,846 sessions, ~15,000 patients, ages 2 days–106 years) uses the same header
format, so the same `scan` works unchanged and yields roughly 10× more age labels
over a far wider range:

```bash
python dataset_maker/make_TUAB_age.py scan --root /path/to/tuh_eeg/v2.0.2/edf
```

Check the printed coverage and sentinel counts before trusting a new corpus —
TUEG's older recordings are less uniformly populated than TUAB's. You will also
need window pickles for it (`dataset_maker/make_h5dataset_for_pretrain.py` or a
TUAB-style preprocessing pass), since the join is by recording stem.

## Files

| Path | Role |
|---|---|
| `labram/data/tuh_metadata.py` | EDF-header parser, sidecar I/O, age lookup |
| `labram/data/age_splits.py` | subject-disjoint split + leakage assertions |
| `labram/data/tuh_datasets.py` | `TUABAgeLoader`, `prepare_TUAB_age_dataset` |
| `labram/data/bundles.py` | `TUAB_AGE` bundle, `task` / `target_stats` |
| `labram/losses/regression.py` | criterion selection + `build_downstream_criterion` |
| `labram/utils/regression_metrics.py` | MAE/RMSE/R²/r + brain-age diagnostics |
| `dataset_maker/make_TUAB_age.py` | `scan` / `split` CLI |
| `labram/configs/defaults/finetune_tuab_age.json` | ready-to-run config |
