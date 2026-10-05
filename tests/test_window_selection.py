"""Recording/window selection for TUAB fine-tuning: case filter, start/end
trimming, multi-window samples, the per-recording eval budget, plus the time
embedding resize that longer inputs need."""
import json
import pickle

import numpy as np
import pytest
import torch

from labram.data.bundles import REGRESSION, DatasetBundle
from labram.data.tuh_datasets import TUABAgeLoader, TUABLoader, TUEVLoader
from labram.data.tuh_metadata import (
    RecordingMetadata, label_from_path, load_label_lookup_for, save_metadata_sidecar,
)
from labram.data.window_selection import (
    WindowSelection, apply_window_selection, scan_recordings, select_files, window_index,
)
from labram.runs.finetune_setup import (
    load_finetune_checkpoint, required_time_patches, resize_time_embed,
)

N_CHANNELS = 23


def _write_recording(directory, stem, n_windows, missing=()):
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(n_windows):
        if i in missing:
            continue
        # Each window's samples encode (window index, position) so concatenation
        # order is checkable.
        X = np.full((N_CHANNELS, 2000), float(i))
        with open(directory / f"{stem}_{i}.pkl", "wb") as fh:
            pickle.dump({"X": X, "y": 0}, fh)
    return [f"{stem}_{i}.pkl" for i in range(n_windows) if i not in missing]


def _select(tmp_path, files, selection, **kw):
    n_windows, available = scan_recordings(str(tmp_path), files)
    return select_files(files, selection, n_windows=n_windows, available=available, **kw)


class TestSelectionConfig:
    @pytest.mark.parametrize("kwargs", [
        dict(case_filter="sick"), dict(window_sec=15), dict(window_sec=70),
        dict(window_sec=0), dict(trim_start_sec=-1), dict(window_sec=60, eval_minutes=0.5),
    ])
    def test_invalid_settings_raise(self, kwargs):
        with pytest.raises(ValueError):
            WindowSelection(**kwargs)

    def test_default_selection_is_a_no_op(self):
        assert WindowSelection().is_default
        bundle = object()
        assert apply_window_selection(bundle, WindowSelection()) is bundle

    def test_trim_rounds_up_to_whole_windows(self):
        sel = WindowSelection(trim_start_sec=55, trim_end_sec=60, eval_minutes=5)
        assert (sel.trim_start_windows, sel.trim_end_windows, sel.eval_windows) == (6, 6, 30)

    def test_from_data_config(self):
        from labram.configs.data_config import DataConfig
        sel = WindowSelection.from_data_config(DataConfig(window_sec=30, eval_minutes=5))
        assert sel.windows_per_sample == 3 and sel.eval_windows == 30


class TestSelectFiles:
    def test_trims_the_first_and_last_minute(self, tmp_path):
        files = _write_recording(tmp_path, "aaa_s001_t000", 20)
        kept = _select(tmp_path, files, WindowSelection(trim_start_sec=60, trim_end_sec=60))
        assert [window_index(f) for f in kept] == list(range(6, 14))

    def test_multi_window_samples_start_on_a_grid(self, tmp_path):
        files = _write_recording(tmp_path, "aaa_s001_t000", 20)
        kept = _select(tmp_path, files, WindowSelection(trim_start_sec=60, window_sec=30))
        # windows 6..19 -> samples [6,7,8] [9,10,11] [12,13,14] [15,16,17]; 18-19 too short.
        assert [window_index(f) for f in kept] == [6, 9, 12, 15]

    def test_eval_budget_limits_only_eval_splits(self, tmp_path):
        files = _write_recording(tmp_path, "aaa_s001_t000", 60)
        sel = WindowSelection(trim_start_sec=60, eval_minutes=2)
        assert len(_select(tmp_path, files, sel, is_eval=True)) == 12
        assert len(_select(tmp_path, files, sel, is_eval=False)) == 54

    def test_case_filter_uses_the_labels(self, tmp_path):
        files = (_write_recording(tmp_path, "aaa_s001_t000", 3)
                 + _write_recording(tmp_path, "bbb_s001_t000", 3))
        labels = {"aaa_s001_t000": "normal", "bbb_s001_t000": "abnormal"}
        kept = _select(tmp_path, files, WindowSelection(case_filter="normal"), labels=labels)
        assert kept == [f for f in files if f.startswith("aaa")]

    def test_samples_with_a_missing_window_are_dropped(self, tmp_path):
        files = _write_recording(tmp_path, "aaa_s001_t000", 9, missing={4})
        kept = _select(tmp_path, files, WindowSelection(window_sec=30))
        assert [window_index(f) for f in kept] == [0, 6]

    def test_selection_is_idempotent(self, tmp_path):
        files = _write_recording(tmp_path / "train", "aaa_s001_t000", 40)
        files = [f"train/{f}" for f in files]
        sel = WindowSelection(trim_start_sec=60, trim_end_sec=60, window_sec=20, eval_minutes=3)
        once = _select(tmp_path, files, sel, is_eval=True)
        assert _select(tmp_path, once, sel, is_eval=True) == once
        assert once and all(f.startswith("train/") for f in once)


def _labelled_corpus(tmp_path):
    processed = tmp_path / "processed"
    meta, files = {}, {}
    plan = {"train": [("aaa", 30, "normal"), ("bbb", 70, "abnormal")],
            "val": [("ccc", 40, "normal")],
            "test": [("ddd", 50, "abnormal"), ("eee", 60, "normal")]}
    for split, recs in plan.items():
        files[split] = []
        for subject, age, label in recs:
            stem = f"{subject}_s001_t000"
            meta[stem] = RecordingMetadata(stem=stem, subject=subject, session="s001",
                                           token="t000", age=age, sex="F", year=2012,
                                           raw_age=age, label=label)
            files[split] += [f"{split}/{f}" for f in
                             _write_recording(processed / split, stem, 24)]
    save_metadata_sidecar(meta, str(processed / "age_metadata.json"))
    ages = {s: float(m.age) for s, m in meta.items()}
    loaders = {split: TUABAgeLoader(str(processed), fs, age_lookup=ages, target_stats=(0.0, 1.0))
               for split, fs in files.items()}
    return DatasetBundle(train=loaders["train"], val=loaders["val"], test=loaders["test"],
                         ch_names=[], nb_classes=1, metrics=["mae"], task=REGRESSION,
                         target_stats=(0.0, 1.0))


class TestApplyToBundle:
    def test_filters_trims_and_limits_every_split(self, tmp_path):
        bundle = apply_window_selection(_labelled_corpus(tmp_path), WindowSelection(
            case_filter="normal", trim_start_sec=60, trim_end_sec=60, window_sec=20,
            eval_minutes=1))
        assert {f.split("/")[1].split("_")[0] for f in bundle.train.files} == {"aaa"}
        assert len(bundle.train.files) == 6          # windows 6..17 in 20 s samples
        assert len(bundle.val.files) == 3             # 1 minute = 3 samples of 20 s
        assert bundle.train.windows_per_item == bundle.val.windows_per_item == 2

    def test_empty_split_raises(self, tmp_path):
        # The only val recording is normal.
        with pytest.raises(ValueError, match="val split empty"):
            apply_window_selection(_labelled_corpus(tmp_path),
                                   WindowSelection(case_filter="abnormal"))

    def test_target_stats_follow_the_selected_train_set(self, tmp_path):
        bundle = apply_window_selection(_labelled_corpus(tmp_path),
                                        WindowSelection(trim_start_sec=60))
        mean, std = bundle.target_stats
        assert mean == pytest.approx(50.0)            # equal windows of ages 30 and 70
        assert bundle.val.target_stats == bundle.target_stats

    def test_concat_datasets_are_rebuilt_with_the_new_lengths(self, tmp_path):
        bundle = _labelled_corpus(tmp_path)
        bundle.train = torch.utils.data.ConcatDataset([bundle.train, bundle.val])
        bundle = apply_window_selection(bundle, WindowSelection(trim_start_sec=60))
        assert len(bundle.train) == sum(len(d.files) for d in bundle.train.datasets)

    def test_case_filter_is_rejected_for_classification(self, tmp_path):
        bundle = _labelled_corpus(tmp_path)
        bundle.task = "classification"
        with pytest.raises(ValueError, match="single class"):
            apply_window_selection(bundle, WindowSelection(case_filter="normal"))

    def test_non_tuab_loaders_are_rejected(self, tmp_path):
        bundle = _labelled_corpus(tmp_path)
        bundle.train = TUEVLoader(str(tmp_path), [])
        with pytest.raises(TypeError):
            apply_window_selection(bundle, WindowSelection(trim_start_sec=60))


def test_loader_concatenates_consecutive_windows(tmp_path):
    files = _write_recording(tmp_path, "aaa_s001_t000", 6)
    loader = TUABLoader(str(tmp_path), [files[2]])
    loader.windows_per_item = 3
    X, _ = loader[0]
    assert X.shape == (N_CHANNELS, 6000)
    assert [X[0, i * 2000].item() for i in range(3)] == [2.0, 3.0, 4.0]


class TestLabels:
    def test_label_from_path(self):
        assert label_from_path("/c/edf/train/abnormal/01_tcp_ar/x.edf") == "abnormal"
        assert label_from_path("/c/edf/eval/normal/01_tcp_ar/x.edf") == "normal"
        assert label_from_path("/c/edf/x.edf") is None

    def test_a_pre_label_sidecar_asks_for_a_rescan(self, tmp_path):
        with open(tmp_path / "age_metadata.json", "w") as fh:
            json.dump({"recordings": {"a_s001_t000": dict(
                stem="a_s001_t000", subject="a", session="s001", token="t000",
                age=40, sex="F", year=2012, raw_age=40)}}, fh)
        with pytest.raises(ValueError, match="scan"):
            load_label_lookup_for(str(tmp_path))


class TestTimeEmbedding:
    def test_required_patches_never_drop_below_the_pretrained_16(self):
        from labram.configs.data_config import DataConfig
        assert required_time_patches(DataConfig(window_sec=10)) == 16
        assert required_time_patches(DataConfig(window_sec=60)) == 60

    def test_resize_keeps_the_endpoints(self):
        te = torch.randn(1, 16, 8)
        out = resize_time_embed(te, 60)
        assert out.shape == (1, 60, 8)
        assert torch.allclose(out[:, 0], te[:, 0]) and torch.allclose(out[:, -1], te[:, -1])
        assert resize_time_embed(te, 16) is te

    def test_pretrained_checkpoint_loads_into_a_longer_model(self, tmp_path):
        from timm.models import create_model
        import labram.models.registry  # noqa: F401
        from types import SimpleNamespace

        short = create_model("labram_base_patch200_200", num_classes=1, init_values=0.1)
        long = create_model("labram_base_patch200_200", num_classes=1, init_values=0.1,
                            max_time_patches=60)
        path = tmp_path / "ckpt.pth"
        torch.save({"model": short.state_dict()}, path)
        load_finetune_checkpoint(long, SimpleNamespace(
            finetune=str(path), model_key="model", model_filter_name="", model_prefix=""))
        assert long.time_embed.shape[1] == 60
        assert torch.allclose(long.time_embed[:, 0], short.time_embed[:, 0])
        out = long(torch.randn(2, 4, 60, 200), channel_indices=None)
        assert out.shape == (2, 1)


def test_a_run_config_saved_before_these_fields_still_loads(tmp_path):
    from labram.eval.loading import load_run_config
    with open("labram/configs/defaults/finetune_tuab_age.json") as fh:
        cfg = json.load(fh)
    for key in ("case_filter", "trim_start_sec", "trim_end_sec", "window_sec", "eval_minutes"):
        del cfg["data"][key]
    path = tmp_path / "run_config.json"
    path.write_text(json.dumps(cfg))
    with pytest.warns(UserWarning):
        loaded = load_run_config(str(path))
    assert loaded.data.window_sec == 10 and loaded.data.eval_minutes == 0.0


def test_whole_numbers_are_accepted_for_seconds_and_minutes():
    from labram.configs.run_configs import FinetuneRunConfig
    cfg = FinetuneRunConfig.load_config(
        "labram/configs/defaults/finetune_tuab_age.json",
        **{"data.eval_minutes": 5, "data.trim_start_sec": 30})
    assert WindowSelection.from_data_config(cfg.data).eval_windows == 30


def test_age_default_config_is_scenario_d():
    """finetune_tuab_age.json is the best ablation (scenario D): 15 epochs at
    lr 1e-4 with 2 warmup epochs, CAR-only input preprocessing, 60 s trims and
    a 5-minute evaluation window. See docs/age_training_scenario_D.md."""
    from labram.configs.run_configs import FinetuneRunConfig
    cfg = FinetuneRunConfig.load_config("labram/configs/defaults/finetune_tuab_age.json")
    assert (cfg.trainer.epochs, cfg.optimizer.lr, cfg.optimizer.warmup_epochs) == (15, 1e-4, 2)
    assert cfg.labram_plus.enabled and cfg.labram_plus.common_average_reference
    assert not cfg.labram_plus.z_score_patches
    assert (cfg.data.trim_start_sec, cfg.data.trim_end_sec, cfg.data.eval_minutes) == (60, 60, 5)
    assert cfg.loss.regression_loss == "huber" and cfg.model.trainable_prefixes == []
