"""data.drop_channels: the npy loader serves only the kept channels, the bundle's
channel names shrink to match, and a fine-tune runs end to end on 21 channels."""
import numpy as np
import pytest
import torch

from labram.data.bundles import REGRESSION, DatasetBundle
from labram.data.channel_selection import drop_channels
from labram.data.eeg_constants import TUH_EEG_CH_NAMES, normalize_ch_names
from labram.data.tuh_datasets import prepare_TUAB_age_dataset
from test_window_qc import _corpus

CH = normalize_ch_names(TUH_EEG_CH_NAMES)


def _bundle(root):
    train, test, val, stats = prepare_TUAB_age_dataset(str(root), data_format="npy")
    return DatasetBundle(train=train, val=val, test=test, ch_names=list(CH), nb_classes=1,
                         metrics=["mae"], task=REGRESSION, target_stats=stats)


def test_dropped_channels_are_left_out_of_every_split(tmp_path):
    root = _corpus(tmp_path)
    full = _bundle(root)
    b = drop_channels(_bundle(root), ["t1", "T2"])
    assert b.ch_names == CH[:21] and "T1" not in b.ch_names
    for split in ("train", "val", "test"):
        x_full, y_full = getattr(full, split)[0]
        x, y = getattr(b, split)[0]
        assert x.shape == (21, 2000) and torch.equal(x, x_full[:21]) and torch.equal(y, y_full)


def test_middle_channels_keep_the_order_of_the_rest(tmp_path):
    root = _corpus(tmp_path)
    x_full = _bundle(root).train[0][0]
    b = drop_channels(_bundle(root), ["A1", "A2"])
    keep = [i for i, c in enumerate(CH) if c not in ("A1", "A2")]
    assert b.ch_names == [CH[i] for i in keep]
    assert torch.equal(b.train[0][0], x_full[keep])


def test_unknown_channel_and_no_op(tmp_path):
    root = _corpus(tmp_path)
    with pytest.raises(ValueError, match="XX"):
        drop_channels(_bundle(root), ["XX"])
    b = _bundle(root)
    assert drop_channels(b, []) is b and b.ch_names == CH


def test_skipped_signal_has_the_kept_width(tmp_path):
    root = _corpus(tmp_path)
    b = drop_channels(_bundle(root), ["T1", "T2"])
    b.train.skip_signal = True
    assert b.train[0][0].shape == (21, 2000)


def test_run_finetune_on_21_channels(tmp_path):
    from labram.configs.run_configs import FinetuneRunConfig
    from labram.runs import run_finetune
    root = _corpus(tmp_path / "data")
    out = tmp_path / "out"
    cfg = FinetuneRunConfig.load_config(
        "labram/configs/defaults/finetune_tuab_age.json",
        **{"data.data_path": str(root), "data.data_format": "npy",
           "data.trim_start_sec": 0.0, "data.trim_end_sec": 0.0, "data.eval_minutes": 0.0,
           "data.num_workers": 0, "data.drop_channels": ["T1", "T2"],
           "trainer.debug": True, "trainer.epochs": 1,
           "distributed.device": "cpu", "finetune_checkpoint.finetune": "",
           "clearml.enabled": False, "output.output_dir": str(out), "output.log_dir": str(out),
           "logging.log_model_graph": False})
    summary = run_finetune.main(cfg)
    assert "mae" in summary["best_val_stats"]
