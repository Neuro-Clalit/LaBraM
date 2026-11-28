"""Frozen-encoder feature cache (scripts/mil_cache.py --pretrained) and the ridge probe."""
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

DROP = ["A1", "A2", "T1", "T2"]


@pytest.fixture()
def setup(tmp_path):
    """The npy fixture, a run config for a tiny CBraMod and its 'released' weight file."""
    from labram.configs.run_configs import FinetuneRunConfig
    from labram.models.backbones import CBraModBackbone
    from test_window_qc import _corpus
    root = _corpus(tmp_path / "data")
    weights = tmp_path / "released.pth"
    torch.manual_seed(0)
    torch.save(CBraModBackbone(n_layer=2, dim_feedforward=64).net.state_dict(), weights)
    cfg = FinetuneRunConfig.load_config(
        "labram/configs/defaults/finetune_tuab_age.json",
        **{"data.data_format": "npy", "data.drop_channels": DROP, "data.trim_start_sec": 0.0,
           "data.trim_end_sec": 0.0, "data.eval_minutes": 0.0, "model.model": "cbramod_tiny_patch200",
           "finetune_checkpoint.finetune": str(weights), "mil.enabled": True})
    path = tmp_path / "run_config.yaml"
    cfg.save_to(str(path))
    return root, str(path), tmp_path


def _build(root, cfg, out, **kw):
    from labram.mil.cache import build_cache
    build_cache({}, str(root), str(out), splits=("train", "val", "test"), n_train_windows=4,
                batch_size=8, num_workers=0, descriptors=False, **kw)


def test_pretrained_encoder_is_cached_without_a_head(setup):
    from labram.mil.cache import Cache
    root, cfg, tmp = setup
    out = tmp / "cache"
    _build(root, cfg, out, pretrained={"enc": (cfg, []), "zs": (cfg, ["model.encoder.input_norm=zscore_window"])})
    info = Cache(str(out), "enc").info
    assert info["pretrained"] and info["model"] == "cbramod_tiny_patch200" and info["checkpoint"] is None
    assert not os.path.exists(out / "enc" / "head.pt")
    train, test = Cache(str(out), "enc").split("train"), Cache(str(out), "enc").split("test")
    assert train["h"].shape == (3 * 4, 200)                     # first 4 windows of 3 training recordings
    assert test["h"].shape == (20, 200) and np.isnan(test["pred"]).all()
    assert test["desc"].shape == (20, 0) and test["desc_columns"] == []
    assert set(test["meta"].columns) >= {"recording", "window", "minute", "age", "cohort", "dataset"}
    assert abs(float(test["meta"]["age"].iloc[0]) - 50.0) < 1e-3   # years, de-normalised
    assert not np.allclose(Cache(str(out), "zs").split("test")["h"], test["h"])   # the override took effect


def test_recordings_filter_and_channel_readout(setup):
    from labram.mil.cache import Cache
    root, cfg, tmp = setup
    out = tmp / "cache"
    _build(root, cfg, out, pretrained={"enc": (cfg, [])}, readout="channel_mean",
           recordings={"train": ["aaa_s001_t000", "ccc_s001_t000"]})
    train = Cache(str(out), "enc").split("train")
    assert train["h"].shape == (2 * 4, 19 * 200)                # one block per channel
    assert sorted(train["meta"]["recording"].unique()) == ["aaa_s001_t000", "ccc_s001_t000"]
    assert Cache(str(out), "enc").split("val")["h"].shape[0] == 20      # a split not named is cached whole
    assert Cache(str(out), "enc").info["readout"] == "channel_mean"


def test_recording_zscore_needs_the_models_own_preprocessing_off(setup):
    from labram.mil.cache import Cache, zscore_by_recording
    root, cfg, tmp = setup
    with pytest.raises(SystemExit, match="labram_plus and channel_drop off"):
        _build(root, cfg, tmp / "bad", pretrained={"enc": (cfg, [])}, recording_zscore=["enc"])
    _build(root, cfg, tmp / "ok", pretrained={"enc": (cfg, ["labram_plus.enabled=false"])},
           recording_zscore=["enc"])
    assert Cache(str(tmp / "ok"), "enc").info["recording_zscore"]
    x = torch.tensor([[[1.0, 3.0, 5.0], [0.0, 0.0, 0.0]]])            # one window, a flat second channel
    stats = {"r": (torch.tensor([3.0, 0.0]), torch.tensor([2.0, 0.0]))}
    z = zscore_by_recording(x, ["r"], stats, clip_sd=0.5) / 100
    assert torch.allclose(z[0, 0], torch.tensor([-0.5, 0.0, 0.5])) and (z[0, 1] == 0).all()


def test_probe_samples_a_fold_and_scores_every_set(setup):
    import fm_probe
    root, cfg, tmp = setup
    manifest = json.load(open(root / "manifest.json"))
    cv_split = {"folds": [{"fold": 0, "groups": ["ddd"]}, {"fold": 1, "groups": ["aaa", "bbb", "ccc"]}]}
    rec = fm_probe.sample_recordings(manifest, cv_split, fold=0, n_train=0, n_val=0)
    assert rec["probe"] == {"train": ["aaa_s001_t000", "bbb_s001_t000", "ccc_s001_t000"], "val": ["ddd_s001_t000"]}
    assert rec["cache"]["val"] == ["ddd_s001_t000"] and len(rec["cache"]["train"]) == 3
    assert len(fm_probe.sample_recordings(manifest, cv_split, 0, n_train=2, n_val=0)["probe"]["train"]) == 2
    out = tmp / "probe"
    _build(root, cfg, out / "main", pretrained={"A": (cfg, []), "B": (cfg, ["labram_plus.enabled=false"])},
           recordings=rec["cache"])
    _build(root, cfg, out / "ext", pretrained={"A": (cfg, []), "B": (cfg, ["labram_plus.enabled=false"])})
    feats = fm_probe.recording_features(str(out / "main" / "A"), "train", "mean_std")
    assert feats.shape == (3, 2 * 200 + 4) and list(feats["n_windows"]) == [4, 4, 4]
    table, paired = fm_probe.run_fit(str(out), ["A", "B"], rec, cohorts=["ext"], out=str(tmp / "res" / "probe"))
    assert set(table["group"]) >= {"test", "ext"} and set(table["features"]) == {"mean", "mean_std"}
    assert np.isfinite(table["mae"]).all() and (table["n_train"] == 3).all()
    assert set(paired["a"]) == {"B"} and set(paired["b"]) == {"A"}
    pred = json.load(open(tmp / "res" / "probe_pred_test.json"))
    assert list(pred["A/mean"]["splits"]["test"]["predictions"]) == ["eee_s001_t000"]
    m = fm_probe.metrics(np.array([1.0, 2.0, 4.0]), np.array([1.0, 3.0, 5.0]))
    assert m["mae"] == pytest.approx(2 / 3) and m["n"] == 3
