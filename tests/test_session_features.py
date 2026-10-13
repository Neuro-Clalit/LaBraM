"""Session-level classical EEG features: the token builder, the loader's
``[age, *features]`` target, the model's extra tokens and the features-only
head, and an end-to-end fine-tune of both."""
import json

import numpy as np
import pytest
import torch
from torch import nn

import labram.models.registry  # noqa: F401
from dataset_maker import make_session_features as msf
from labram.configs.model_config import FinetuneModelConfig, TransformerArchConfig
from labram.data.session_features import load_session_features
from labram.losses import SessionFeatureAgeCriterion, model_condition, regression_target
from labram.models.neural_transformer import NeuralTransformer
from test_window_qc import _corpus as _npy_corpus

GROUPS = [[19, 5], [6, 171]]
N_FEATS = 19 * 5 + 6 * 171


def _tiny(mode, head_type="linear", **kw):
    return NeuralTransformer(TransformerArchConfig(
        eeg_window_size=200, patch_size=200, in_chans=1, out_chans=8, num_classes=1,
        embed_dim=200, depth=2, num_heads=10, init_values=0.1, head_type=head_type,
        session_features=mode, session_feature_groups=GROUPS, **kw))


# ------------------------------------------------------------------ builder

def _spd(rng, n):
    a = rng.normal(size=(n, 6, 19, 30))
    return a @ np.swapaxes(a, -1, -2) / 30


def _cache(path, stems, rng, families=("time", "psd_bins"), with_cov=True, offset=0.0):
    arrays = {f"rec/{f}": rng.normal(size=(len(stems), 19, d)) + offset
              for f, d in zip(families, (3, 2))}
    if with_cov:
        arrays["fbcov"] = _spd(rng, len(stems))
    np.savez(path, **{"meta/stem": np.array(stems)}, **arrays)


def _cohort(root, splits):
    root.mkdir(parents=True)
    json.dump({"recordings": [{"stem": s, "source_split": sp} for s, sp in splits.items()]},
              open(root / "manifest.json", "w"))
    return root


def test_builder_standardizes_with_training_stats_only(tmp_path):
    rng = np.random.default_rng(0)
    stems = [f"r{i:03d}" for i in range(40)]
    splits = {s: ("train" if i < 30 else "test") for i, s in enumerate(stems)}
    _cache(tmp_path / "a.npz", stems[:25], rng)
    _cache(tmp_path / "b.npz", stems[20:], rng, offset=100.0)    # r020-r024: a wins
    cohort = _cohort(tmp_path / "cohort", splits)
    msf.main(["--fit-cohort", str(cohort), "--cache", str(tmp_path / "a.npz"),
              "--cache", str(tmp_path / "b.npz"), "--families", "time", "psd_bins"])
    f = load_session_features(str(cohort / msf.DEFAULT_NAME))
    assert f.groups == [[19, 5], [6, 171]] and f.group_names == ["channels", "riemann"]
    train = f.matrix[[f.rows[s] for s in stems[:30]]]
    assert np.allclose(train.mean(0), 0, atol=1e-4)
    assert np.isfinite(f.matrix).all() and np.abs(f.matrix).max() <= 6.0


def test_builder_refuses_recordings_missing_from_every_cache(tmp_path):
    rng = np.random.default_rng(0)
    _cache(tmp_path / "a.npz", ["r1", "r2"], rng)
    cohort = _cohort(tmp_path / "cohort", {"r1": "train", "r2": "train", "r3": "test"})
    with pytest.raises(ValueError, match="r3"):
        msf.main(["--fit-cohort", str(cohort), "--cache", str(tmp_path / "a.npz"),
                  "--families", "time", "psd_bins"])


def test_tangent_space_of_the_reference_is_zero():
    rng = np.random.default_rng(1)
    covs = msf.project_covs(_spd(rng, 8))[:, 0]
    ref = msf.mean_riemann(covs)
    assert np.allclose(msf.tangent_space(ref[None], ref), 0, atol=1e-8)
    assert np.allclose(msf.tangent_space(covs, ref).mean(0), 0, atol=1e-6)   # Karcher mean


# ------------------------------------------------------------------ model

def test_tokens_mode_appends_one_token_per_feature_row():
    torch.manual_seed(0)
    model = _tiny("tokens").eval()
    x, feats = torch.randn(2, 23, 1, 200), torch.randn(2, N_FEATS)
    tokens = model.forward_features(x, return_all_tokens=True, session=feats)
    assert tokens.shape[1] == 1 + 23 + 19 + 6
    a = model(x, condition=feats)
    b = model(x, condition=torch.randn(2, N_FEATS))     # (LayerNorm cancels a shift)
    assert a.shape == (2, 1) and not torch.allclose(a, b, rtol=0, atol=1e-9)
    with pytest.raises(ValueError, match="session features"):
        model(x)


def test_only_mode_ignores_the_eeg():
    torch.manual_seed(0)
    model = _tiny("only", head_type="mlp").eval()
    feats = torch.randn(3, N_FEATS)
    a = model(torch.randn(3, 23, 1, 200), condition=feats)
    b = model(torch.zeros(3, 23, 1, 200), condition=feats)
    assert torch.equal(a, b) and not torch.allclose(a, model(None, condition=-feats))


def test_session_parameters_train_at_the_head_learning_rate():
    from labram.optim_factory import get_num_layer_for_vit
    model = _tiny("tokens")
    for name, _ in model.named_parameters():
        if name.startswith("session_"):
            assert get_num_layer_for_vit(name, 4) == 3


@pytest.mark.parametrize("kw, match", [
    (dict(session_features="maybe"), "must be"),
    (dict(session_features="tokens", task="classification"), "regression"),
    (dict(session_features="only", head_type="attention"), "linear or mlp"),
    (dict(session_features="tokens", condition_on_abnormal=True), "not supported"),
])
def test_config_is_validated(kw, match):
    with pytest.raises(ValueError, match=match):
        FinetuneModelConfig(**{"task": "regression", **kw}).validate()


def test_criterion_feeds_the_features_and_scores_age_only():
    crit = SessionFeatureAgeCriterion(nn.HuberLoss())
    tgt = torch.tensor([[0.3, 1.0, 2.0], [-0.2, 0.0, 4.0]])
    out = torch.tensor([[0.1], [0.0]])
    assert torch.equal(model_condition(crit, tgt), tgt[:, 1:])
    assert torch.equal(regression_target(crit, tgt), tgt[:, :1])
    assert torch.allclose(crit(out, tgt), nn.HuberLoss()(out, tgt[:, :1]))


# ------------------------------------------------------------------ data + run

def _feature_file(root, rng):
    stems = [r["stem"] for r in json.load(open(root / "manifest.json"))["recordings"]]
    msf.write(str(root / "feats.npz"), stems,
              {"channels": rng.normal(size=(len(stems), 19, 5)).astype(np.float32),
               "riemann": rng.normal(size=(len(stems), 6, 171)).astype(np.float32)}, {})
    return stems


def test_loader_returns_age_and_features_and_can_skip_the_eeg(tmp_path):
    from labram.data.tuh_datasets import prepare_TUAB_age_dataset
    from labram.data.window_selection import enable_session_features
    root = _npy_corpus(tmp_path)
    _feature_file(root, np.random.default_rng(0))
    train, _, _, stats = prepare_TUAB_age_dataset(str(root), data_format="npy")
    assert enable_session_features(train, "feats.npz") == GROUPS
    feats = load_session_features(str(root / "feats.npz"))
    x, y = train[0]
    stem = train.files[0].split("/")[-1].rsplit("_", 1)[0]
    assert y.shape == (1 + N_FEATS,) and torch.equal(y[1:], torch.from_numpy(feats.matrix[feats.rows[stem]]))
    assert x.abs().sum() > 0
    enable_session_features(train, str(root / "feats.npz"), skip_signal=True)
    assert train[0][0].abs().sum() == 0 and torch.equal(train[0][1], y)


@pytest.mark.parametrize("mode, head", [("tokens", "linear"), ("only", "mlp")])
def test_run_finetune_end_to_end(tmp_path, mode, head):
    from labram.configs.run_configs import FinetuneRunConfig
    from labram.runs import run_finetune
    root = _npy_corpus(tmp_path / "data")
    _feature_file(root, np.random.default_rng(0))
    out = tmp_path / "out"
    cfg = FinetuneRunConfig.load_config(
        "labram/configs/defaults/finetune_tuab_age.json",
        **{"data.data_path": str(root), "data.data_format": "npy",
           "data.trim_start_sec": 0.0, "data.trim_end_sec": 0.0, "data.eval_minutes": 0.0,
           "data.num_workers": 0, "data.session_features": "feats.npz",
           "trainer.debug": True, "trainer.epochs": 1,
           "distributed.device": "cpu", "finetune_checkpoint.finetune": "",
           "clearml.enabled": False, "output.output_dir": str(out), "output.log_dir": str(out),
           "model.session_features": mode, "model.head_type": head,
           "mixup.enabled": True, "logging.log_model_graph": False})
    summary = run_finetune.main(cfg)
    assert summary["best_metric_name"] == "mae" and "mae" in summary["best_val_stats"]
    assert cfg.model.session_feature_groups == GROUPS


def test_model_and_data_switches_go_together(tmp_path):
    from labram.configs.run_configs import FinetuneRunConfig
    from labram.runs import run_finetune
    root = _npy_corpus(tmp_path / "data")
    cfg = FinetuneRunConfig.load_config(
        "labram/configs/defaults/finetune_tuab_age.json",
        **{"data.data_path": str(root), "data.data_format": "npy", "data.num_workers": 0,
           "data.trim_start_sec": 0.0, "data.trim_end_sec": 0.0, "data.eval_minutes": 0.0,
           "trainer.debug": True, "distributed.device": "cpu", "finetune_checkpoint.finetune": "",
           "clearml.enabled": False, "output.output_dir": str(tmp_path / "o"),
           "output.log_dir": str(tmp_path / "o"), "model.session_features": "tokens"})
    with pytest.raises(ValueError, match="go together"):
        run_finetune.main(cfg)


@pytest.mark.parametrize("norm, pre, post", [("post", False, True), ("pre", True, False),
                                             ("both", True, True), ("none", False, False)])
def test_feature_token_normalization(norm, pre, post):
    torch.manual_seed(0)
    model = _tiny("tokens", session_feature_norm=norm, session_token_gain=0.5).eval()
    proj = model.session_proj[0]
    assert isinstance(proj[0], nn.LayerNorm) == pre and isinstance(proj[2], nn.LayerNorm) == post
    if post:      # unit-RMS tokens scaled to the gain: the EEG tokens' scale
        assert torch.all(proj[2].weight == 0.5)
        with torch.no_grad():
            tokens = model.session_tokens(torch.randn(4, N_FEATS)) - model.session_token_embed
        rms = tokens.pow(2).mean(-1).sqrt()          # (LayerNorm eps shaves a little off)
        assert 0.45 < rms.min() and rms.max() <= 0.5 + 1e-4


def test_unknown_normalization_is_rejected():
    with pytest.raises(ValueError, match="session_feature_norm"):
        FinetuneModelConfig(task="regression", session_features="tokens",
                            session_feature_norm="batch").validate()
