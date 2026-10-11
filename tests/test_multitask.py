"""Multi-task brain age: a shared MLP trunk feeding a dedicated age-regression
branch and a dedicated normal/abnormal branch (model.head_type=multitask)."""
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import labram.models.registry  # noqa: F401
from labram.configs.model_config import FinetuneModelConfig, TransformerArchConfig
from labram.layers import MultiTaskHead
from labram.losses import LossBreakdown, MultiTaskAgeCriterion, regression_output, regression_target
from labram.models.neural_transformer import NeuralTransformer


def _tiny(head_type="multitask", **kw):
    return NeuralTransformer(TransformerArchConfig(
        eeg_window_size=200, patch_size=200, in_chans=1, out_chans=8, num_classes=1,
        embed_dim=200, depth=2, num_heads=10, init_values=0.1, head_type=head_type, **kw))


# ------------------------------------------------------------------ head
def test_head_has_a_shared_trunk_and_two_dedicated_branches():
    model = _tiny(multitask_head={"shared_dims": [128, 64], "age_dims": [32, 16],
                                  "abnormal_dims": [8], "drop": 0.1})
    head = model.head
    assert isinstance(head, MultiTaskHead)
    widths = lambda seq: [m.out_features for m in seq if isinstance(m, nn.Linear)]
    assert widths(head.shared) == [128, 64]
    assert widths(head.age) == [32, 16] and head.age_fc.in_features == 16
    assert widths(head.abnormal) == [8] and head.abnormal_fc.in_features == 8
    out = model(torch.randn(3, 4, 1, 200))
    assert out.shape == (3, 2)                      # [age, abnormal_logit]


def test_both_branches_train_the_shared_trunk_and_init_scale_shrinks_both_outputs():
    torch.manual_seed(0)
    small = _tiny(init_scale=0.001).head
    torch.manual_seed(0)
    full = _tiny(init_scale=1.0).head
    for name in ("age_fc", "abnormal_fc"):
        assert torch.isclose(getattr(small, name).weight.abs().mean(),
                             getattr(full, name).weight.abs().mean() * 0.001)
    model = _tiny()
    out = model(torch.randn(2, 4, 1, 200))
    out[:, 1].sum().backward()                      # abnormal branch only
    assert model.head.shared[0].weight.grad.abs().sum() > 0
    assert model.head.abnormal[0].weight.grad.abs().sum() > 0
    assert model.head.age[0].weight.grad is None or model.head.age[0].weight.grad.abs().sum() == 0
    assert model.blocks[0].attn.qkv.weight.grad is not None


def test_head_survives_reset_and_is_a_top_layer_group():
    from labram.optim_factory import get_num_layer_for_vit
    model = _tiny()
    model.reset_classifier(1)
    assert isinstance(model.head, MultiTaskHead)
    assert get_num_layer_for_vit("head.abnormal_fc.weight", 4) == 3


@pytest.mark.parametrize("kw, match", [
    (dict(head_type="multitask", task="classification"), "regression"),
    (dict(head_type="multitask", task="regression", lora={"enabled": True}), "lora"),
])
def test_config_is_validated(kw, match):
    from labram.configs.model_config import LoRAConfig
    if "lora" in kw:
        kw["lora"] = LoRAConfig(**kw["lora"])
    with pytest.raises(ValueError, match=match):
        FinetuneModelConfig(**kw).validate()


# ------------------------------------------------------------- criterion
def test_criterion_sums_age_loss_and_weighted_bce_and_scores_the_age_column():
    crit = MultiTaskAgeCriterion(nn.HuberLoss(), abnormal_weight=0.5)
    out = torch.tensor([[0.2, 1.5], [-0.3, -2.0]])
    tgt = torch.tensor([[0.0, 1.0], [0.1, 0.0]])
    res = crit(out, tgt)
    assert isinstance(res, LossBreakdown)
    age = nn.HuberLoss()(out[:, :1], tgt[:, :1])
    bce = nn.functional.binary_cross_entropy_with_logits(out[:, 1], tgt[:, 1])
    assert torch.allclose(res.components["regression"], age)
    assert torch.allclose(res.components["abnormal"], bce)
    assert torch.allclose(res.total, age + 0.5 * bce)
    assert torch.equal(regression_output(crit, out), out[:, :1])
    assert torch.equal(regression_target(crit, tgt), tgt[:, :1])
    # Every other criterion passes outputs/targets through untouched.
    assert regression_target(nn.HuberLoss(), tgt) is tgt


def test_cmixup_pairs_on_the_age_column_of_a_two_column_target():
    from labram.train.mixup import mixup_partners
    torch.manual_seed(0)
    t = torch.tensor([[0.0, 1.0], [0.01, 0.0], [5.0, 1.0], [5.01, 0.0]])
    partners = torch.stack([mixup_partners(t, sigma=0.5) for _ in range(200)], dim=1)
    assert (partners[0] == 1).float().mean() > 0.99 and (partners[2] == 3).float().mean() > 0.99


# ------------------------------------------------------------------ data
def _labelled_corpus(tmp_path, n_subjects=6, windows=2):
    from labram.data.tuh_metadata import RecordingMetadata, save_metadata_sidecar
    processed = tmp_path / "processed"
    meta = {}
    for split, subjects in (("train", range(0, n_subjects)), ("val", range(n_subjects, n_subjects + 2)),
                            ("test", range(n_subjects + 2, n_subjects + 4))):
        (processed / split).mkdir(parents=True, exist_ok=True)
        for k in subjects:
            stem = f"aaaaa{k:03d}_s001_t000"
            meta[stem] = RecordingMetadata(
                stem=stem, subject=stem[:8], session="s001", token="t000", age=20 + 5 * k,
                sex="F", year=2012, raw_age=20 + 5 * k, label="abnormal" if k % 2 else "normal")
            for w in range(windows):
                with open(processed / split / f"{stem}_{w}.pkl", "wb") as fh:
                    pickle.dump({"X": np.random.randn(23, 2000), "y": 0}, fh)
    save_metadata_sidecar(meta, str(processed / "age_metadata.json"))
    return tmp_path, processed, meta


def test_loader_returns_age_and_abnormal_once_enabled(tmp_path):
    from labram.data.tuh_datasets import TUABAgeLoader
    from labram.data.window_selection import enable_abnormal_target
    _, processed, meta = _labelled_corpus(tmp_path)
    files = sorted(f.name for f in (processed / "train").glob("*.pkl"))
    loader = TUABAgeLoader(str(processed / "train"), files)
    assert loader[0][1].ndim == 0                   # plain age target by default
    enable_abnormal_target(loader)
    for i, name in enumerate(files):
        stem = name.rsplit("_", 1)[0]
        y = loader[i][1]
        assert y.shape == (2,)
        assert float(y[0]) == meta[stem].age
        assert float(y[1]) == float(meta[stem].label == "abnormal")


def test_enabling_refuses_recordings_without_a_label(tmp_path):
    from labram.data.tuh_datasets import TUABAgeLoader
    from labram.data.tuh_metadata import load_metadata_sidecar, save_metadata_sidecar
    from labram.data.window_selection import enable_abnormal_target
    _, processed, _ = _labelled_corpus(tmp_path)
    meta = load_metadata_sidecar(str(processed / "age_metadata.json"))
    first = sorted(meta)[0]
    import dataclasses
    meta[first] = dataclasses.replace(meta[first], label=None)
    save_metadata_sidecar(meta, str(processed / "age_metadata.json"))
    files = sorted(f.name for f in (processed / "train").glob("*.pkl"))
    loader = TUABAgeLoader(str(processed / "train"), files)
    with pytest.raises(ValueError, match="no normal/abnormal label"):
        enable_abnormal_target(loader)


# ---------------------------------------------------------- train / eval
class _Cases(Dataset):
    """(X, [age_z, abnormal], case_id): 4 cases x 2 windows."""

    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randn(8, 4, 200, generator=g)
        self.case = [f"c{i // 2}" for i in range(8)]
        self.y = torch.tensor([[(i // 2) - 1.5, float((i // 2) % 2)] for i in range(8)])

    def __len__(self):
        return 8

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.case[i]


def test_train_and_evaluate_report_age_and_abnormal_metrics():
    from labram.configs.optim_config import OptimizerConfig
    from labram.configs.train_config import EvaluationConfig, MixupConfig
    from labram.train.train_finetune import evaluate, train_one_epoch
    from test_finetune import _make_epoch_args
    torch.manual_seed(0)
    model = _tiny()
    loader = DataLoader(_Cases(), batch_size=4)
    crit = MultiTaskAgeCriterion(nn.HuberLoss(), 0.5)
    args = _make_epoch_args(model, loader, crit, is_binary=False)
    args.update(task="regression", nb_classes=1, target_stats=(50.0, 15.0),
                optim_cfg=OptimizerConfig(clip_grad=1.0),
                mixup_cfg=MixupConfig(enabled=True, alpha=0.4, sigma=5.0))
    stats = train_one_epoch(**args)
    assert stats["mae"] >= 0 and "regression_loss" in stats and "abnormal_loss" in stats
    res = evaluate(data_loader=loader, model=model, device=torch.device("cpu"),
                   metrics=["mae", "r2"], is_binary=False, nb_classes=1, task="regression",
                   target_stats=(50.0, 15.0), criterion=crit,
                   eval_cfg=EvaluationConfig(agg_windows="mean", detailed_metrics=False))
    assert res["mae"] >= 0 and "window_mae" in res                     # age, per case
    for key in ("abnormal_roc_auc", "abnormal_balanced_accuracy",
                "window_abnormal_roc_auc", "window_abnormal_accuracy"):
        assert 0.0 <= res[key] <= 1.0, key
    assert "abnormal_loss" in res and "regression_loss" in res


def test_run_finetune_trains_a_multitask_model_end_to_end(tmp_path):
    from labram.configs.run_configs import FinetuneRunConfig
    from labram.runs import run_finetune
    root, _, _ = _labelled_corpus(tmp_path / "data")
    out = tmp_path / "out"
    cfg = FinetuneRunConfig.load_config(
        "labram/configs/defaults/finetune_tuab_age.json",
        **{"data.data_path": str(root), "data.data_format": "pickle",
           "data.trim_start_sec": 0.0, "data.trim_end_sec": 0.0, "data.eval_minutes": 0.0,
           "data.num_workers": 0, "trainer.debug": True, "trainer.epochs": 1,
           "distributed.device": "cpu", "finetune_checkpoint.finetune": "",
           "clearml.enabled": False, "output.output_dir": str(out), "output.log_dir": str(out),
           "model.head_type": "multitask", "mixup.enabled": True, "logging.log_model_graph": False})
    summary = run_finetune.main(cfg)
    val = summary["best_val_stats"]
    assert summary["best_metric_name"] == "mae"            # selection stays on age
    for key in ("mae", "abnormal_roc_auc", "window_abnormal_accuracy",
                "regression_loss", "abnormal_loss"):
        assert key in val, key


def test_multitask_config_is_the_best_run_with_the_multitask_head():
    """finetune_tuab_age_multitask.json = M1 (scenario D + mixup 0.4, 4 epochs /
    1 warmup, pretrained) with the shared-trunk age + abnormal head."""
    from labram.configs.run_configs import FinetuneRunConfig
    cfg = FinetuneRunConfig.load_config("labram/configs/defaults/finetune_tuab_age_multitask.json")
    assert cfg.model.head_type == "multitask" and cfg.loss.abnormal_weight == 0.5
    assert (cfg.trainer.epochs, cfg.optimizer.warmup_epochs, cfg.optimizer.lr) == (4, 1, 1e-4)
    assert cfg.mixup.enabled and cfg.mixup.alpha == 0.4
    assert cfg.labram_plus.enabled and not cfg.labram_plus.z_score_patches
    assert cfg.finetune_checkpoint.finetune.endswith("labram-base.pth")
    assert cfg.optimizer.layer_decay == 0.65 and cfg.model.depth is None
