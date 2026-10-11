"""Age head conditioned on the known normal/abnormal label
(model.condition_on_abnormal): the label is an input feature, not predicted."""
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

import labram.models.registry  # noqa: F401
from labram.configs.model_config import FinetuneModelConfig, TransformerArchConfig
from labram.losses import AbnormalConditionedAgeCriterion, model_condition, regression_target
from labram.models.neural_transformer import NeuralTransformer
from test_multitask import _Cases, _labelled_corpus


def _tiny(head_type="mlp", **kw):
    return NeuralTransformer(TransformerArchConfig(
        eeg_window_size=200, patch_size=200, in_chans=1, out_chans=8, num_classes=1,
        embed_dim=200, depth=2, num_heads=10, init_values=0.1, head_type=head_type,
        condition_on_abnormal=True, **kw))


def test_label_feature_is_concatenated_to_the_pooled_feature():
    model = _tiny(abnormal_embed_dim=16)
    assert model.num_features == 216 and model.head.hidden[0].in_features == 216
    x = torch.randn(3, 4, 1, 200)
    assert model(x, condition=torch.tensor([0.0, 1.0, 1.0])).shape == (3, 1)
    with pytest.raises(ValueError, match="condition"):
        model(x)
    lin = _tiny(head_type="linear", abnormal_embed_dim=8)
    assert lin.head.in_features == 208


def test_the_label_changes_the_prediction_and_a_soft_label_interpolates_the_feature():
    torch.manual_seed(0)
    model = _tiny(init_scale=1.0).eval()
    x = torch.randn(2, 4, 1, 200)
    with torch.no_grad():
        normal = model(x, condition=torch.zeros(2))
        abnormal = model(x, condition=torch.ones(2))
        assert not torch.allclose(normal, abnormal)
        e0, e1, eh = (model.embed_abnormal(torch.tensor([v])) for v in (0.0, 1.0, 0.5))
        assert torch.allclose(eh, 0.5 * (e0 + e1), atol=1e-6)    # mixup-consistent


def test_label_embedding_trains_and_is_a_top_layer_group():
    from labram.optim_factory import get_num_layer_for_vit
    model = _tiny()
    model(torch.randn(2, 4, 1, 200), condition=torch.tensor([0.0, 1.0])).sum().backward()
    assert model.abnormal_embed.weight.grad.abs().sum() > 0
    assert get_num_layer_for_vit("abnormal_embed.weight", 4) == 3


@pytest.mark.parametrize("kw, match", [
    (dict(task="classification"), "regression"),
    (dict(task="regression", head_type="attention"), "linear or mlp"),
    (dict(task="regression", head_type="multitask"), "linear or mlp"),
    (dict(task="regression", abnormal_embed_dim=0), "abnormal_embed_dim"),
])
def test_config_is_validated(kw, match):
    with pytest.raises(ValueError, match=match):
        FinetuneModelConfig(condition_on_abnormal=True, **kw).validate()


def test_criterion_feeds_the_label_to_the_model_and_scores_age_only():
    crit = AbnormalConditionedAgeCriterion(nn.HuberLoss())
    tgt = torch.tensor([[0.3, 1.0], [-0.2, 0.0]])
    out = torch.tensor([[0.1], [0.0]])
    assert torch.equal(model_condition(crit, tgt), tgt[:, 1])
    assert torch.equal(regression_target(crit, tgt), tgt[:, :1])
    assert torch.allclose(crit(out, tgt), nn.HuberLoss()(out, tgt[:, :1]))
    assert model_condition(nn.HuberLoss(), tgt) is None


def test_train_and_evaluate_pass_the_true_label_to_the_model():
    from labram.configs.train_config import EvaluationConfig, MixupConfig
    from labram.train.train_finetune import evaluate, train_one_epoch
    from test_finetune import _make_epoch_args
    torch.manual_seed(0)
    model = _tiny()
    seen = []
    real = model.embed_abnormal
    model.embed_abnormal = lambda a: (seen.append(a.detach().clone()), real(a))[1]
    loader = DataLoader(_Cases(), batch_size=4)
    crit = AbnormalConditionedAgeCriterion(nn.HuberLoss())
    args = _make_epoch_args(model, loader, crit, is_binary=False)
    args.update(task="regression", nb_classes=1, target_stats=(50.0, 15.0),
                mixup_cfg=MixupConfig(enabled=True, alpha=0.4))
    stats = train_one_epoch(**args)
    assert stats["mae"] >= 0 and seen
    seen.clear()
    res = evaluate(data_loader=loader, model=model, device=torch.device("cpu"),
                   metrics=["mae"], is_binary=False, nb_classes=1, task="regression",
                   target_stats=(50.0, 15.0), criterion=crit,
                   eval_cfg=EvaluationConfig(agg_windows="mean", detailed_metrics=False))
    assert res["mae"] >= 0 and not any(k.startswith("abnormal_") for k in res)
    labels = torch.cat(seen)
    assert torch.equal(labels, _Cases().y[:, 1])        # exact 0/1 labels at eval


def test_run_finetune_trains_a_label_conditioned_model_end_to_end(tmp_path):
    from labram.configs.run_configs import FinetuneRunConfig
    from labram.runs import run_finetune
    root, _, _ = _labelled_corpus(tmp_path / "data")
    out = tmp_path / "out"
    cfg = FinetuneRunConfig.load_config(
        "labram/configs/defaults/finetune_tuab_age_label_cond.json",
        **{"data.data_path": str(root), "data.data_format": "pickle",
           "data.trim_start_sec": 0.0, "data.trim_end_sec": 0.0, "data.eval_minutes": 0.0,
           "data.num_workers": 0, "trainer.debug": True, "trainer.epochs": 1,
           "distributed.device": "cpu", "finetune_checkpoint.finetune": "",
           "clearml.enabled": False, "output.output_dir": str(out), "output.log_dir": str(out),
           "logging.log_model_graph": False})
    summary = run_finetune.main(cfg)
    assert summary["best_metric_name"] == "mae" and "mae" in summary["best_val_stats"]


def test_label_cond_config_is_the_best_run_with_a_label_conditioned_mlp_head():
    from labram.configs.run_configs import FinetuneRunConfig
    cfg = FinetuneRunConfig.load_config("labram/configs/defaults/finetune_tuab_age_label_cond.json")
    assert cfg.model.condition_on_abnormal and cfg.model.head_type == "mlp"
    assert cfg.model.abnormal_embed_dim == 16 and cfg.model.mlp_head.hidden_dims == [256]
    assert (cfg.trainer.epochs, cfg.optimizer.warmup_epochs, cfg.optimizer.lr) == (4, 1, 1e-4)
    assert cfg.mixup.enabled and cfg.mixup.alpha == 0.4
    assert cfg.finetune_checkpoint.finetune.endswith("labram-base.pth")
