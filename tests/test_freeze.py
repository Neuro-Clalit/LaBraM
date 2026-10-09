"""model.trainable_prefixes: freeze everything except the named parameters."""
import pytest
import torch

import labram.models.registry  # noqa: F401
from labram.optim_factory import get_parameter_groups
from labram.runs.finetune_setup import freeze_except


def _model():
    from timm.models import create_model
    return create_model("labram_base_patch200_200", num_classes=1, init_values=0.1)


def test_head_only_leaves_just_the_head_trainable():
    model = _model()
    trainable = freeze_except(model, ["head"])
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert names == {"head.weight", "head.bias"}
    assert trainable == 201                      # 200-dim pooled feature -> 1 output


def test_frozen_parameters_never_reach_the_optimizer():
    model = _model()
    freeze_except(model, ["head", "blocks.11"])
    grouped = {id(t) for g in get_parameter_groups(model, 0.05, set()) for t in g["params"]}
    expected = {id(p) for n, p in model.named_parameters() if n.startswith(("head", "blocks.11"))}
    assert grouped == expected


def test_frozen_backbone_gets_no_gradient():
    model = _model()
    freeze_except(model, ["head"])
    model(torch.randn(2, 4, 10, 200)).sum().backward()
    assert model.head.weight.grad is not None
    assert all(p.grad is None for n, p in model.named_parameters() if not n.startswith("head"))


def test_an_unmatched_prefix_raises():
    with pytest.raises(ValueError, match="matches no parameter"):
        freeze_except(_model(), ["heads"])
