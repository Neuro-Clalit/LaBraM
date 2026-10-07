"""Fine-tune head input: mean patch token, cls token, or both concatenated."""
import pytest
import torch

import labram.models.registry  # noqa: F401
from labram.configs.model_config import FinetuneModelConfig


def _model(**kw):
    from timm.models import create_model
    return create_model("labram_base_patch200_200", num_classes=1, init_values=0.1, **kw)


def test_mean_pooling_is_the_default_and_unchanged():
    model = _model()
    assert model.cls_norm is None
    assert model.head.in_features == 200
    assert model.forward_features(torch.randn(2, 4, 10, 200)).shape == (2, 200)


def test_cls_token_head_uses_the_final_norm():
    model = _model(use_mean_pooling=False)
    assert model.fc_norm is None and model.cls_norm is None
    assert isinstance(model.norm, torch.nn.LayerNorm)
    assert model(torch.randn(2, 4, 10, 200)).shape == (2, 1)


def test_concat_cls_token_doubles_the_head_input():
    model = _model(concat_cls_token=True)
    x = torch.randn(2, 4, 10, 200)
    feats = model.forward_features(x)
    assert feats.shape == (2, 400)
    assert model.head.in_features == 400
    model(x).sum().backward()
    assert model.cls_norm.weight.grad is not None
    assert model.cls_token.grad is not None
    model.reset_classifier(3)
    assert model.head.in_features == 400


def test_concat_cls_token_first_half_is_the_mean_pooled_feature():
    torch.manual_seed(0)
    model = _model(concat_cls_token=True).eval()
    x = torch.randn(2, 4, 10, 200)
    with torch.no_grad():
        feats = model.forward_features(x)
        model.cls_norm = None
        assert torch.allclose(feats[:, :200], model.forward_features(x))


def test_concat_cls_token_requires_mean_pooling():
    cfg = FinetuneModelConfig(task="regression", use_mean_pooling=False, concat_cls_token=True)
    with pytest.raises(ValueError, match="concat_cls_token"):
        cfg.validate()
