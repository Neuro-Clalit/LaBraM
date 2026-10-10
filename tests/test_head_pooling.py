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


# --- head_type="attention": CLSAttentionHead ------------------------------

def _attn_model(**kw):
    return _model(head_type="attention", **kw)


def test_attention_head_predicts_from_all_tokens():
    from labram.layers import CLSAttentionHead
    model = _attn_model()
    assert isinstance(model.head, CLSAttentionHead)
    assert model.fc_norm is None and model.cls_norm is None
    assert isinstance(model.norm, torch.nn.LayerNorm)  # pretrained final norm kept
    x = torch.randn(2, 4, 10, 200)
    assert model.forward_features(x).shape == (2, 41, 200)  # cls + 4 x 10 patches
    model(x).sum().backward()
    for name in ("head.q.weight", "head.kv.weight", "head.proj.weight", "head.fc.weight",
                 "cls_token", "blocks.0.attn.qkv.weight"):
        assert dict(model.named_parameters())[name].grad is not None, name


def test_attention_head_returns_cls_attention_weights():
    model = _attn_model(attention_head={"num_heads": 5}).eval()
    x = torch.randn(3, 4, 10, 200)
    with torch.no_grad():
        out, attn = model.forward_with_attention(x)
        assert torch.allclose(out, model(x))  # same prediction as forward
    assert out.shape == (3, 1)
    assert attn.shape == (3, 5, 41)
    assert torch.allclose(attn.sum(-1), torch.ones(3, 5))


def test_attention_head_equals_cls_row_of_full_self_attention():
    from labram.layers import CLSAttentionHead
    torch.manual_seed(0)
    head = CLSAttentionHead(16, 1, num_heads=4, mlp_hidden_dim=8).eval()
    tokens = torch.randn(2, 7, 16)
    cls, attn = head._attend(tokens)
    q = head.q(tokens).reshape(2, 7, 4, 4).transpose(1, 2)
    k, v = head.kv(tokens).reshape(2, 7, 2, 4, 4).permute(2, 0, 3, 1, 4)
    full = (q @ k.transpose(-2, -1) * 4 ** -0.5).softmax(-1)
    rows = (full @ v).transpose(1, 2).reshape(2, 7, 16)
    assert torch.allclose(attn, full[:, :, 0], atol=1e-6)
    assert torch.allclose(cls, tokens[:, 0] + head.proj(rows[:, 0]), atol=1e-6)


def test_attention_head_output_layer_uses_init_scale():
    torch.manual_seed(0)
    small = _attn_model(init_scale=0.001).head.fc.weight.abs().mean()
    torch.manual_seed(0)
    full = _attn_model(init_scale=1.0).head.fc.weight.abs().mean()
    assert torch.isclose(small, full * 0.001)


def test_attention_head_survives_reset_and_is_a_top_layer_for_layer_decay():
    from labram.layers import CLSAttentionHead
    from labram.optim_factory import get_num_layer_for_vit
    model = _attn_model()
    model.reset_classifier(30)  # e.g. soft-label bins
    assert isinstance(model.head, CLSAttentionHead) and model.head.fc.out_features == 30
    assert get_num_layer_for_vit("head.kv.weight", 14) == 13


def test_forward_with_attention_requires_the_attention_head():
    with pytest.raises(ValueError, match="head_type=attention"):
        _model().forward_with_attention(torch.randn(1, 4, 10, 200))


def test_linear_head_checkpoint_loads_into_attention_head(tmp_path):
    from types import SimpleNamespace
    from labram.runs.finetune_setup import load_finetune_checkpoint
    src = _model()
    path = tmp_path / "ckpt.pth"
    torch.save({"model": src.state_dict()}, path)
    dst = _attn_model()
    load_finetune_checkpoint(dst, SimpleNamespace(
        finetune=str(path), model_key="model", model_filter_name="", model_prefix=""))
    assert torch.equal(dst.blocks[3].mlp.fc1.weight, src.blocks[3].mlp.fc1.weight)


@pytest.mark.parametrize("kw, match", [
    (dict(head_type="transformer"), "head_type"),
    (dict(head_type="attention", concat_cls_token=True), "concat_cls_token"),
    (dict(head_type="attention", attention_head={"num_heads": 3, "attn_dim": 200}), "divisible"),
])
def test_attention_head_config_is_validated(kw, match):
    from labram.configs.model_config import AttentionHeadConfig
    if "attention_head" in kw:
        kw["attention_head"] = AttentionHeadConfig(**kw["attention_head"])
    with pytest.raises(ValueError, match=match):
        FinetuneModelConfig(task="regression", **kw).validate()


def test_attention_head_config_is_the_best_run_plus_clip_and_attention():
    """finetune_tuab_age_attn_head.json = the best brain-age run so far (M1:
    scenario D + mixup 0.4 on the 4-epoch schedule) + clip_grad 1.0 + the
    CLS-attention head."""
    from labram.configs.run_configs import FinetuneRunConfig
    cfg = FinetuneRunConfig.load_config("labram/configs/defaults/finetune_tuab_age_attn_head.json")
    assert cfg.model.head_type == "attention" and cfg.optimizer.clip_grad == 1.0
    assert (cfg.trainer.epochs, cfg.optimizer.warmup_epochs, cfg.optimizer.lr) == (4, 1, 1e-4)
    assert cfg.mixup.enabled and cfg.mixup.alpha == 0.4 and cfg.mixup.sigma == 0.0
    assert cfg.labram_plus.enabled and not cfg.labram_plus.z_score_patches
    cfg.model.validate()


def test_attention_head_trains_and_evaluates_with_mixup_and_grad_clip():
    """The attn-head config's recipe through the real loop: regression,
    mixup 0.4, clip_grad 1.0, then evaluate."""
    from torch import nn
    from labram.configs.model_config import TransformerArchConfig
    from labram.configs.optim_config import OptimizerConfig
    from labram.configs.train_config import MixupConfig
    from labram.models.neural_transformer import NeuralTransformer
    from labram.train.train_finetune import evaluate, train_one_epoch
    from test_anti_memorization import _regression_loader
    from test_finetune import _make_epoch_args

    model = NeuralTransformer(TransformerArchConfig(
        eeg_window_size=200, patch_size=200, in_chans=1, out_chans=8, num_classes=1,
        embed_dim=200, depth=2, num_heads=10, init_values=0.1, head_type="attention"))
    loader = _regression_loader()
    args = _make_epoch_args(model, loader, nn.HuberLoss(), is_binary=False)
    args.update(task="regression", nb_classes=1, target_stats=(10.0, 5.0),
                optim_cfg=OptimizerConfig(clip_grad=1.0),
                mixup_cfg=MixupConfig(enabled=True, alpha=0.4))
    before = model.head.q.weight.detach().clone()
    stats = train_one_epoch(**args)
    assert stats["mae"] >= 0 and not torch.equal(before, model.head.q.weight)
    res = evaluate(data_loader=loader, model=model, device=torch.device("cpu"),
                   metrics=["mae", "r2"], is_binary=False, nb_classes=1,
                   task="regression", target_stats=(10.0, 5.0))
    assert res["mae"] >= 0


# --- head_type="mlp" and model.depth ---------------------------------------

def test_mlp_head_on_the_pooled_feature():
    from labram.layers import MLPHead
    model = _model(head_type="mlp", mlp_head={"hidden_dims": [128, 64], "drop": 0.1})
    assert isinstance(model.head, MLPHead) and model.fc_norm is not None
    assert [m.out_features for m in model.head.hidden if isinstance(m, torch.nn.Linear)] == [128, 64]
    assert model.head.fc.in_features == 64 and model.head.fc.out_features == 1
    x = torch.randn(2, 4, 10, 200)
    assert model.forward_features(x).shape == (2, 200)  # same pooled input as the linear head
    model(x).sum().backward()
    assert model.head.hidden[0].weight.grad is not None and model.blocks[0].attn.qkv.weight.grad is not None
    model.reset_classifier(30)
    assert isinstance(model.head, MLPHead) and model.head.fc.out_features == 30


def test_mlp_head_output_layer_uses_init_scale_and_concat_cls_widens_its_input():
    torch.manual_seed(0)
    small = _model(head_type="mlp", init_scale=0.001).head.fc.weight.abs().mean()
    torch.manual_seed(0)
    full = _model(head_type="mlp", init_scale=1.0).head.fc.weight.abs().mean()
    assert torch.isclose(small, full * 0.001)
    assert _model(head_type="mlp", concat_cls_token=True).head.hidden[0].in_features == 400


def test_depth_sets_the_number_of_blocks_and_layer_decay_groups():
    from labram.configs.run_configs import FinetuneRunConfig
    from labram.runs.run_finetune import get_model
    cfg = FinetuneRunConfig.load_config(
        "labram/configs/defaults/finetune_tuab_age.json",
        **{"model.depth": 6, "model.head_type": "mlp", "finetune_checkpoint.finetune": ""})
    cfg.model.nb_classes = 1
    cfg.model.validate()
    model = get_model(cfg)
    assert len(model.blocks) == 6 and model.get_num_layers() == 6
    full = _model()
    n = lambda m: sum(p.numel() for p in m.blocks.parameters())
    assert n(model) * 2 == n(full)


def test_pretrained_checkpoint_loads_into_a_shallower_encoder(tmp_path):
    from types import SimpleNamespace
    from labram.runs.finetune_setup import load_finetune_checkpoint
    src = _model()
    path = tmp_path / "ckpt.pth"
    torch.save({"model": src.state_dict()}, path)
    dst = _model(depth=4, head_type="mlp")
    load_finetune_checkpoint(dst, SimpleNamespace(
        finetune=str(path), model_key="model", model_filter_name="", model_prefix=""))
    assert torch.equal(dst.blocks[3].mlp.fc1.weight, src.blocks[3].mlp.fc1.weight)


@pytest.mark.parametrize("kw, match", [
    (dict(depth=0), "depth"),
    (dict(head_type="mlp", codebook_reg={"enabled": True}), "head_type=mlp"),
])
def test_mlp_head_and_depth_config_is_validated(kw, match):
    from labram.configs.model_config import CodebookRegConfig
    if "codebook_reg" in kw:
        kw["codebook_reg"] = CodebookRegConfig(**kw["codebook_reg"])
    with pytest.raises(ValueError, match=match):
        FinetuneModelConfig(task="regression", **kw).validate()
