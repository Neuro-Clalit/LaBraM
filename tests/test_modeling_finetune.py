"""Tests for modeling_finetune.{PatchEmbed, TemporalConv, NeuralTransformer}.

Focus: surface-level regressions in the renamed kwargs (eeg_window_size,
channel_indices) and the basic forward shapes for the building blocks
that do *not* depend on the channel-pos-embed broadcast (which has a
known limitation when channel_indices is None and use_abs_pos_emb is on).
"""
import pytest
import torch

from modeling_finetune import (
    Block,
    NeuralTransformer,
    PatchEmbed,
    TemporalConv,
)


class TestTemporalConv:
    def test_forward_shape(self):
        tc = TemporalConv(in_chans=1, out_chans=8)
        # input layout: (B, N_channels, n_patches, patch_size)
        x = torch.randn(2, 4, 8, 200)
        out = tc(x)
        # TemporalConv collapses (N, A) and emits (B, N*A, T_out * out_chans)
        # patch_size 200 -> conv1 stride 8, padding 7, kernel 15 keeps T at ceil(200/8) = 25
        # subsequent convs preserve T. With out_chans=8, output last dim = 25*8 = 200.
        assert out.shape == (2, 4 * 8, 200)

    def test_in_chans_default_one(self):
        tc = TemporalConv()
        assert tc.conv1.in_channels == 1
        assert tc.conv1.out_channels == 8


class TestPatchEmbed:
    def test_eeg_window_size_kwarg(self):
        pe = PatchEmbed(eeg_window_size=2000, patch_size=200, in_chans=1, embed_dim=200)
        assert pe.eeg_window_size == 2000
        assert pe.patch_size == 200
        # 62 channels (default) * (2000//200=10) = 620 patches
        assert pe.num_patches == 620

    def test_old_kwarg_silently_swallowed(self):
        # Document the current behavior: **kwargs is not present on PatchEmbed,
        # so passing the old EEG_size= raises TypeError. This is the desired
        # behavior post-rename.
        with pytest.raises(TypeError):
            PatchEmbed(EEG_size=2000, patch_size=200, in_chans=1, embed_dim=200)

    def test_forward_shape(self):
        pe = PatchEmbed(eeg_window_size=2000, patch_size=200, in_chans=1, embed_dim=200)
        # input shape: (B, in_chans, H, W) where H*W reduces via the conv
        # Conv2d kernel=(1, patch_size), stride=(1, patch_size)
        x = torch.randn(2, 1, 1, 2000)
        out = pe(x)
        # (B, num_patches=10, embed_dim=200)
        assert out.shape == (2, 10, 200)


def _make_tiny_model(**overrides):
    """Tiny 2-block NeuralTransformer suitable for fast CPU tests."""
    cfg = dict(
        eeg_window_size=200,
        patch_size=200,
        in_chans=1,
        out_chans=8,
        num_classes=1,
        embed_dim=200,
        depth=2,
        num_heads=10,
        init_values=0.1,
        qkv_bias=True,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
    )
    cfg.update(overrides)
    return NeuralTransformer(**cfg)


class TestNeuralTransformerForward:
    def test_binary_head_shape(self):
        model = _make_tiny_model(num_classes=1)
        x = torch.randn(2, 4, 1, 200)
        out = model(x, channel_indices=None)
        assert out.shape == (2, 1)

    def test_multiclass_head_shape(self):
        model = _make_tiny_model(num_classes=5)
        x = torch.randn(2, 4, 1, 200)
        out = model(x, channel_indices=None)
        assert out.shape == (2, 5)

    def test_get_num_layers(self):
        model = _make_tiny_model(depth=4)
        assert model.get_num_layers() == 4

    def test_no_weight_decay_returns_expected_set(self):
        model = _make_tiny_model()
        nwd = model.no_weight_decay()
        assert 'pos_embed' in nwd
        assert 'cls_token' in nwd
        assert 'time_embed' in nwd

    def test_return_patch_tokens(self):
        model = _make_tiny_model(num_classes=1)
        x = torch.randn(2, 4, 1, 200)
        out = model(x, channel_indices=None, return_patch_tokens=True)
        # Token output, head not applied; should have a feature dim of embed_dim
        assert out.dim() >= 2

    def test_eeg_window_size_kwarg_propagates(self):
        # With patch_size=200 and eeg_window_size=400, time_window=2
        model = NeuralTransformer(
            eeg_window_size=400, patch_size=200, in_chans=1, out_chans=8,
            num_classes=1, embed_dim=200, depth=2, num_heads=10,
            init_values=0.1, qkv_bias=True, use_abs_pos_emb=False,
            use_rel_pos_bias=True,
        )
        assert model.time_window == 2

    def test_old_kwarg_raises_typeerror(self):
        # After the C3 rename cleanup, NeuralTransformer.__init__ no longer
        # swallows unknown kwargs. Passing the old EEG_size= name (or any
        # other typo) now raises TypeError at construction time, which is
        # what would have caught the test_finetune.py bug fixed in PR #4
        # before it ever shipped.
        with pytest.raises(TypeError):
            NeuralTransformer(
                EEG_size=999,
                patch_size=200, in_chans=1, out_chans=8, num_classes=1,
                embed_dim=200, depth=2, num_heads=10, init_values=0.1,
                qkv_bias=True, use_abs_pos_emb=False, use_rel_pos_bias=True,
            )


class TestNeuralTransformerGradients:
    def test_backward_propagates(self):
        model = _make_tiny_model(num_classes=1)
        x = torch.randn(2, 4, 1, 200, requires_grad=False)
        out = model(x, channel_indices=None)
        loss = out.sum()
        loss.backward()
        # At least one parameter should have a gradient
        assert any(p.grad is not None for p in model.parameters())


class TestBlockShape:
    def test_block_preserves_shape(self):
        block = Block(dim=200, num_heads=10, init_values=0.1, qkv_bias=True)
        x = torch.randn(2, 16, 200)  # (B, seq_len, dim)
        out = block(x)
        assert out.shape == x.shape


class TestEmbedInputs:
    """Pin the shared _embed_inputs helper extracted in this PR."""

    def test_shape_no_abs_pos_emb(self):
        # use_abs_pos_emb=False -> pos_embed is None, helper skips it.
        model = _make_tiny_model(num_classes=1, use_abs_pos_emb=False)
        x = torch.randn(2, 4, 1, 200)
        out = model._embed_inputs(x, channel_indices=None)
        # (B, N*A + 1 cls, embed_dim) = (2, 4*1+1, 200)
        assert out.shape == (2, 5, 200)

    def test_shape_with_channel_indices(self):
        model = _make_tiny_model(num_classes=1, use_abs_pos_emb=True)
        x = torch.randn(2, 4, 1, 200)
        channel_indices = [0, 1, 2, 3, 4]  # cls + 4 channels
        out = model._embed_inputs(x, channel_indices=channel_indices)
        assert out.shape == (2, 5, 200)

    def test_pos_embed_with_channel_indices_none_no_longer_broadcasts_to_128(self):
        # Regression for the latent pos_embed bug: previously, when
        # channel_indices=None and use_abs_pos_emb=True, the helper expanded
        # the full 128-channel pos_embed regardless of x's channel count.
        # This produced a tensor of shape (B, 128*time_window + 1, ...) that
        # was then added to (B, N*A + 1, ...) and crashed unless N happened to
        # be 128.
        model = _make_tiny_model(num_classes=1, use_abs_pos_emb=True)
        x = torch.randn(2, 4, 1, 200)  # 4 channels
        out = model._embed_inputs(x, channel_indices=None)
        # Must match the patch_embed + cls shape, not 128*time_window+1.
        assert out.shape == (2, 5, 200)

    def test_full_forward_with_pos_embed_and_no_channel_indices(self):
        # End-to-end version of the bug fix: forward(x, channel_indices=None)
        # with use_abs_pos_emb=True used to RuntimeError on the
        # `x = x + pos_embed` line. Should now succeed.
        model = _make_tiny_model(num_classes=1, use_abs_pos_emb=True)
        x = torch.randn(2, 4, 1, 200)
        out = model(x, channel_indices=None)
        assert out.shape == (2, 1)

