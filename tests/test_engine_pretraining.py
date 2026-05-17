"""Tests for engine_for_pretraining.train_one_epoch on tiny synthetic models.

The pretraining loop is the biggest untested surface in the codebase. It runs
the masked-EEG student (NeuralTransformerForMEM) against a frozen VQNSP
tokenizer, computes a masked-recovery cross-entropy + its symmetric variant,
and feeds the result through the AMP scaler.

These tests construct tiny CPU-only versions of both models so a one-epoch
run finishes in ~1 second, then assert that:
- train_one_epoch returns a dict with the engine's expected metric keys,
- the loss is finite and positive (CE on a 32-token vocab with random init
  starts around ln(32) ~= 3.4),
- the optimizer step actually moves model parameters.
"""
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from engine_for_pretraining import random_masking, train_one_epoch
from modeling_pretrain import NeuralTransformerForMEM
from modeling_vqnsp import VQNSP
from utils import NativeScalerWithGradNormCount


N_CHANNELS = 4
PATCH_SIZE = 200          # engine hardcodes `rearrange(..., T=200)`
# 2 patches per channel: must use EEG_WINDOW_SIZE > PATCH_SIZE so that the
# 4-dim shape unpacking `B, N, A, T = x.shape` inside
# NeuralTransformer._embed_inputs is unambiguous. With A == patch_size,
# the helper's `a if t == self.patch_size else t` branch collapses and
# pos_embed broadcasts on the wrong axis.
EEG_WINDOW_SIZE = 400
VOCAB_SIZE = 32           # also the codebook size
QUANTIZER_DIM = 8
# TemporalConv's output feature dim is (PATCH_SIZE // 8) * out_chans
# = 25 * 8 = 200 by construction, so the encoder's embed_dim must be 200
# for the cls-concat shapes to line up. This matches every production
# factory in the codebase.
EMBED_DIM = 200
DEPTH = 2
NUM_HEADS = 10
BATCH = 2


def _base_transformer_config():
    """Common kwargs for the encoder/decoder NeuralTransformers inside VQNSP."""
    return dict(
        eeg_window_size=EEG_WINDOW_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=1,
        out_chans=8,
        num_classes=0,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.1,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
        use_mean_pooling=True,
        init_scale=0.001,
    )


def _make_tiny_vqnsp():
    """Tiny VQNSP whose encoder/decoder are 2-block NeuralTransformers."""
    encoder_config = _base_transformer_config()
    decoder_config = _base_transformer_config()
    # Match the production factory's decoder reshape pattern:
    #   decoder.eeg_window_size = encoder.eeg_window_size // decoder.patch_size
    # then decoder.patch_size = 1, decoder.in_chans = quantizer_dim.
    decoder_config['eeg_window_size'] = EEG_WINDOW_SIZE // PATCH_SIZE
    decoder_config['patch_size'] = 1
    decoder_config['in_chans'] = QUANTIZER_DIM
    decoder_config['depth'] = 1

    return VQNSP(
        encoder_config,
        decoder_config,
        num_codebook_tokens=VOCAB_SIZE,
        quantizer_dim=QUANTIZER_DIM,
        decoder_out_dim=PATCH_SIZE,
        quantize_kmeans_init=False,
    )


def _make_tiny_mem():
    """Tiny NeuralTransformerForMEM matching the engine's expected interface."""
    return NeuralTransformerForMEM(
        eeg_window_size=EEG_WINDOW_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=1,
        out_chans=8,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.1,
        use_abs_pos_emb=True,
    )


def _make_loader(n_samples: int = 4):
    """Returns a DataLoader yielding (B, N, EEG_WINDOW_SIZE) tensors.

    The engine rearranges to (B, N, A=EEG_WINDOW_SIZE/PATCH_SIZE, T=PATCH_SIZE).
    """
    x = torch.randn(n_samples, N_CHANNELS, EEG_WINDOW_SIZE) * 0.1  # / 100 happens in the engine
    dataset = _RawDataset(x)
    return DataLoader(dataset, batch_size=BATCH, drop_last=True)


class _RawDataset(torch.utils.data.Dataset):
    """Returns the raw sample tensor (no label), matching the pretrain
    dataloader contract."""

    def __init__(self, tensor):
        self.tensor = tensor

    def __len__(self):
        return self.tensor.size(0)

    def __getitem__(self, idx):
        return self.tensor[idx]


def _ch_names():
    """Channel names that resolve via utils.get_channel_indices."""
    return ['FP1', 'FP2', 'F3', 'F4']


def _make_args(distributed: bool = False, gradient_accumulation_steps: int = 1, clip_grad: float = None):
    return SimpleNamespace(
        distributed=distributed,
        gradient_accumulation_steps=gradient_accumulation_steps,
        clip_grad=clip_grad,
    )


class TestTrainOneEpochSmoke:
    @pytest.fixture
    def trained(self):
        torch.manual_seed(0)
        student = _make_tiny_mem()
        vqnsp = _make_tiny_vqnsp().eval()
        loader = _make_loader(n_samples=4)
        # Snapshot a couple of trainable params before training so we can
        # later confirm they actually moved. lm_head.weight is initialized
        # by trunc_normal in the model __init__ and is touched by every
        # gradient step through the masked-prediction head.
        before = student.lm_head.weight.detach().clone()
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
        loss_scaler = NativeScalerWithGradNormCount()
        stats = train_one_epoch(
            model=student,
            vqnsp=vqnsp,
            data_loader_list=[loader],
            optimizer=optimizer,
            device=torch.device('cpu'),
            epoch=0,
            loss_scaler=loss_scaler,
            max_norm=None,
            start_steps=0,
            lr_schedule_values=None,
            wd_schedule_values=None,
            ch_names_list=[_ch_names()],
            args=_make_args(),
        )
        return student, stats, before

    def test_returns_expected_keys(self, trained):
        _, stats, _ = trained
        for k in ('mlm_acc', 'mlm_acc_sym', 'loss_rec', 'loss', 'lr', 'min_lr'):
            assert k in stats, f"missing key {k} in {list(stats)}"

    def test_loss_finite_and_positive(self, trained):
        _, stats, _ = trained
        assert stats['loss'] > 0
        # CrossEntropy on a small-vocab masked-prediction task with random
        # init starts above ln(VOCAB_SIZE) ~= 3.46; the engine reports
        # `loss = loss_rec + loss_rec_sym` so bound loosely.
        assert stats['loss'] < 50

    def test_optimizer_step_actually_moves_params(self, trained):
        student, _, before = trained
        # After the AMP scaler step, optimizer.zero_grad() clears grads
        # (set_to_none=True by default in modern torch), so a `p.grad is
        # not None` assertion is brittle. Compare param values instead:
        # at least one element of lm_head.weight must have changed.
        after = student.lm_head.weight.detach()
        assert not torch.equal(before, after), "lm_head.weight did not change"


class TestRandomMasking:
    """Unit tests for the random_masking helper (also covered indirectly above)."""

    def test_mask_ratio_zero_keeps_everything(self):
        x = torch.randn(2, 10, 4)
        mask = random_masking(x, mask_ratio=0.0)
        # mask=True means "remove" -> 0 removes => mask is all False.
        assert mask.dtype == torch.bool
        assert mask.shape == (2, 10)
        assert mask.sum().item() == 0

    def test_mask_ratio_half_masks_about_half(self):
        x = torch.randn(2, 10, 4)
        mask = random_masking(x, mask_ratio=0.5)
        # Exactly 50% removed (per-sample shuffle hits `len_keep = L*(1-r)`).
        assert mask.sum(dim=1).tolist() == [5, 5]

    def test_per_sample_independent(self):
        torch.manual_seed(0)
        x = torch.randn(4, 16, 8)
        m1 = random_masking(x, mask_ratio=0.5)
        # Different rows should generally NOT have identical masks.
        rows = [tuple(m1[i].tolist()) for i in range(4)]
        assert len(set(rows)) > 1, "all per-sample masks are identical"
