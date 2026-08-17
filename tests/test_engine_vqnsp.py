"""Tests for engine_for_vqnsp.train_one_epoch + evaluate on a tiny VQNSP.

The VQ-NSP tokenizer training loop is the second-biggest untested surface in
the codebase. It runs VQNSP forward (which internally encodes -> quantizes ->
decodes amplitude + angle), computes a smooth-L1 reconstruction loss + an
embedding loss from the EMA codebook, and feeds the result through the AMP
scaler.

These tests build a tiny CPU-only VQNSP so a one-epoch run finishes in ~1
second, then assert:
- train_one_epoch returns the engine's metric keys (incl. `unused_code`
  from the codebook usage accounting),
- the codebook EMA cluster-size buffer actually advances after a step,
- evaluate (no-grad) returns matching metric keys,
- train_one_epoch is also wired up to advance the encoder parameters.
"""
import pytest
import torch

from labram.train.train_vqnsp import evaluate, train_one_epoch
from labram.configs.optim_config import OptimizerConfig
from labram.configs.train_config import OutputConfig
from labram.configs.model_config import QuantizerConfig, TransformerArchConfig, VQNSPArchConfig
from labram.models.vqnsp import VQNSP
from labram.utils import NativeScalerWithGradNormCount


N_CHANNELS = 4
PATCH_SIZE = 200
# 2 patches per channel: avoids the A == patch_size degenerate case in
# NeuralTransformer._embed_inputs (where the shape-role assignment
# `a if t == patch_size else t` collapses).
EEG_WINDOW_SIZE = 400
VOCAB_SIZE = 32
QUANTIZER_DIM = 8
EMBED_DIM = 200  # forced by TemporalConv output dim (25 patches * 8 out_chans)
DEPTH = 2
NUM_HEADS = 10
BATCH = 2


def _base_arch_cfg(**overrides) -> TransformerArchConfig:
    defaults = dict(
        eeg_window_size=EEG_WINDOW_SIZE, patch_size=PATCH_SIZE,
        in_chans=1, out_chans=8, num_classes=0, embed_dim=EMBED_DIM,
        depth=DEPTH, num_heads=NUM_HEADS, mlp_ratio=4.0, qkv_bias=True,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
        init_values=0.1, use_abs_pos_emb=True, use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False, use_mean_pooling=True, init_scale=0.001,
    )
    return TransformerArchConfig(**{**defaults, **overrides})


def _make_tiny_vqnsp():
    enc_cfg = _base_arch_cfg()
    dec_cfg = _base_arch_cfg(
        eeg_window_size=EEG_WINDOW_SIZE // PATCH_SIZE,
        patch_size=1,
        in_chans=QUANTIZER_DIM,
        depth=1,
    )
    arch_cfg = VQNSPArchConfig(
        encoder=enc_cfg,
        decoder=dec_cfg,
        quantizer=QuantizerConfig(
            num_codebook_tokens=VOCAB_SIZE,
            quantizer_dim=QUANTIZER_DIM,
            kmeans_init=False,
        ),
        decoder_out_dim=PATCH_SIZE,
    )
    return VQNSP(arch_cfg)


def _make_loader(n_samples: int = 4):
    """Yield raw (B, N, EEG_WINDOW_SIZE) tensors; VQNSP.forward rearranges to (B, N, A, T=200) internally."""
    x = torch.randn(n_samples, N_CHANNELS, EEG_WINDOW_SIZE) * 0.1
    dataset = _RawDataset(x)
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH, drop_last=True)


class _RawDataset(torch.utils.data.Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __len__(self):
        return self.tensor.size(0)

    def __getitem__(self, idx):
        return self.tensor[idx]


def _ch_names():
    return ['FP1', 'FP2', 'F3', 'F4']


class TestTrainOneEpoch:
    @pytest.fixture
    def trained(self, request):
        # pytest's `request` fixture is here so pytest is invoked
        # without a pytest namespace dependency at module scope.
        torch.manual_seed(0)
        model = _make_tiny_vqnsp()
        loader = _make_loader(n_samples=4)
        before = model.encoder.cls_token.detach().clone()
        cluster_size_before = model.quantizer.cluster_size.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_scaler = NativeScalerWithGradNormCount()
        stats = train_one_epoch(
            model=model,
            data_loader_list=[loader],
            optimizer=optimizer,
            device=torch.device('cpu'),
            epoch=0,
            loss_scaler=loss_scaler,
            optim_cfg=OptimizerConfig(clip_grad=None),
            output_cfg=OutputConfig(output_dir=''),
            start_steps=0,
            lr_schedule_values=None,
            ch_names_list=[_ch_names()],
        )
        return model, stats, before, cluster_size_before

    def test_returns_expected_keys(self, trained):
        _, stats, _, _ = trained
        # Engine logs loss, the per-key VQNSP loss_dict (quant/rec/rec_angle),
        # then the codebook-usage accounting at the end.
        for k in ('loss', 'lr', 'min_lr', 'grad_norm', 'unused_code',
                  'data_time_sec', 'step_time_sec', 'host_compute_time_sec',
                  'samples_processed'):
            assert k in stats, f"missing key {k} in {sorted(stats)}"

    def test_codebook_cluster_size_advances(self, trained):
        model, _, _, cluster_size_before = trained
        # The EMA cluster-size buffer is updated inside the quantizer's
        # forward() during training. After at least one step, at least one
        # of the per-token counts should be nonzero (since EMA seed starts
        # at zeros).
        cluster_size_after = model.quantizer.cluster_size.detach()
        assert (cluster_size_after.sum() - cluster_size_before.sum()).abs() > 0

    def test_loss_finite_and_positive(self, trained):
        _, stats, _, _ = trained
        assert stats['loss'] > 0
        # On tiny random data the VQNSP composite loss is bounded loosely.
        assert stats['loss'] < 50

    def test_optimizer_step_moves_encoder_params(self, trained):
        model, _, before, _ = trained
        after = model.encoder.cls_token.detach()
        assert not torch.equal(before, after), "encoder.cls_token did not change"


class TestEvaluate:
    def test_evaluate_returns_loss_and_unused_code(self):
        torch.manual_seed(1)
        model = _make_tiny_vqnsp()
        loader = _make_loader(n_samples=4)
        stats = evaluate(
            data_loader_list=[loader],
            model=model,
            device=torch.device('cpu'),
            log_writer=None,
            epoch=0,
            ch_names_list=[_ch_names()],
        )
        assert 'loss' in stats
        for key in ('data_time_sec', 'step_time_sec', 'host_compute_time_sec',
                    'samples_processed'):
            assert key in stats
        assert stats['loss'] > 0
        # unused_code is the count of zero-EMA entries; for a freshly-built
        # VQNSP with kmeans_init=False the count starts at the codebook size.
        assert 'unused_code' in stats
        assert 0 <= stats['unused_code'] <= VOCAB_SIZE

    def test_evaluate_does_not_train_params(self):
        torch.manual_seed(2)
        model = _make_tiny_vqnsp()
        loader = _make_loader(n_samples=4)
        before = {n: p.detach().clone() for n, p in model.encoder.named_parameters()}
        evaluate(
            data_loader_list=[loader],
            model=model,
            device=torch.device('cpu'),
            log_writer=None,
            epoch=0,
            ch_names_list=[_ch_names()],
        )
        # evaluate() is decorated with @torch.no_grad(); learnable encoder
        # parameters should be byte-identical afterwards.
        for n, p in model.encoder.named_parameters():
            assert torch.equal(before[n], p.detach()), f"encoder.{n} changed during evaluate()"

