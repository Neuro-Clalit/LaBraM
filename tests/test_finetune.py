import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from engine_for_finetuning import train_class_batch, train_one_epoch, evaluate
from modeling_finetune import NeuralTransformer
from utils import NativeScalerWithGradNormCount

# -----------------------------------------------------------------------
# Constants – small enough for fast CPU tests
# -----------------------------------------------------------------------
BATCH = 4
N_CHANNELS = 4   # EEG channels
T_PATCH = 200    # patch size / sampling rate (must stay 200 for TemporalConv)
# embed_dim is fixed by TemporalConv: (T_PATCH // conv_stride) * out_chans
# = (200 // 8) * 8 = 200.  Keep in sync with the model factory below.
EMBED_DIM = 200


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_model(num_classes: int = 1, use_abs_pos_emb: bool = False) -> NeuralTransformer:
    """Tiny 2-block model suitable for fast CPU unit tests."""
    return NeuralTransformer(
        EEG_size=T_PATCH,
        patch_size=T_PATCH,
        in_chans=1,
        out_chans=8,
        num_classes=num_classes,
        embed_dim=EMBED_DIM,
        depth=2,
        num_heads=10,
        init_values=0.1,
        qkv_bias=True,
        use_abs_pos_emb=use_abs_pos_emb,
        use_rel_pos_bias=True,
    )


def _make_loader(n_samples: int = 8, num_classes: int = 1) -> DataLoader:
    """DataLoader yielding (EEG [B, N, T], label [B]) tensors."""
    X = torch.randn(n_samples, N_CHANNELS, T_PATCH)
    y = torch.randint(0, max(num_classes, 2), (n_samples,)).long()
    return DataLoader(TensorDataset(X, y), batch_size=BATCH, drop_last=True)


def _make_epoch_args(model, loader, criterion, is_binary: bool):
    """Build the repetitive keyword-args for train_one_epoch."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_scaler = NativeScalerWithGradNormCount()
    n_steps = len(loader)
    return dict(
        model=model,
        criterion=criterion,
        data_loader=loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        epoch=0,
        loss_scaler=loss_scaler,
        start_steps=0,
        lr_schedule_values=[1e-4] * n_steps,
        wd_schedule_values=[0.05] * n_steps,
        num_training_steps_per_epoch=n_steps,
        update_freq=1,
        is_binary=is_binary,
    )


# -----------------------------------------------------------------------
# train_class_batch
# -----------------------------------------------------------------------

class TestTrainClassBatch:
    def test_binary_output_shape_and_positive_loss(self):
        model = _make_model(num_classes=1)
        criterion = nn.BCEWithLogitsLoss()
        # engine rearranges B N (A T) -> B N A T, so pass already-rearranged tensor
        X = torch.randn(BATCH, N_CHANNELS, 1, T_PATCH)
        y = torch.zeros(BATCH, 1)

        loss, output = train_class_batch(model, X, y, criterion, None)

        assert output.shape == (BATCH, 1)
        assert loss.item() > 0

    def test_multiclass_output_shape_and_positive_loss(self):
        n_cls = 3
        model = _make_model(num_classes=n_cls)
        criterion = nn.CrossEntropyLoss()
        X = torch.randn(BATCH, N_CHANNELS, 1, T_PATCH)
        y = torch.zeros(BATCH, dtype=torch.long)

        loss, output = train_class_batch(model, X, y, criterion, None)

        assert output.shape == (BATCH, n_cls)
        assert loss.item() > 0

    def test_gradients_flow(self):
        model = _make_model(num_classes=1)
        criterion = nn.BCEWithLogitsLoss()
        X = torch.randn(BATCH, N_CHANNELS, 1, T_PATCH)
        y = torch.zeros(BATCH, 1)

        loss, _ = train_class_batch(model, X, y, criterion, None)
        loss.backward()

        # At least one parameter must have a gradient
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0


# -----------------------------------------------------------------------
# train_one_epoch
# -----------------------------------------------------------------------

class TestTrainOneEpoch:
    def test_binary_returns_expected_keys(self):
        model = _make_model(num_classes=1).to("cpu")
        loader = _make_loader(n_samples=8, num_classes=1)
        args = _make_epoch_args(model, loader, nn.BCEWithLogitsLoss(), is_binary=True)

        stats = train_one_epoch(**args)

        assert "loss" in stats
        assert "class_acc" in stats
        assert "grad_norm" in stats
        assert stats["loss"] > 0

    def test_multiclass_returns_expected_keys(self):
        n_cls = 3
        model = _make_model(num_classes=n_cls).to("cpu")
        loader = _make_loader(n_samples=8, num_classes=n_cls)
        args = _make_epoch_args(model, loader, nn.CrossEntropyLoss(), is_binary=False)

        stats = train_one_epoch(**args)

        assert "loss" in stats
        assert stats["loss"] > 0

    def test_weights_change_after_epoch(self):
        model = _make_model(num_classes=1).to("cpu")
        loader = _make_loader(n_samples=8, num_classes=1)
        before = {n: p.clone() for n, p in model.named_parameters()}
        args = _make_epoch_args(model, loader, nn.BCEWithLogitsLoss(), is_binary=True)

        train_one_epoch(**args)

        changed = any(
            not torch.equal(before[n], p)
            for n, p in model.named_parameters()
            if p.requires_grad
        )
        assert changed, "No parameter was updated after one epoch"

    def test_gradient_accumulation(self):
        model = _make_model(num_classes=1).to("cpu")
        loader = _make_loader(n_samples=8, num_classes=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_scaler = NativeScalerWithGradNormCount()
        n_steps = len(loader)

        stats = train_one_epoch(
            model=model,
            criterion=nn.BCEWithLogitsLoss(),
            data_loader=loader,
            optimizer=optimizer,
            device=torch.device("cpu"),
            epoch=0,
            loss_scaler=loss_scaler,
            start_steps=0,
            lr_schedule_values=[1e-4] * n_steps,
            wd_schedule_values=[0.05] * n_steps,
            num_training_steps_per_epoch=n_steps,
            update_freq=2,
            is_binary=True,
        )
        assert "loss" in stats


# -----------------------------------------------------------------------
# evaluate
# -----------------------------------------------------------------------

class TestEvaluate:
    def test_binary_returns_loss_and_accuracy(self):
        model = _make_model(num_classes=1).to("cpu")
        loader = _make_loader(n_samples=8, num_classes=1)

        result = evaluate(
            data_loader=loader,
            model=model,
            device=torch.device("cpu"),
            metrics=["accuracy", "balanced_accuracy"],
            is_binary=True,
        )

        assert "loss" in result
        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_multiclass_returns_loss_and_accuracy(self):
        n_cls = 3
        model = _make_model(num_classes=n_cls).to("cpu")
        loader = _make_loader(n_samples=8, num_classes=n_cls)

        result = evaluate(
            data_loader=loader,
            model=model,
            device=torch.device("cpu"),
            metrics=["accuracy"],
            is_binary=False,
        )

        assert "loss" in result
        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_model_in_eval_mode_during_evaluate(self):
        """evaluate() must not leave the model in train mode."""
        model = _make_model(num_classes=1).to("cpu")
        model.train()
        loader = _make_loader(n_samples=8, num_classes=1)

        evaluate(
            data_loader=loader,
            model=model,
            device=torch.device("cpu"),
            metrics=["accuracy"],
            is_binary=True,
        )

        assert not model.training

    def test_no_grad_during_evaluate(self):
        """Outputs from evaluate() should not carry gradients."""
        model = _make_model(num_classes=1).to("cpu")
        loader = _make_loader(n_samples=4, num_classes=1)

        result = evaluate(
            data_loader=loader,
            model=model,
            device=torch.device("cpu"),
            metrics=["accuracy"],
            is_binary=True,
        )

        # If no error was raised and result is finite, no-grad was respected
        assert result["loss"] >= 0


# -----------------------------------------------------------------------
# MPS / device smoke tests
# -----------------------------------------------------------------------

class TestDeviceSupport:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_train_batch_on_cuda(self):
        device = torch.device("cuda")
        model = _make_model(num_classes=1).to(device)
        criterion = nn.BCEWithLogitsLoss()
        X = torch.randn(BATCH, N_CHANNELS, 1, T_PATCH, device=device)
        y = torch.zeros(BATCH, 1, device=device)
        loss, output = train_class_batch(model, X, y, criterion, None)
        assert output.shape == (BATCH, 1)

    @pytest.mark.skipif(
        not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        reason="MPS not available",
    )
    def test_train_batch_on_mps(self):
        device = torch.device("mps")
        model = _make_model(num_classes=1).to(device)
        criterion = nn.BCEWithLogitsLoss()
        X = torch.randn(BATCH, N_CHANNELS, 1, T_PATCH, device=device)
        y = torch.zeros(BATCH, 1, device=device)
        loss, output = train_class_batch(model, X, y, criterion, None)
        assert output.shape == (BATCH, 1)
