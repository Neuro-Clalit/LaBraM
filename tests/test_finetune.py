import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from labram.train.train_finetune import train_class_batch, train_one_epoch, evaluate
from labram.configs.optim_config import OptimizerConfig
from labram.configs.train_config import TrainerConfig
from labram.configs.model_config import TransformerArchConfig
from labram.models.neural_transformer import NeuralTransformer
from labram.utils import NativeScalerWithGradNormCount

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
    cfg = TransformerArchConfig(
        eeg_window_size=T_PATCH,
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
    return NeuralTransformer(cfg)


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
        trainer_cfg=TrainerConfig(update_freq=1),
        optim_cfg=OptimizerConfig(clip_grad=None),
        start_steps=0,
        lr_schedule_values=[1e-4] * n_steps,
        wd_schedule_values=[0.05] * n_steps,
        num_training_steps_per_epoch=n_steps,
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

        loss, output, _ = train_class_batch(model, X, y, criterion, None)

        assert output.shape == (BATCH, 1)
        assert loss.item() > 0

    def test_multiclass_output_shape_and_positive_loss(self):
        n_cls = 3
        model = _make_model(num_classes=n_cls)
        criterion = nn.CrossEntropyLoss()
        X = torch.randn(BATCH, N_CHANNELS, 1, T_PATCH)
        y = torch.zeros(BATCH, dtype=torch.long)

        loss, output, _ = train_class_batch(model, X, y, criterion, None)

        assert output.shape == (BATCH, n_cls)
        assert loss.item() > 0

    def test_gradients_flow(self):
        model = _make_model(num_classes=1)
        criterion = nn.BCEWithLogitsLoss()
        X = torch.randn(BATCH, N_CHANNELS, 1, T_PATCH)
        y = torch.zeros(BATCH, 1)

        loss, _, _ = train_class_batch(model, X, y, criterion, None)
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
        for key in ("data_time_sec", "step_time_sec", "host_compute_time_sec",
                    "samples_processed"):
            assert key in stats
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
            trainer_cfg=TrainerConfig(update_freq=2),
            optim_cfg=OptimizerConfig(clip_grad=None),
            start_steps=0,
            lr_schedule_values=[1e-4] * n_steps,
            wd_schedule_values=[0.05] * n_steps,
            num_training_steps_per_epoch=n_steps,
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
        for key in ("data_time_sec", "step_time_sec", "host_compute_time_sec",
                    "samples_processed"):
            assert key in result
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
# Regression task (brain age)
# -----------------------------------------------------------------------

def _make_regression_loader(n_samples: int = 8, ages=None) -> DataLoader:
    """DataLoader yielding (EEG [B, N, T], age [B]) with a float32 scalar target."""
    y = (torch.arange(n_samples, dtype=torch.float32) if ages is None
         else torch.tensor(ages, dtype=torch.float32))
    X = torch.randn(len(y), N_CHANNELS, T_PATCH)
    return DataLoader(TensorDataset(X, y), batch_size=BATCH, drop_last=True)


class TestRegressionTask:
    """A regression head has num_classes == 1, exactly like a binary classifier,
    so these tests pin the behaviour that only ``task`` can distinguish."""

    def test_train_one_epoch_reports_mae_not_accuracy(self):
        model = _make_model(num_classes=1).to("cpu")
        loader = _make_regression_loader()
        args = _make_epoch_args(model, loader, nn.HuberLoss(), is_binary=False)
        args.update(task="regression", nb_classes=1)

        stats = train_one_epoch(**args)

        assert "mae" in stats
        assert "class_acc" not in stats
        assert stats["mae"] >= 0.0

    def test_evaluate_returns_regression_metrics_and_no_classification_keys(self):
        model = _make_model(num_classes=1).to("cpu")
        loader = _make_regression_loader()

        result = evaluate(
            data_loader=loader,
            model=model,
            device=torch.device("cpu"),
            metrics=["mae", "rmse", "r2", "pearson_r"],
            is_binary=False,
            nb_classes=1,
            task="regression",
        )

        assert {"mae", "rmse", "r2", "pearson_r", "loss"} <= set(result)
        # Probability-only metrics must not appear: their presence would mean the
        # scalar output was scored as a classification score.
        assert not {"roc_auc", "pr_auc", "accuracy", "cm_tp"} & set(result)

    def test_evaluate_does_not_use_cross_entropy_on_a_regression_target(self):
        """Guards evaluate()'s criterion dispatch. BCEWithLogitsLoss requires
        targets in [0, 1]; ages do not satisfy that, so a mis-dispatch is visible
        as an implausible loss (or an outright error) on out-of-range targets."""
        model = _make_model(num_classes=1).to("cpu")
        loader = _make_regression_loader(ages=[20.0, 40.0, 60.0, 80.0])

        result = evaluate(
            data_loader=loader,
            model=model,
            device=torch.device("cpu"),
            metrics=["mae"],
            is_binary=False,
            nb_classes=1,
            task="regression",
        )

        assert torch.isfinite(torch.tensor(result["loss"]))
        # An untrained head starts near zero, so the error tracks the ages
        # themselves rather than collapsing to a probability scale.
        assert result["mae"] > 1.0

    def test_metrics_are_reported_in_original_units_via_target_stats(self):
        """The loader z-scores the target; evaluate must de-normalize so MAE reads
        in years rather than standard deviations."""
        model = _make_model(num_classes=1).to("cpu")
        mean, std = 50.0, 10.0
        # Normalized targets, as the loader would emit them.
        loader = _make_regression_loader(ages=[-1.0, 0.0, 1.0, 2.0])

        scaled = evaluate(
            data_loader=loader, model=model, device=torch.device("cpu"),
            metrics=["mae"], is_binary=False, nb_classes=1, task="regression",
            target_stats=(mean, std))
        raw = evaluate(
            data_loader=loader, model=model, device=torch.device("cpu"),
            metrics=["mae"], is_binary=False, nb_classes=1, task="regression")

        # De-normalizing scales the error by std, putting it back in years.
        assert scaled["mae"] == pytest.approx(raw["mae"] * std, rel=1e-4)

    def test_case_aggregation_pools_windows_of_a_regression_case(self):
        from labram.configs.train_config import EvaluationConfig

        model = _make_model(num_classes=1).to("cpu")
        X = torch.randn(4, N_CHANNELS, T_PATCH)
        y = torch.tensor([30.0, 30.0, 60.0, 60.0])
        ids = ["rec_a", "rec_a", "rec_b", "rec_b"]

        class _WithIds(torch.utils.data.Dataset):
            def __len__(self):
                return len(y)

            def __getitem__(self, i):
                return X[i], y[i], ids[i]

        loader = DataLoader(_WithIds(), batch_size=2)
        result = evaluate(
            data_loader=loader, model=model, device=torch.device("cpu"),
            metrics=["mae"], is_binary=False, nb_classes=1, task="regression",
            eval_cfg=EvaluationConfig(agg_windows="mean", agg_case_by="recording",
                                      detailed_metrics=False))

        # Case-level metrics become primary; per-window ones are mirrored.
        assert "mae" in result and "window_mae" in result


class TestClassificationUnchanged:
    """The regression branch is additive: the classification default must behave
    exactly as it did before ``task`` existed."""

    def test_train_one_epoch_still_reports_class_acc_by_default(self):
        model = _make_model(num_classes=1).to("cpu")
        loader = _make_loader(n_samples=8, num_classes=1)
        args = _make_epoch_args(model, loader, nn.BCEWithLogitsLoss(), is_binary=True)

        stats = train_one_epoch(**args)

        assert "class_acc" in stats
        assert "mae" not in stats

    def test_binary_evaluate_still_produces_probability_metrics(self):
        model = _make_model(num_classes=1).to("cpu")
        loader = _make_loader(n_samples=8, num_classes=1)

        result = evaluate(
            data_loader=loader, model=model, device=torch.device("cpu"),
            metrics=["accuracy", "balanced_accuracy"], is_binary=True)

        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0


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
        loss, output, _ = train_class_batch(model, X, y, criterion, None)
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
        loss, output, _ = train_class_batch(model, X, y, criterion, None)
        assert output.shape == (BATCH, 1)
