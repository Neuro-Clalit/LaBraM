"""Mixup / C-Mixup, LoRA, soft-label regression and EMA evaluation — the
anti-memorization options of the brain-age fine-tune."""

import torch
import torch.nn as nn
import pytest
from torch.utils.data import DataLoader, TensorDataset

from labram.configs.loss_config import LossConfig
from labram.configs.model_config import FinetuneModelConfig, LoRAConfig
from labram.configs.train_config import MixupConfig
from labram.layers.lora import LoRALinear, inject_lora, is_lora_param, mark_only_lora_trainable
from labram.losses.regression import (
    SoftLabelRegressionLoss, build_downstream_criterion, regression_output, soft_label_n_bins)
from labram.train.mixup import mixup_batch, mixup_partners
from labram.train.train_finetune import evaluate, train_one_epoch
from test_finetune import N_CHANNELS, T_PATCH, _make_epoch_args, _make_model


def _regression_loader(n=8, batch=4):
    y = torch.arange(n, dtype=torch.float32)
    X = torch.randn(n, N_CHANNELS, T_PATCH)
    return DataLoader(TensorDataset(X, y), batch_size=batch, drop_last=True)


# ----------------------------------------------------------------- mixup
class TestMixup:
    def test_mixes_inputs_and_targets_with_one_lambda(self):
        torch.manual_seed(0)
        x = torch.randn(6, 3, 10)
        y = torch.arange(6, dtype=torch.float32).unsqueeze(-1)
        mx, my, lam = mixup_batch(x, y, alpha=0.4)
        assert mx.shape == x.shape and my.shape == y.shape
        assert 0.0 <= lam <= 1.0
        # Each mixed target is a convex combination of two original targets.
        for i in range(6):
            diffs = (my[i] - (lam * y[i] + (1 - lam) * y)).abs().squeeze(-1)
            assert diffs.min() < 1e-5

    def test_alpha_zero_or_single_sample_is_identity(self):
        x, y = torch.randn(4, 2, 5), torch.randn(4, 1)
        mx, my, lam = mixup_batch(x, y, alpha=0.0)
        assert lam == 1.0 and torch.equal(mx, x) and torch.equal(my, y)
        mx, my, lam = mixup_batch(x[:1], y[:1], alpha=0.4)
        assert lam == 1.0 and torch.equal(mx, x[:1])

    def test_cmixup_prefers_close_targets_and_never_self(self):
        torch.manual_seed(0)
        y = torch.tensor([0.0, 0.1, 50.0, 50.1])
        partners = torch.stack([mixup_partners(y, sigma=1.0) for _ in range(200)])
        assert (partners != torch.arange(4)).all()
        # 0 <-> 1 and 2 <-> 3 are the only partners within a few sigma.
        assert (partners[:, 0] == 1).float().mean() > 0.99
        assert (partners[:, 2] == 3).float().mean() > 0.99

    def test_train_one_epoch_with_mixup_runs_and_skips_case_pooling(self):
        model = _make_model(num_classes=1)
        args = _make_epoch_args(model, _regression_loader(), nn.HuberLoss(), is_binary=False)
        args.update(task="regression", nb_classes=1, target_stats=(10.0, 5.0),
                    mixup_cfg=MixupConfig(enabled=True, alpha=0.4, sigma=2.0))
        stats = train_one_epoch(**args)
        assert "mae" in stats and stats["mae"] >= 0

    def test_mixup_rejected_for_classification(self):
        model = _make_model(num_classes=1)
        from test_finetune import _make_loader
        args = _make_epoch_args(model, _make_loader(), nn.BCEWithLogitsLoss(), is_binary=True)
        args.update(mixup_cfg=MixupConfig(enabled=True))
        with pytest.raises(ValueError, match="regression"):
            train_one_epoch(**args)


# ------------------------------------------------------------------ LoRA
class TestLoRA:
    def test_injected_model_is_unchanged_at_init_and_only_lora_trains(self):
        torch.manual_seed(0)
        model = _make_model(num_classes=1)
        x = torch.randn(2, N_CHANNELS, 1, T_PATCH)
        before = model(x).detach().clone()
        n = inject_lora(model, ["qkv", "proj", "fc1", "fc2"], rank=4, alpha=8.0)
        assert n == 2 * 4   # 2 blocks x 4 linears
        assert isinstance(model.blocks[0].attn.qkv, LoRALinear)
        assert torch.allclose(model(x), before, atol=1e-6)   # B = 0 -> identity

        n_trainable = mark_only_lora_trainable(model, ["head", "fc_norm"])
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        assert all(is_lora_param(n) or n.startswith(("head", "fc_norm")) for n in names)
        assert n_trainable == sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert not model.blocks[0].attn.qkv.weight.requires_grad

    def test_lora_changes_output_after_a_step_and_keeps_checkpoint_keys(self):
        torch.manual_seed(0)
        model = _make_model(num_classes=1)
        base_keys = set(model.state_dict())
        inject_lora(model, ["qkv"], rank=4, alpha=8.0)
        mark_only_lora_trainable(model, ["head"])
        keys = set(model.state_dict())
        assert base_keys <= keys
        assert all(is_lora_param(k) for k in keys - base_keys)
        x = torch.randn(2, N_CHANNELS, 1, T_PATCH)
        before = model(x).detach().clone()
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1.0)
        model(x).sum().backward()
        opt.step()
        assert not torch.allclose(model(x), before)

    def test_config_rejects_conflicts(self):
        with pytest.raises(ValueError):
            FinetuneModelConfig(lora=LoRAConfig(enabled=True), trainable_prefixes=["head"]).validate()
        with pytest.raises(ValueError):
            inject_lora(_make_model(), ["nope"], rank=4, alpha=8.0)


# ---------------------------------------------------------- soft labels
class TestSoftLabel:
    def test_bins_and_prediction_in_target_space(self):
        cfg = LossConfig(regression_loss="soft_label", soft_label_min=1, soft_label_max=89,
                         soft_label_bin_width=1.0, soft_label_sigma=2.5)
        assert soft_label_n_bins(cfg) == 89
        crit = build_downstream_criterion("regression", 1, cfg, target_stats=(49.0, 17.0))
        assert isinstance(crit, SoftLabelRegressionLoss) and crit.n_bins == 89
        # Peaked logits at the soft target of age 40 -> prediction ~ z(40), loss ~ 0.
        z = torch.tensor([[(40.0 - 49.0) / 17.0]])
        logits = torch.log(crit.soft_targets(z).clamp_min(1e-12))
        assert crit(logits, z).item() < 1e-5
        pred = regression_output(crit, logits)
        assert pred.shape == (1, 1)
        assert abs(pred.item() * 17.0 + 49.0 - 40.0) < 0.05
        # Wrong head width is caught.
        with pytest.raises(ValueError):
            crit(torch.zeros(1, 10), z)

    def test_scalar_criteria_pass_through(self):
        crit = build_downstream_criterion("regression", 1, LossConfig())
        out = torch.randn(3, 1)
        assert regression_output(crit, out) is out

    def test_train_and_evaluate_with_soft_label_head(self):
        cfg = LossConfig(regression_loss="soft_label", soft_label_min=0, soft_label_max=10,
                         soft_label_bin_width=1.0, soft_label_sigma=1.0)
        n_bins = soft_label_n_bins(cfg)
        model = _make_model(num_classes=n_bins)
        crit = build_downstream_criterion("regression", n_bins, cfg, target_stats=(5.0, 2.0))
        loader = _regression_loader()
        args = _make_epoch_args(model, loader, crit, is_binary=False)
        args.update(task="regression", nb_classes=n_bins, target_stats=(5.0, 2.0))
        stats = train_one_epoch(**args)
        assert stats["mae"] >= 0 and "soft_label_loss" not in stats  # term name is writer-only
        res = evaluate(data_loader=loader, model=model, device=torch.device("cpu"),
                       metrics=["mae", "r2"], is_binary=False, nb_classes=n_bins,
                       task="regression", target_stats=(5.0, 2.0), loss_cfg=cfg)
        assert res["mae"] >= 0 and "accuracy" not in res


# ----------------------------------------------------------- age balance
class TestAgeBalancedLoss:
    def _cfg(self, **kw):
        return LossConfig(regression_loss="huber", balance="sqrt_inverse",
                          balance_lds_sigma=0.0, **kw)

    def test_rare_ages_get_larger_weights_and_mean_is_one(self):
        from labram.losses.regression import age_balance_weights
        ages = [20.0] * 90 + [70.0] * 10
        low, width, w = age_balance_weights(ages, self._cfg())
        idx = ((torch.tensor(ages) - low) / width).floor().long()
        assert w[idx].mean().item() == pytest.approx(1.0, rel=1e-5)
        w20, w70 = w[int(20 - low)].item(), w[int(70 - low)].item()
        assert w70 / w20 == pytest.approx(3.0, rel=1e-4)   # sqrt(90 / 10)

    def test_inverse_is_clipped_at_max_weight(self):
        from labram.losses.regression import age_balance_weights
        ages = [20.0] * 999 + [70.0]
        _, _, w = age_balance_weights(ages, self._cfg(balance_max_weight=5.0).__class__(
            regression_loss="huber", balance="inverse", balance_lds_sigma=0.0,
            balance_max_weight=5.0))
        assert (w.max() / w.min()).item() == pytest.approx(5.0, rel=1e-4)

    def test_lds_smoothing_fills_empty_bins(self):
        from labram.losses.regression import age_balance_weights
        _, _, w = age_balance_weights([10.0, 10.0, 14.0], LossConfig(
            balance="inverse", balance_lds_sigma=2.0))
        assert torch.isfinite(w).all() and (w > 0).all()

    def test_criterion_weights_samples_in_years(self):
        from labram.losses.regression import BalancedRegressionLoss
        stats = (50.0, 10.0)
        ages = [20.0] * 90 + [70.0] * 10
        crit = build_downstream_criterion("regression", 1, self._cfg(), stats, train_targets=ages)
        assert isinstance(crit, BalancedRegressionLoss)
        z = lambda a: (torch.tensor([[a]]) - stats[0]) / stats[1]
        pred = torch.zeros(1, 1)
        plain = nn.HuberLoss()
        ratio_rare = crit(pred, z(70.0)) / plain(pred, z(70.0))
        ratio_common = crit(pred, z(20.0)) / plain(pred, z(20.0))
        assert (ratio_rare / ratio_common).item() == pytest.approx(3.0, rel=1e-4)

    def test_without_train_targets_or_none_mode_stays_unweighted(self):
        assert isinstance(build_downstream_criterion("regression", 1, self._cfg()), nn.HuberLoss)
        crit = build_downstream_criterion("regression", 1, LossConfig(), train_targets=[1.0, 2.0])
        assert isinstance(crit, nn.HuberLoss)

    def test_soft_label_and_unknown_mode_rejected(self):
        with pytest.raises(ValueError):
            build_downstream_criterion("regression", 89, LossConfig(
                regression_loss="soft_label", balance="inverse"), train_targets=[1.0, 2.0])
        from labram.losses.regression import age_balance_weights
        with pytest.raises(ValueError):
            age_balance_weights([1.0, 2.0], LossConfig(balance="cubic"))

    def test_trains_one_epoch(self):
        torch.manual_seed(0)
        model = _make_model(num_classes=1)
        crit = build_downstream_criterion("regression", 1, self._cfg(), (3.5, 2.0),
                                          train_targets=[float(i) for i in range(8)])
        args = _make_epoch_args(model, _regression_loader(), crit, is_binary=False)
        args["task"] = "regression"
        stats = train_one_epoch(**args)
        assert torch.isfinite(torch.tensor(stats["loss"]))
