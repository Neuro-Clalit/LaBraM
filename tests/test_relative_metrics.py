"""Relative (scale-free) metric logging.

Covers both halves of ``LoggingConfig``'s relative options:

* ``relative_loss_components`` — per-component losses / gradient norms are
  reported as shares of the component total instead of raw magnitudes.
* ``relative_step_axis`` — metrics are plotted against normalized training
  progress instead of the raw global iteration.
"""

import pytest
import torch

import labram.runs.common as runner_common
from labram.configs.optim_config import OptimizerConfig
from labram.configs.run_configs import FinetuneRunConfig
from labram.configs.train_config import EvaluationConfig, LoggingConfig, TrainerConfig
from labram.losses import CodebookRegularizedCriterion, LossConfig
from labram.models.outputs import PredictorOutput
from labram.train.train_finetune import _log_eval_stats, train_one_epoch
from labram.utils import (
    ClearMLLogger,
    MultiWriter,
    NativeScalerWithGradNormCount,
    relative_components,
    relative_components_if_enabled,
)


class _RecordingWriter:
    """Minimal writer surface, records everything it is asked to log."""

    def __init__(self):
        self.updates = []  # (head, step, kwargs)
        self.step = 0

    def set_step(self, step=None):
        self.step = self.step + 1 if step is None else step

    def update(self, head="scalar", step=None, **kwargs):
        self.updates.append((head, self.step if step is None else step, kwargs))

    def keys(self, head=None):
        return {k for h, _, kw in self.updates if head in (None, h) for k in kw}


class _FakeClearMLLogger:
    def __init__(self):
        self.scalars = []  # (title, series, value, iteration)

    def report_scalar(self, title, series, value, iteration):
        self.scalars.append((title, series, value, iteration))


# ------------------------------------------------------------------
# relative_components
# ------------------------------------------------------------------

class TestRelativeComponents:
    def test_components_become_shares_that_sum_to_one(self):
        rel = relative_components({"a_loss": 1.0, "b_loss": 3.0})
        assert rel == {"a_loss_rel": 0.25, "b_loss_rel": 0.75}
        assert sum(rel.values()) == pytest.approx(1.0)

    def test_shares_are_invariant_to_the_overall_scale(self):
        small = relative_components({"a": 1e-6, "b": 3e-6})
        large = relative_components({"a": 1e3, "b": 3e3})
        assert small == pytest.approx(large)

    def test_tensor_values_are_reduced_to_floats(self):
        rel = relative_components({"a": torch.tensor([1.0, 1.0]), "b": torch.tensor(2.0)})
        assert rel["a_rel"] == pytest.approx(1 / 3)
        assert rel["b_rel"] == pytest.approx(2 / 3)

    def test_zero_and_none_are_handled_without_nan(self):
        rel = relative_components({"a": 0.0, "b": 0.0, "c": None})
        assert rel == {"a_rel": 0.0, "b_rel": 0.0}

    def test_negative_components_use_the_absolute_total(self):
        rel = relative_components({"a": -1.0, "b": 3.0})
        assert rel["a_rel"] == pytest.approx(-0.25)
        assert rel["b_rel"] == pytest.approx(0.75)

    def test_gated_helper_is_a_no_op_when_disabled(self):
        values = {"a_loss": 1.0, "b_loss": 3.0}
        off = LoggingConfig(relative_loss_components=False)
        assert relative_components_if_enabled(values, off) == values
        assert relative_components_if_enabled(values, None) == values
        on = LoggingConfig(relative_loss_components=True)
        assert relative_components_if_enabled(values, on) == {
            "a_loss_rel": 0.25, "b_loss_rel": 0.75}

    def test_gated_helper_passes_empty_input_through(self):
        cfg = LoggingConfig(relative_loss_components=True)
        assert relative_components_if_enabled(None, cfg) is None
        assert relative_components_if_enabled({}, cfg) == {}


# ------------------------------------------------------------------
# Relative x-axis on the writers
# ------------------------------------------------------------------

class TestRelativeStepAxis:
    def _writer(self):
        return ClearMLLogger(clearml_logger=_FakeClearMLLogger())

    def test_absolute_steps_are_passed_through_until_configured(self):
        w = self._writer()
        w.set_step(37)
        assert w.step == 37
        w.set_step()
        assert w.step == 38
        assert w.epoch_step(4) == 4

    def test_steps_map_onto_the_progress_axis(self):
        w = self._writer()
        assert w.configure_relative_steps(total_steps=200, total_epochs=10, scale=1000)
        w.set_step(0)
        assert w.step == 0
        w.set_step(100)   # halfway through the run
        assert w.step == 500
        w.set_step(200)
        assert w.step == 1000

    def test_increments_track_absolute_steps_not_mapped_ones(self):
        w = self._writer()
        w.configure_relative_steps(total_steps=100, total_epochs=10, scale=1000)
        w.set_step(50)
        w.set_step()      # absolute 51 -> 510, not 500 + 1
        assert w.step == 510

    def test_steps_beyond_the_planned_run_clamp_to_full_progress(self):
        w = self._writer()
        w.configure_relative_steps(total_steps=100, total_epochs=10, scale=1000)
        w.set_step(250)
        assert w.step == 1000

    def test_epoch_steps_land_on_the_same_axis(self):
        w = self._writer()
        w.configure_relative_steps(total_steps=200, total_epochs=10, scale=1000)
        # Epoch e reports the state *after* the epoch -> progress (e + 1) / E,
        # which is exactly where the iteration counter is at that moment.
        w.set_step(20)
        assert w.epoch_step(0) == w.step == 100
        assert w.epoch_step(9) == 1000

    def test_non_positive_totals_keep_the_absolute_axis(self):
        w = self._writer()
        assert not w.configure_relative_steps(total_steps=0, total_epochs=10)
        w.set_step(7)
        assert w.step == 7

    def test_reported_iteration_uses_the_relative_step(self):
        fake = _FakeClearMLLogger()
        w = ClearMLLogger(clearml_logger=fake)
        w.configure_relative_steps(total_steps=200, total_epochs=10, scale=1000)
        w.set_step(50)
        w.update(loss=0.5, head="loss")
        assert fake.scalars == [("loss", "loss", 0.5, 250)]


class TestMultiWriterRelativeSteps:
    def _multi(self):
        a, b = _FakeClearMLLogger(), _FakeClearMLLogger()
        return MultiWriter([ClearMLLogger(clearml_logger=a),
                            ClearMLLogger(clearml_logger=b)]), a, b

    def test_configuration_fans_out_to_every_child(self):
        multi, a, b = self._multi()
        assert multi.configure_relative_steps(total_steps=100, total_epochs=5, scale=1000)
        multi.set_step(25)
        multi.update(loss=1.0, head="loss")
        assert multi.step == 250
        assert a.scalars[-1][-1] == 250
        assert b.scalars[-1][-1] == 250

    def test_children_map_the_step_exactly_once(self):
        multi, a, _ = self._multi()
        multi.configure_relative_steps(total_steps=100, total_epochs=5, scale=1000)
        multi.set_step(50)
        multi.set_step()  # absolute 51 for every writer
        multi.update(loss=1.0, head="loss")
        assert multi.step == 510
        assert a.scalars[-1][-1] == 510


class TestConfigureRelativeStepAxis:
    def _config(self, **logging_kwargs):
        config = FinetuneRunConfig()
        config.trainer = TrainerConfig(epochs=10, update_freq=2)
        config.logging = LoggingConfig(**logging_kwargs)
        return config

    def test_enabled_config_puts_the_writer_on_the_relative_axis(self):
        w = ClearMLLogger(clearml_logger=_FakeClearMLLogger())
        config = self._config(relative_step_axis=True, relative_step_scale=1000)
        assert runner_common.configure_relative_step_axis(
            w, config, num_training_steps_per_epoch=5, steps_per_logged_step=2)
        # 10 epochs * 5 steps * update_freq 2 = 100 logging steps.
        w.set_step(50)
        assert w.step == 500
        assert w.epoch_step(4) == 500

    def test_disabled_config_keeps_absolute_steps(self):
        w = ClearMLLogger(clearml_logger=_FakeClearMLLogger())
        config = self._config(relative_step_axis=False)
        assert not runner_common.configure_relative_step_axis(
            w, config, num_training_steps_per_epoch=5)
        w.set_step(50)
        assert w.step == 50

    def test_missing_writer_or_logging_config_is_a_no_op(self):
        config = self._config(relative_step_axis=True)
        assert not runner_common.configure_relative_step_axis(None, config, 5)
        assert not runner_common.configure_relative_step_axis(
            ClearMLLogger(clearml_logger=_FakeClearMLLogger()), object(), 5)


class TestEvalStatsOnRelativeAxis:
    def test_epoch_metrics_follow_the_writer_axis(self):
        w = ClearMLLogger(clearml_logger=_FakeClearMLLogger())
        w.configure_relative_steps(total_steps=100, total_epochs=5, scale=1000)
        fake = w._logger
        _log_eval_stats(w, {"accuracy": 0.9, "cm_tp": 12}, head="val", epoch=1)
        # Epoch 1 of 5 -> 40% through the run, on both the rate and count plots.
        assert {(title, iteration) for title, _, _, iteration in fake.scalars} == {
            ("val", 400), ("val_cm", 400)}

    def test_absolute_axis_still_logs_the_raw_epoch(self):
        w = _RecordingWriter()  # no epoch_step -> untouched
        _log_eval_stats(w, {"accuracy": 0.9}, head="val", epoch=3)
        assert w.updates == [("val", 3, {"accuracy": 0.9})]


# ------------------------------------------------------------------
# Fine-tune loop integration
# ------------------------------------------------------------------

class _TinyRegModel(torch.nn.Module):
    """Model emitting every codebook-regularized loss component."""

    def __init__(self, n_classes=3):
        super().__init__()
        self.head = torch.nn.Linear(200, n_classes)
        self.dec = torch.nn.Linear(200, 200)

    def forward(self, samples, channel_indices=None, classify_only=False, **kw):
        b = samples.shape[0]
        flat = samples.reshape(b, -1)[:, :200]
        logits = self.head(flat)
        if classify_only:
            return PredictorOutput(logits=logits)
        recon = self.dec(flat).reshape(b, 1, 1, 200).expand_as(samples)
        return PredictorOutput(
            logits=logits, recon_magnitude=recon, recon_phase=recon,
            quantize_loss=(flat ** 2).mean(), x_patched=samples,
        )


def _run_finetune_epoch(logging_cfg, writer):
    torch.manual_seed(0)
    xs, ys = torch.randn(4, 1, 200), torch.arange(4) % 3
    loader = [(xs[i:i + 2], ys[i:i + 2]) for i in range(0, 4, 2)]
    model = _TinyRegModel()
    return train_one_epoch(
        model, CodebookRegularizedCriterion(torch.nn.CrossEntropyLoss(), LossConfig()),
        loader, torch.optim.SGD(model.parameters(), lr=1e-3), torch.device("cpu"),
        epoch=0, loss_scaler=NativeScalerWithGradNormCount(),
        trainer_cfg=TrainerConfig(update_freq=1), optim_cfg=OptimizerConfig(),
        num_training_steps_per_epoch=2, log_writer=writer, start_steps=0,
        is_binary=False, nb_classes=3,
        eval_cfg=EvaluationConfig(log_grad_components=True, log_grad_freq=1,
                                  detailed_metrics=False),
        logging_cfg=logging_cfg,
    )


class TestFinetuneLoopComponentLogging:
    def test_components_are_logged_as_shares(self):
        writer = _RecordingWriter()
        stats = _run_finetune_epoch(LoggingConfig(relative_loss_components=True), writer)

        loss_keys = writer.keys(head="loss")
        for term in ("classifier", "magnitude", "phase", "quantize"):
            assert f"{term}_loss_rel" in loss_keys
            assert f"{term}_loss" not in loss_keys      # absolute series replaced
            assert f"grad_norm_{term}_rel" in writer.keys(head="grad")
        # The aggregate loss keeps its absolute value.
        assert "loss" in loss_keys

        shares = [v for _, _, kw in writer.updates for k, v in kw.items()
                  if k.endswith("_loss_rel")]
        assert shares and all(0.0 <= s <= 1.0 for s in shares)
        # Console/summary stats stay absolute so log.txt keeps raw magnitudes.
        assert "classifier_loss" in stats and "classifier_loss_rel" not in stats

    def test_disabled_flag_restores_absolute_component_series(self):
        writer = _RecordingWriter()
        _run_finetune_epoch(LoggingConfig(relative_loss_components=False), writer)
        loss_keys = writer.keys(head="loss")
        for term in ("classifier", "magnitude", "phase", "quantize"):
            assert f"{term}_loss" in loss_keys
            assert f"{term}_loss_rel" not in loss_keys


class TestVQNSPWriterLossValues:
    """The VQNSP loss dict mixes components with aggregates and counters."""

    def _values(self):
        return {"loss": 4.0, "total_loss": 4.0, "unused_code": 17,
                "rec_loss": 1.0, "rec_angle_loss": 2.0, "quant_loss": 1.0}

    def test_only_components_become_shares(self):
        from labram.train.train_vqnsp import _writer_loss_values

        out = _writer_loss_values(self._values(), LoggingConfig(relative_loss_components=True))
        assert out == {
            "loss": 4.0, "total_loss": 4.0, "unused_code": 17,
            "rec_loss_rel": 0.25, "rec_angle_loss_rel": 0.5, "quant_loss_rel": 0.25,
        }

    def test_disabled_flag_leaves_the_dict_untouched(self):
        from labram.train.train_vqnsp import _writer_loss_values

        values = self._values()
        assert _writer_loss_values(values, LoggingConfig(relative_loss_components=False)) == values
        assert _writer_loss_values(values, None) == values
