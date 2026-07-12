"""Tests for runner_common helpers (extracted from the three runs)."""
import pytest
import torch
import torch.utils.data

import labram.runs.common as runner_common
import labram.utils
from labram.configs.optim_config import OptimizerConfig
from labram.configs.train_config import ClearMLConfig, DistributedConfig, OutputConfig, TrainerConfig


class _SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, n=20):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.zeros(3, 200), torch.tensor(i % 2, dtype=torch.long)


def _datasets(count: int, n: int = 20):
    return [_SyntheticDataset(n=n) for _ in range(count)]


class TestBuildDistributedTrainSamplerList:
    def test_returns_one_sampler_per_dataset(self):
        datasets = _datasets(3)
        samplers = runner_common.build_distributed_train_sampler_list(
            datasets, num_tasks=1, rank=0,
        )
        assert len(samplers) == 3
        for s in samplers:
            assert isinstance(s, torch.utils.data.DistributedSampler)


class TestBuildDistributedEvalSamplerList:
    def test_dist_eval_true_yields_distributed_samplers(self):
        samplers = runner_common.build_distributed_eval_sampler_list(
            _datasets(2), num_tasks=1, rank=0, dist_eval=True,
        )
        for s in samplers:
            assert isinstance(s, torch.utils.data.DistributedSampler)

    def test_dist_eval_false_yields_sequential_samplers(self):
        samplers = runner_common.build_distributed_eval_sampler_list(
            _datasets(2), num_tasks=1, rank=0, dist_eval=False,
        )
        for s in samplers:
            assert isinstance(s, torch.utils.data.SequentialSampler)


class TestBuildDataloaderList:
    def test_pairs_datasets_with_samplers(self):
        datasets = _datasets(3)
        samplers = runner_common.build_distributed_train_sampler_list(
            datasets, num_tasks=1, rank=0,
        )
        loaders = runner_common.build_dataloader_list(
            datasets, samplers,
            batch_size=4, num_workers=0, pin_memory=False, drop_last=True,
        )
        assert len(loaders) == 3
        for loader in loaders:
            assert isinstance(loader, torch.utils.data.DataLoader)
            assert loader.batch_size == 4
            assert loader.drop_last is True


class TestMakeSchedules:
    def _optim_cfg(self, **overrides):
        defaults = dict(lr=5e-4, min_lr=1e-6, warmup_epochs=1, warmup_steps=-1,
                        weight_decay=0.05, weight_decay_end=None)
        return OptimizerConfig(**{k: v for k, v in {**defaults, **overrides}.items()
                                  if OptimizerConfig.exist(k)})

    def _trainer_cfg(self, epochs=4):
        return TrainerConfig(epochs=epochs)

    def test_lr_schedule_length_matches_epoch_step_product(self):
        optim_cfg = self._optim_cfg()
        trainer_cfg = self._trainer_cfg(epochs=4)
        sched = runner_common.make_lr_schedule(optim_cfg, trainer_cfg, num_training_steps_per_epoch=10)
        assert len(sched) == 4 * 10

    def test_lr_schedule_starts_at_warmup_value_then_rises(self):
        optim_cfg = self._optim_cfg(lr=1e-3, warmup_epochs=1)
        trainer_cfg = self._trainer_cfg(epochs=4)
        sched = runner_common.make_lr_schedule(optim_cfg, trainer_cfg, num_training_steps_per_epoch=10)
        assert sched[0] == 0.0
        assert abs(sched[9] - 1e-3) < 1e-9
        assert sched[10] <= sched[9] + 1e-9

    def test_wd_schedule_constant_when_end_not_set(self):
        optim_cfg = self._optim_cfg(weight_decay=0.05, weight_decay_end=None, warmup_epochs=0)
        trainer_cfg = self._trainer_cfg(epochs=1)
        sched = runner_common.make_wd_schedule(optim_cfg, trainer_cfg, num_training_steps_per_epoch=5)
        assert all(abs(v - 0.05) < 1e-9 for v in sched)


class TestCreateLogWriter:
    def test_none_when_not_rank_zero(self, tmp_path):
        output_cfg = OutputConfig(log_dir=str(tmp_path))
        writer = runner_common.create_log_writer(output_cfg, global_rank=1)
        assert writer is None

    def test_none_when_log_dir_not_set(self):
        output_cfg = OutputConfig(log_dir='')
        writer = runner_common.create_log_writer(output_cfg, global_rank=0)
        assert writer is None

    def test_returns_writer_for_rank_zero_with_log_dir(self, tmp_path):
        output_cfg = OutputConfig(log_dir=str(tmp_path / "tb"))
        writer = runner_common.create_log_writer(output_cfg, global_rank=0)
        assert writer is not None
        assert (tmp_path / "tb").is_dir()

    def test_clearml_disabled_returns_tensorboard_only(self, tmp_path):
        output_cfg = OutputConfig(log_dir=str(tmp_path / "tb"))
        clearml_cfg = ClearMLConfig(enabled=False)
        writer = runner_common.create_log_writer(
            output_cfg, global_rank=0, clearml_cfg=clearml_cfg)
        # A single sink is returned bare (not wrapped in MultiWriter).
        from labram.utils import TensorboardLogger
        assert isinstance(writer, TensorboardLogger)

    def test_clearml_enabled_without_package_degrades_gracefully(self, tmp_path, monkeypatch):
        # Simulate clearml not being importable: init_clearml_task hits the
        # ImportError branch, returns None, and we fall back to TensorBoard.
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "clearml" or name.startswith("clearml."):
                raise ImportError("no clearml")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        output_cfg = OutputConfig(log_dir=str(tmp_path / "tb"))
        clearml_cfg = ClearMLConfig(enabled=True)
        writer = runner_common.create_log_writer(
            output_cfg, global_rank=0, clearml_cfg=clearml_cfg)
        from labram.utils import TensorboardLogger
        assert isinstance(writer, TensorboardLogger)

    def test_none_when_not_rank_zero_even_with_clearml(self, tmp_path):
        output_cfg = OutputConfig(log_dir=str(tmp_path / "tb"))
        clearml_cfg = ClearMLConfig(enabled=True)
        writer = runner_common.create_log_writer(
            output_cfg, global_rank=1, clearml_cfg=clearml_cfg)
        assert writer is None


class TestAppendLogLine:
    def test_no_op_when_output_dir_empty(self, tmp_path):
        output_cfg = OutputConfig(output_dir="")
        runner_common.append_log_line(output_cfg, {"epoch": 0, "loss": 0.5})
        assert not (tmp_path / "log.txt").exists()

    def test_appends_json_line(self, tmp_path):
        output_cfg = OutputConfig(output_dir=str(tmp_path))
        runner_common.append_log_line(output_cfg, {"epoch": 0, "loss": 0.5})
        runner_common.append_log_line(output_cfg, {"epoch": 1, "loss": 0.4})

        log_path = tmp_path / "log.txt"
        assert log_path.exists()
        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert '"epoch": 0' in lines[0]
        assert '"epoch": 1' in lines[1]


class TestPrintTrainingTime:
    def test_no_exception_and_logs_hhmmss(self, capsys):
        import logging
        import time
        labram.utils.configure_logging(level=logging.INFO)
        runner_common.print_training_time(time.time() - 65)
        captured = capsys.readouterr()
        assert "Training time " in captured.out
        assert ":01:" in captured.out


class TestWrapDistributed:
    def test_no_op_when_not_distributed(self):
        dist_cfg = DistributedConfig(distributed=False)
        model = torch.nn.Linear(4, 2)
        wrapped, without_ddp = runner_common.wrap_distributed(dist_cfg, model)
        assert wrapped is model
        assert without_ddp is model
