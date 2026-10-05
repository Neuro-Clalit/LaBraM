"""Per-case train metrics, the shared ``err`` plot, per-term loss logging and the
best-vs-last summary tables."""
from collections import defaultdict

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from labram.configs.train_config import EvaluationConfig
from labram.runs.common import build_summary_tables, log_summary_tables
from labram.train.train_finetune import _log_split_errors, _loss_term_name, train_one_epoch
from labram.utils.logging import MultiWriter, TensorboardLogger
from test_finetune import N_CHANNELS, T_PATCH, _make_epoch_args, _make_model


class _RecordingWriter:
    def __init__(self):
        self.scalars = defaultdict(dict)
        self.tables = {}

    def set_step(self, step=None):
        pass

    def update(self, head="scalar", step=None, **kwargs):
        for k, v in kwargs.items():
            self.scalars[head][k] = v

    def report_figure(self, *args, **kwargs):
        pass

    def report_table(self, title, series, rows, step=None):
        self.tables[(title, series)] = rows


def _case_loader():
    X = torch.randn(8, N_CHANNELS, T_PATCH)
    y = torch.tensor([30.0] * 4 + [60.0] * 4)
    ids = ["rec_a"] * 4 + ["rec_b"] * 4

    class _WithIds(torch.utils.data.Dataset):
        def __len__(self):
            return len(y)

        def __getitem__(self, i):
            return X[i], y[i], ids[i]

    return DataLoader(_WithIds(), batch_size=4)


def _train_with_case_ids(writer):
    model = _make_model(num_classes=1)
    args = _make_epoch_args(model, _case_loader(), nn.HuberLoss(), is_binary=False)
    args.update(task="regression", nb_classes=1, log_writer=writer,
                eval_cfg=EvaluationConfig(agg_windows="mean", detailed_metrics=False))
    return train_one_epoch(**args)


class TestTrainCaseMetrics:
    def test_train_reports_case_level_mae_with_windows_mirrored(self):
        stats = _train_with_case_ids(_RecordingWriter())
        assert {"mae", "window_mae"} <= set(stats)

    def test_case_and_window_series_go_to_their_own_plots(self):
        writer = _RecordingWriter()
        _train_with_case_ids(writer)
        assert "mae" in writer.scalars["train_err"]
        assert "mae" in writer.scalars["train_window_err"]

    def test_running_mae_moves_off_the_err_plot(self):
        writer = _RecordingWriter()
        _train_with_case_ids(writer)
        assert "mae" in writer.scalars["train_step"]
        assert "err" not in writer.scalars

    def test_two_tuple_batches_still_train(self):
        model = _make_model(num_classes=1)
        X, y = torch.randn(8, N_CHANNELS, T_PATCH), torch.arange(8, dtype=torch.float32)
        loader = DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=4)
        args = _make_epoch_args(model, loader, nn.HuberLoss(), is_binary=False)
        args.update(task="regression", nb_classes=1,
                    eval_cfg=EvaluationConfig(agg_windows="mean", detailed_metrics=False))
        stats = train_one_epoch(**args)
        assert "mae" in stats and "window_mae" not in stats


class TestLossTerms:
    def test_single_criterion_logs_its_named_term(self):
        writer = _RecordingWriter()
        _train_with_case_ids(writer)
        assert set(writer.scalars["loss_terms"]) == {"huber_loss"}

    @pytest.mark.parametrize("criterion, name", [
        (nn.HuberLoss(), "huber"), (nn.MSELoss(), "mse"), (nn.L1Loss(), "l1"),
        (nn.BCEWithLogitsLoss(), "bce"), (nn.CrossEntropyLoss(), "ce"),
    ])
    def test_term_names(self, criterion, name):
        assert _loss_term_name(criterion) == name


def test_split_errors_share_one_plot():
    writer = _RecordingWriter()
    _log_split_errors(writer, 3, {"mae": 0.4}, {"mae": 9.3}, {"mae": 10.9})
    assert writer.scalars["err"] == {"train": 0.4, "val": 9.3, "test": 10.9}


def _summary():
    return {
        "best_epoch": 7, "last_epoch": 24,
        "best_train_stats": {"loss": 0.0031, "mae": 0.7468, "lr": 2e-4},
        "last_train_stats": {"loss": 0.0007, "mae": 0.3628, "lr": 1e-6},
        "best_val_stats": {"loss": 0.2723, "mae": 9.3231, "r2": 0.4523,
                           "window_mae": 10.731, "window_loss": 0.2723,
                           "step_time_sec": 0.065},
        "last_val_stats": {"loss": 0.2789, "mae": 9.364, "r2": 0.4334,
                           "window_mae": 10.85, "window_loss": 0.2789,
                           "step_time_sec": 0.065},
        "best_test_stats": {"mae": 10.918},
        "last_test_stats": {"mae": 11.104},
    }


class TestSummaryTables:
    def test_one_table_per_split_with_best_and_last_rows(self):
        tables = build_summary_tables(_summary())
        assert set(tables) == {"train", "val", "test"}
        for rows in tables.values():
            assert [r[:2] for r in rows[1:]] == [["best", "7"], ["last", "24"]]

    def test_values_use_two_decimals(self):
        val = build_summary_tables(_summary())["val"]
        assert val[0] == ["", "epoch", "loss", "mae", "r2", "window_mae"]
        assert val[1][2:] == ["0.27", "9.32", "0.45", "10.73"]

    def test_timing_and_optimizer_state_are_left_out(self):
        tables = build_summary_tables(_summary())
        assert "lr" not in tables["train"][0]
        assert "step_time_sec" not in tables["val"][0]

    def test_loss_terms_follow_the_total_loss(self):
        summary = {"best_epoch": 0, "last_epoch": 0,
                   "best_train_stats": {"loss": 1.0, "mae": 2.0, "phase_loss": 0.1,
                                        "classifier_loss": 0.5}}
        header = build_summary_tables(summary)["train"][0]
        assert header == ["", "epoch", "loss", "classifier_loss", "phase_loss", "mae"]

    def test_missing_splits_and_bad_input_are_skipped(self):
        assert set(build_summary_tables({"best_epoch": 1, "last_epoch": 1,
                                         "last_train_stats": {"mae": 1.0}})) == {"train"}
        assert build_summary_tables(None) == {}
        assert build_summary_tables({"best_epoch": -1, "best_val_stats": {"mae": 1}}) == {}

    def test_tables_are_reported_per_split(self):
        writer = _RecordingWriter()
        log_summary_tables(writer, _summary())
        assert set(writer.tables) == {("summary", s) for s in ("train", "val", "test")}


def test_multiwriter_fans_out_tables(tmp_path):
    a, b = _RecordingWriter(), _RecordingWriter()
    MultiWriter([a, b]).report_table("summary", "val", [["", "mae"], ["best", "1.00"]])
    assert a.tables == b.tables == {("summary", "val"): [["", "mae"], ["best", "1.00"]]}


def test_tensorboard_writes_the_table_as_text(tmp_path):
    writer = TensorboardLogger(str(tmp_path))
    writer.report_table("summary", "val", [["", "mae"], ["best", "1.00"]])
    writer.flush()
    assert any(tmp_path.iterdir())
