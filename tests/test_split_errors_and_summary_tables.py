"""Per-case train metrics, the per-metric regression epoch plots, per-term loss
logging and the best-vs-last summary tables."""
import re
from collections import defaultdict

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from labram.configs.train_config import EvaluationConfig
from labram.runs.common import build_summary_tables, log_summary_tables
from labram.losses import CodebookRegularizedCriterion
from labram.losses.regression import downstream_term_name
from labram.train.train_finetune import _log_regression_epoch, train_one_epoch
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

    def test_mean_and_median_case_pooling_are_both_reported(self):
        stats = _train_with_case_ids(_RecordingWriter())
        for mode in ("mean", "median"):
            assert {f"case_{mode}_{m}" for m in ("mae", "rmse", "r2")} <= set(stats)
        assert stats["case_mean_mae"] == pytest.approx(stats["mae"])

    def test_regression_train_scalars_leave_the_per_split_plots(self):
        writer = _RecordingWriter()
        _train_with_case_ids(writer)
        assert "mae" in writer.scalars["train_step"]
        assert not {"train", "train_err", "train_window", "train_window_err"} & set(writer.scalars)

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
    def test_single_criterion_logs_the_task_term(self):
        writer = _RecordingWriter()
        _train_with_case_ids(writer)
        assert set(writer.scalars["loss_terms"]) == {"train_regression_loss"}

    @pytest.mark.parametrize("task, name", [("regression", "regression"),
                                            ("classification", "classifier")])
    def test_term_names(self, task, name):
        assert downstream_term_name(task) == name

    def test_codebook_criterion_uses_the_same_term_name(self):
        crit = CodebookRegularizedCriterion(nn.HuberLoss(), term_name="regression")
        assert crit.term_name == "regression"


def _epoch_stats(mae, loss, **extra):
    return {"loss": loss, "mae": mae, "case_mean_mae": mae, "case_median_mae": mae - 0.5,
            "window_mae": mae + 1, "rmse": mae + 2, "case_mean_rmse": mae + 2,
            "case_median_rmse": mae + 1.5, "window_rmse": mae + 3,
            "r2": 0.6, "case_mean_r2": 0.6, "case_median_r2": 0.62, "window_r2": 0.5,
            "mse": 99.0, "window_mse": 99.0, **extra}


class TestRegressionEpochPlots:
    def _log(self, **val_extra):
        writer = _RecordingWriter()
        _log_regression_epoch(writer, 3, {
            "train": _epoch_stats(1.0, 0.1),
            "val": _epoch_stats(9.0, 0.4, **val_extra),
            "test": _epoch_stats(10.0, 0.5),
        })
        return writer.scalars

    def test_one_plot_per_metric_and_pooling_with_split_series(self):
        scalars = self._log()
        for metric in ("mae", "rmse", "r2"):
            for agg in ("case_mean", "case_median", "window"):
                assert set(scalars[f"{metric}_{agg}"]) == {"train", "val", "test"}
        assert scalars["mae_case_mean"] == {"train": 1.0, "val": 9.0, "test": 10.0}
        assert scalars["mae_case_median"]["val"] == 8.5
        assert scalars["mae_window"]["test"] == 11.0

    def test_total_loss_per_split(self):
        assert self._log()["loss_epoch"] == {"train": 0.1, "val": 0.4, "test": 0.5}

    def test_mse_is_not_plotted(self):
        names = [n for title, plot in self._log().items() for n in (title, *plot)]
        assert not [n for n in names if re.search(r"(^|_)mse", n)]

    def test_val_and_test_loss_terms_join_the_loss_terms_plot(self):
        terms = self._log()["loss_terms"]
        assert terms == {"val_regression_loss": 0.4, "test_regression_loss": 0.5}

    def test_regularized_terms_come_with_their_total(self):
        terms = self._log(regression_loss=0.3, magnitude_loss=0.2,
                          window_loss=0.4)["loss_terms"]
        assert terms["val_regression_loss"] == 0.3
        assert terms["val_magnitude_loss"] == 0.2
        assert terms["val_total_loss"] == 0.4
        assert "val_window_loss" not in terms

    def test_window_level_runs_plot_their_primary_metrics_as_window(self):
        writer = _RecordingWriter()
        _log_regression_epoch(writer, 0, {"train": {"loss": 0.2, "mae": 3.0}, "val": None})
        assert writer.scalars["mae_window"] == {"train": 3.0}
        assert "mae_case_mean" not in writer.scalars


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
                                        "regression_loss": 0.5}}
        header = build_summary_tables(summary)["train"][0]
        assert header == ["", "epoch", "loss", "phase_loss", "regression_loss", "mae"]

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
