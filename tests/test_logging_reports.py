"""Tests for the confusion-matrix / figure reporting added to the writers
(TensorboardLogger, ClearMLLogger, MultiWriter) and the plot builders."""

import numpy as np
import pytest

from labram.utils import plots
from labram.utils.logging import (
    ClearMLLogger,
    MultiWriter,
    TensorboardLogger,
    _confusion_matrix_markdown,
)

CM = np.array([[5, 1], [2, 7]])


def test_confusion_matrix_markdown_shape():
    md = _confusion_matrix_markdown(CM, [0, 1])
    lines = md.splitlines()
    assert lines[0].startswith("| true\\pred |")
    assert "5" in md and "7" in md
    assert len(lines) == 2 + CM.shape[0]  # header + separator + rows


def test_tensorboard_report_methods_run(tmp_path):
    tb = TensorboardLogger(log_dir=str(tmp_path))
    tb.set_step(3)
    tb.report_confusion_matrix("val", CM, labels=[0, 1])
    fig = plots.confusion_matrix_figure(CM, [0, 1])
    tb.report_figure("val", fig, series="confusion_matrix")
    tb.report_figure("val", None)  # None is a no-op, must not raise
    tb.flush()
    assert any(p.name.startswith("events.out") for p in tmp_path.iterdir())


class _FakeClearMLLogger:
    def __init__(self):
        self.calls = []

    def report_scalar(self, **k):
        self.calls.append(("scalar", k))

    def report_confusion_matrix(self, **k):
        self.calls.append(("cm", k))

    def report_matplotlib_figure(self, **k):
        self.calls.append(("fig", k))

    def report_media(self, **k):
        self.calls.append(("media", k))

    def flush(self):
        pass


def test_clearml_report_confusion_matrix_and_figure():
    fake = _FakeClearMLLogger()
    cl = ClearMLLogger(clearml_logger=fake)
    cl.set_step(2)
    cl.report_confusion_matrix("test", CM, labels=[0, 1])
    fig = plots.roc_curve_figure(np.array([0, 1]), np.array([0, 1]), 0.9)
    cl.report_figure("test", fig, series="roc_curve")

    kinds = [c[0] for c in fake.calls]
    assert "cm" in kinds and "fig" in kinds
    cm_call = dict(c for c in fake.calls if c[0] == "cm")["cm"]
    assert cm_call["title"] == "test"
    assert cm_call["iteration"] == 2
    assert cm_call["xlabels"] == ["0", "1"]


def test_clearml_report_media():
    fake = _FakeClearMLLogger()
    cl = ClearMLLogger(clearml_logger=fake)
    cl.set_step(1)
    cl.report_media("model", "architecture", "/tmp/model_graph.svg")
    media = dict(c for c in fake.calls if c[0] == "media")["media"]
    assert media["local_path"] == "/tmp/model_graph.svg"
    assert media["title"] == "model"


def test_clearml_no_logger_is_noop():
    cl = ClearMLLogger(task=None)  # no logger available
    cl.report_confusion_matrix("t", CM)   # must not raise
    cl.report_figure("t", object())
    cl.report_media("t", "s", "/tmp/x.svg")


class _RecordingWriter:
    def __init__(self):
        self.cm = []
        self.fig = []

    def set_step(self, step=None):
        pass

    def report_confusion_matrix(self, title, matrix, step=None, labels=None, series="confusion_matrix"):
        self.cm.append((title, series))

    def report_figure(self, title, figure, step=None, series="figure"):
        self.fig.append((title, series))

    def report_media(self, title, series, local_path, step=None):
        self.media = getattr(self, "media", [])
        self.media.append((title, series, local_path))


def test_multiwriter_fans_out_reports():
    a, b = _RecordingWriter(), _RecordingWriter()
    mw = MultiWriter([a, b])
    mw.report_confusion_matrix("val", CM, labels=[0, 1])
    mw.report_figure("val", object(), series="roc_curve")
    mw.report_media("model", "architecture", "/tmp/g.svg")
    assert a.cm == b.cm == [("val", "confusion_matrix")]
    assert a.fig == b.fig == [("val", "roc_curve")]
    assert a.media == b.media == [("model", "architecture", "/tmp/g.svg")]


def test_plot_builders_return_figures_or_none():
    # matplotlib is available in the test env; builders should return Figures.
    assert plots.confusion_matrix_figure(CM, [0, 1]) is not None
    assert plots.roc_curve_figure(np.array([0, 0.5, 1]), np.array([0, 0.7, 1]), 0.8) is not None
    assert plots.pr_curve_figure(np.array([1, 0.5, 0]), np.array([0.4, 0.6, 1]), 0.7) is not None
