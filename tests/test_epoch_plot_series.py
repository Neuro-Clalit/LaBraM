"""The per-epoch confusion-matrix / ROC / PR figures are logged under a
distinct series per epoch (``plot_per_epoch``) so every epoch is retained and
viewable, instead of each epoch overwriting the previous one."""

import numpy as np

from labram.configs.train_config import EvaluationConfig
from labram.train.train_finetune import _epoch_series, _log_detailed_report
from labram.utils.eval_metrics import classification_report


class _RecordingWriter:
    def __init__(self):
        self.cm = []   # (series, step)
        self.fig = []  # (series, step)

    def report_confusion_matrix(self, title, matrix, step=None, labels=None, series="confusion_matrix"):
        self.cm.append((series, step))

    def report_figure(self, title, figure, step=None, series="figure"):
        self.fig.append((series, step))


def _binary_report():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.2, 0.4, 0.6, 0.9])
    return classification_report(p, y, is_binary=True, nb_classes=1)


def test_epoch_series_appends_epoch_when_enabled():
    cfg = EvaluationConfig(plot_per_epoch=True)
    assert _epoch_series("confusion_matrix", 3, cfg) == "confusion_matrix/epoch_003"
    assert _epoch_series("roc_curve", 12, cfg) == "roc_curve/epoch_012"


def test_epoch_series_single_series_when_disabled():
    cfg = EvaluationConfig(plot_per_epoch=False)
    assert _epoch_series("confusion_matrix", 3, cfg) == "confusion_matrix"
    # A ``None`` step (e.g. inference-only) also keeps the base series.
    assert _epoch_series("roc_curve", None, EvaluationConfig(plot_per_epoch=True)) == "roc_curve"


def test_log_detailed_report_uses_per_epoch_series():
    w = _RecordingWriter()
    cfg = EvaluationConfig(plot_per_epoch=True, log_confusion_matrix=True, log_curves=True)
    _log_detailed_report(w, _binary_report(), head="val", step=5, eval_cfg=cfg)
    assert w.cm == [("confusion_matrix/epoch_005", 5)]
    assert ("roc_curve/epoch_005", 5) in w.fig
    assert ("pr_curve/epoch_005", 5) in w.fig


def test_log_detailed_report_single_series_when_disabled():
    w = _RecordingWriter()
    cfg = EvaluationConfig(plot_per_epoch=False, log_confusion_matrix=True, log_curves=True)
    _log_detailed_report(w, _binary_report(), head="val", step=5, eval_cfg=cfg)
    assert w.cm == [("confusion_matrix", 5)]
    assert {s for s, _ in w.fig} == {"roc_curve", "pr_curve"}
