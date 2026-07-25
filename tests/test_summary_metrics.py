"""Final/best eval metrics logged as ClearML single values (comparable as a
table in ClearML compare mode) + a final_metrics config section."""

from labram.runs.common import flatten_summary_metrics, log_summary_metrics
from labram.utils.logging import ClearMLLogger, MultiWriter, TensorboardLogger


# ------------------------------------------------------------------ flatten


def test_flatten_summary_metrics_from_train_loop():
    summary = {
        'max_accuracy': 81.0, 'max_accuracy_test': 79.0, 'best_epoch': 12,
        'best_val_stats': {'accuracy': 81.0, 'balanced_accuracy': 0.80, 'note': 'x'},
        'best_test_stats': {'accuracy': 79.0, 'pr_auc': 0.77, 'roc_auc': 0.83},
    }
    flat = flatten_summary_metrics(summary)
    assert flat['best_epoch'] == 12
    assert flat['val_accuracy'] == 81.0
    assert flat['test_accuracy'] == 79.0
    assert flat['test_pr_auc'] == 0.77
    assert 'val_note' not in flat  # non-numeric dropped


def test_flatten_summary_metrics_eval_only():
    flat = flatten_summary_metrics({'accuracy': 70.0, 'balanced_accuracy': 0.68})
    assert flat == {'test_accuracy': 70.0, 'test_balanced_accuracy': 0.68}


def test_flatten_ignores_negative_best_epoch_and_nondict():
    assert 'best_epoch' not in flatten_summary_metrics({'best_epoch': -1})
    assert flatten_summary_metrics(None) == {}


# ------------------------------------------------------------------ writers


class _FakeClearmlLogger:
    def __init__(self):
        self.singles = {}

    def report_single_value(self, name, value):
        self.singles[name] = value


class _FakeTask:
    def __init__(self):
        self.connected = {}

    def connect(self, obj, name=None):
        self.connected[name] = dict(obj)


def test_clearml_logger_report_single_value_forwards():
    cl = _FakeClearmlLogger()
    writer = ClearMLLogger(task=_FakeTask(), clearml_logger=cl)
    writer.report_single_value('test_accuracy', 79.0)
    assert cl.singles == {'test_accuracy': 79.0}


def test_tensorboard_report_single_value(tmp_path):
    tb = TensorboardLogger(log_dir=str(tmp_path))
    tb.report_single_value('test_accuracy', 79.0)  # must not raise
    tb.flush()


def test_multiwriter_fans_out_single_value(tmp_path):
    cl = _FakeClearmlLogger()
    mw = MultiWriter([TensorboardLogger(log_dir=str(tmp_path)),
                      ClearMLLogger(task=_FakeTask(), clearml_logger=cl)])
    mw.report_single_value('test_pr_auc', 0.77)
    assert cl.singles == {'test_pr_auc': 0.77}


# ------------------------------------------------------------------ log helper


def test_log_summary_metrics_reports_singles_and_connects_section():
    cl = _FakeClearmlLogger()
    task = _FakeTask()
    writer = ClearMLLogger(task=task, clearml_logger=cl)
    summary = {'best_epoch': 5, 'best_val_stats': {'accuracy': 80.0},
               'best_test_stats': {'accuracy': 78.0, 'pr_auc': 0.7}}

    flat = log_summary_metrics(writer, summary, config=None)

    # Reported as single values (the ClearML compare-mode table) ...
    assert cl.singles['test_accuracy'] == 78.0
    assert cl.singles['val_accuracy'] == 80.0
    assert cl.singles['test_pr_auc'] == 0.7
    # ... and connected as a 'final_metrics' config section (sortable columns).
    assert task.connected['final_metrics']['test_accuracy'] == 78.0
    assert flat['best_epoch'] == 5


def test_log_summary_metrics_empty_is_noop():
    cl = _FakeClearmlLogger()
    writer = ClearMLLogger(task=_FakeTask(), clearml_logger=cl)
    assert log_summary_metrics(writer, {}, config=None) == {}
    assert cl.singles == {}
