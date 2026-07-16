"""Tests for the local ClearML experiment analysis toolkit
(labram.eval.clearml_analysis): pure heuristic analysis over an
ExperimentSnapshot, snapshot (de)serialisation, report export, and a
fake-ClearML fetch."""

import sys
import types

import pytest

from labram.eval.clearml_analysis import (
    ExperimentSnapshot,
    Insight,
    ScalarSeries,
    analyze_experiment,
    load_clearml_experiment,
    render_report,
    save_experiment_report,
)


def _series(title, series, values, iters=None):
    iters = iters if iters is not None else list(range(len(values)))
    return ScalarSeries(title=title, series=series, iterations=list(iters),
                        values=list(values))


def _snap(**scalars):
    snap = ExperimentSnapshot(task_id='t', task_name='demo', status='completed')
    for key, s in scalars.items():
        snap.scalars[s.key] = s
    return snap


# ---------------------------------------------------------------------------
# ScalarSeries helpers
# ---------------------------------------------------------------------------

def test_scalar_series_best_and_finite():
    s = _series('val', 'loss', [1.0, float('nan'), 0.3, 0.5])
    assert s.first == 1.0
    assert s.last == 0.5
    assert s.best('min') == (2, 0.3)          # index 2, value 0.3
    assert s.best('max') == (0, 1.0)
    assert s.has_nonfinite() is True
    assert len(s.finite()) == 3


def test_scalar_series_empty():
    s = ScalarSeries('a', 'b')
    assert s.last is None and s.first is None and s.best() is None


# ---------------------------------------------------------------------------
# Analysis heuristics
# ---------------------------------------------------------------------------

def test_overfitting_detected():
    snap = _snap(
        tr=_series('train', 'loss', [1.0, 0.7, 0.5, 0.35, 0.25]),
        va=_series('val', 'loss', [1.0, 0.6, 0.5, 0.62, 0.8]),
    )
    cats = {i.category for i in analyze_experiment(snap)}
    assert 'overfitting' in cats


def test_generalization_gap_detected():
    snap = _snap(
        tra=_series('train', 'accuracy', [0.7, 0.9, 0.99]),
        vaa=_series('val', 'accuracy', [0.65, 0.75, 0.77]),
    )
    ins = [i for i in analyze_experiment(snap) if i.category == 'generalization-gap']
    assert ins and ins[0].evidence['gap'] == pytest.approx(0.22, abs=1e-6)


def test_nonfinite_loss_is_critical():
    snap = _snap(l=_series('loss', 'loss', [1.0, 0.5, float('inf'), float('nan')]))
    ins = [i for i in analyze_experiment(snap) if i.category == 'divergence']
    assert ins and ins[0].severity == 'critical'


def test_checkpoint_selection_when_best_before_end():
    snap = _snap(va=_series('val', 'accuracy',
                            [0.6, 0.7, 0.85, 0.8, 0.78, 0.77]))
    ins = [i for i in analyze_experiment(snap) if i.category == 'checkpoint-selection']
    assert ins and ins[0].evidence['best_epoch'] == 2


def test_undertraining_detected():
    # Loss falling steeply right to the end (roughly geometric).
    vals = [1.0, 0.8, 0.64, 0.51, 0.41, 0.33, 0.26, 0.21]
    snap = _snap(tl=_series('train', 'loss', vals))
    cats = {i.category for i in analyze_experiment(snap)}
    assert 'undertraining' in cats


def test_plateau_detected():
    vals = [0.5, 0.6, 0.7, 0.78, 0.80, 0.801, 0.802, 0.801, 0.802, 0.801]
    snap = _snap(va=_series('val', 'accuracy', vals))
    cats = {i.category for i in analyze_experiment(snap)}
    assert 'plateau' in cats


def test_lr_schedule_not_decayed():
    snap = _snap(lr=_series('opt', 'lr', [1e-4, 1e-4, 9e-5, 9e-5]))
    ins = [i for i in analyze_experiment(snap) if i.category == 'lr-schedule']
    assert ins and ins[0].severity == 'info'


def test_lr_collapsed_early_is_warning():
    snap = _snap(lr=_series('opt', 'lr',
                            [1e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    ins = [i for i in analyze_experiment(snap) if i.category == 'lr-schedule']
    assert any(i.severity == 'warning' for i in ins)


def test_grad_instability_detected():
    vals = [1.0] * 12 + [50.0]
    snap = _snap(g=_series('opt', 'grad_norm', vals))
    cats = {i.category for i in analyze_experiment(snap)}
    assert 'grad-instability' in cats


def test_loss_scale_collapse_detected():
    snap = _snap(ls=_series('opt', 'loss_scale',
                            [65536, 65536, 32768, 128, 32, 16]))
    cats = {i.category for i in analyze_experiment(snap)}
    assert 'amp-loss-scale' in cats


def test_class_imbalance_detected():
    snap = _snap(
        acc=_series('val', 'accuracy', [0.9, 0.92]),
        bal=_series('val', 'balanced_accuracy', [0.7, 0.72]),
    )
    cats = {i.category for i in analyze_experiment(snap)}
    assert 'class-imbalance' in cats


def test_failed_run_is_critical_and_sorted_first():
    snap = _snap(va=_series('val', 'accuracy', [0.6, 0.85, 0.8, 0.78, 0.77, 0.76]))
    snap.status = 'failed'
    ins = analyze_experiment(snap)
    assert ins[0].category == 'run-status' and ins[0].severity == 'critical'


def test_clean_run_has_no_issues():
    # Smooth converging loss + decaying LR + matched acc, no red flags.
    snap = _snap(
        tl=_series('train', 'loss', [1.0, 0.6, 0.4, 0.32, 0.30, 0.299]),
        vl=_series('val', 'loss', [1.0, 0.62, 0.42, 0.35, 0.33, 0.329]),
        lr=_series('opt', 'lr', [1e-4, 8e-5, 5e-5, 2e-5, 5e-6, 1e-7]),
    )
    cats = {i.category for i in analyze_experiment(snap)}
    assert 'overfitting' not in cats and 'divergence' not in cats


# ---------------------------------------------------------------------------
# Snapshot (de)serialisation + report export
# ---------------------------------------------------------------------------

def test_snapshot_roundtrip(tmp_path):
    snap = _snap(va=_series('val', 'accuracy', [0.6, 0.8]))
    snap.hyperparameters = {'General/lr': 5e-4, 'General/epochs': 50}
    path = tmp_path / 'snap.json'
    snap.save_json(str(path))
    back = ExperimentSnapshot.load_json(str(path))
    assert back.task_id == 'demo' or back.task_id == 't'
    assert back.get('val/accuracy').last == pytest.approx(0.8)
    assert back.hyperparameters['General/epochs'] == 50


def test_save_experiment_report_writes_files(tmp_path):
    snap = _snap(
        tr=_series('train', 'loss', [1.0, 0.7, 0.5, 0.35, 0.25]),
        va=_series('val', 'loss', [1.0, 0.6, 0.5, 0.62, 0.8]),
    )
    out = save_experiment_report(snap, str(tmp_path / 'rep'))
    assert out['snapshot'].endswith('snapshot.json')
    report = (tmp_path / 'rep' / 'report.md').read_text()
    assert '# Experiment analysis' in report
    assert 'overfitting' in report
    assert '## Metric summary' in report


def test_render_report_no_issues():
    snap = ExperimentSnapshot(task_id='x', status='completed')
    text = render_report(snap, [])
    assert 'No heuristic issues detected' in text


# ---------------------------------------------------------------------------
# Fake-ClearML fetch
# ---------------------------------------------------------------------------

class _FakeLogger:
    pass


class _FakeData:
    created = '2026-01-01T00:00:00'
    started = '2026-01-01T00:01:00'
    completed = '2026-01-01T01:00:00'


class _FakeArtifact:
    pass


class _FakeTask:
    id = 'abc123'
    name = 'fake-run'
    comment = 'a note'
    data = _FakeData()
    artifacts = {'run_config': _FakeArtifact(), 'data_split': _FakeArtifact()}
    models = {'output': [types.SimpleNamespace(name='trained_model')]}

    def get_project_name(self):
        return 'LaBraM/finetune'

    def get_status(self):
        return 'completed'

    def get_tags(self):
        return ['base', 'tuab']

    def get_parameters_as_dict(self):
        return {'General': {'lr': 5e-4, 'epochs': 50}}

    def get_reported_scalars(self):
        return {
            'val': {'loss': {'x': [0, 1, 2, 3, 4],
                             'y': [1.0, 0.6, 0.5, 0.62, 0.8]}},
            'train': {'loss': {'x': [0, 1, 2, 3, 4],
                               'y': [1.0, 0.7, 0.5, 0.35, 0.25]}},
        }

    def get_reported_console_output(self, number_of_reports=100):
        return ['epoch 0', 'epoch 1', 'done']


@pytest.fixture
def fake_clearml(monkeypatch):
    captured = {}

    class Task:
        @staticmethod
        def get_task(task_id=None, project_name=None, task_name=None):
            captured['task_id'] = task_id
            captured['project_name'] = project_name
            captured['task_name'] = task_name
            return _FakeTask()

    mod = types.ModuleType('clearml')
    mod.Task = Task
    monkeypatch.setitem(sys.modules, 'clearml', mod)
    return captured


def test_load_clearml_experiment_parses_snapshot(fake_clearml):
    snap = load_clearml_experiment(task_id='abc123')
    assert fake_clearml['task_id'] == 'abc123'
    assert snap.task_id == 'abc123'
    assert snap.project_name == 'LaBraM/finetune'
    assert snap.status == 'completed'
    assert snap.tags == ['base', 'tuab']
    assert snap.hyperparameters['General/lr'] == pytest.approx(5e-4)
    assert snap.get('val/loss').best('min') == (2, 0.5)
    assert snap.artifacts == ['data_split', 'run_config']
    assert snap.models == ['trained_model']
    assert snap.console_tail[-1] == 'done'
    # End-to-end: analysis over the fetched snapshot flags the overfit.
    assert 'overfitting' in {i.category for i in analyze_experiment(snap)}


def test_load_clearml_experiment_missing_task_raises(monkeypatch):
    class Task:
        @staticmethod
        def get_task(**kwargs):
            return None

    mod = types.ModuleType('clearml')
    mod.Task = Task
    monkeypatch.setitem(sys.modules, 'clearml', mod)
    with pytest.raises(ValueError):
        load_clearml_experiment(task_id='missing')
