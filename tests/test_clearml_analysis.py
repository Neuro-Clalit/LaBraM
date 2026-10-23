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


# ---------------------------------------------------------------------------
# Regression (EEG brain age) analysis
# ---------------------------------------------------------------------------

_AGE_HPARAMS = {
    'config/model/task': 'regression',
    'config/data/dataset': 'TUAB_AGE',
    'config/trainer/epochs': '5',
    'config/logging/relative_step_axis': 'True',
    'config/logging/relative_step_scale': '1000',
    'config/loss/regression_loss': 'huber',
    'config/loss/huber_delta': '1.0',
}
# Epoch e of 5 lands at round((e + 1) / 5 * 1000) on the relative axis.
_REL_X = [200, 400, 600, 800, 1000]


def _age_snap(hparams=None, **metrics):
    """Regression snapshot on the relative axis; ``metrics`` maps
    ``'<title>__<series>'`` to five per-epoch values (or one, repeated)."""
    snap = ExperimentSnapshot(task_id='age', task_name='tuab-age',
                              status='completed', tags=['brain_age'])
    snap.hyperparameters = dict(_AGE_HPARAMS if hparams is None else hparams)
    for name, values in metrics.items():
        title, series = name.split('__')
        values = values if isinstance(values, list) else [values] * len(_REL_X)
        s = _series(title, series, values, _REL_X)
        snap.scalars[s.key] = s
    return snap


def _by_cat(snap, category):
    return [i for i in analyze_experiment(snap) if i.category == category]


def test_regression_checkpoint_selection_minimises_mae_on_epoch_axis():
    snap = _age_snap(val_err__mae=[12.0, 10.0, 9.0, 9.6, 10.2])
    ins = _by_cat(snap, 'checkpoint-selection')
    assert ins and ins[0].evidence['best_epoch'] == 2      # x=600 -> epoch 2
    assert ins[0].evidence['best_value'] == pytest.approx(9.0)
    assert 'epoch 4' in ins[0].message                     # final, not "1000"


def test_regression_run_is_not_judged_on_classification_metrics():
    # Falling MAE would read as "best at epoch 0" if maximised like accuracy.
    snap = _age_snap(val_err__mae=[12.0, 11.0, 10.0, 9.5, 9.0])
    assert not _by_cat(snap, 'checkpoint-selection')


def test_task_inferred_from_err_plot_without_hparams():
    snap = _age_snap(hparams={}, val_err__mae=[12.0, 10.0, 9.0, 9.6, 10.2])
    assert _by_cat(snap, 'checkpoint-selection')[0].evidence['best_epoch'] == 600


def test_mean_collapse_is_critical():
    # sqrt(2/pi) * 16 ~= 12.77; MAE 12.5 barely beats predicting the mean.
    snap = _age_snap(val_err__mae=12.5, val_err__target_std=16.0,
                     val_err__pred_std=1.0)
    ins = _by_cat(snap, 'mean-collapse')
    assert ins and ins[0].severity == 'critical'
    assert ins[0].evidence['mean_predictor_mae'] == pytest.approx(12.766, abs=1e-3)


def test_no_mean_collapse_for_a_trained_model():
    snap = _age_snap(val_err__mae=8.0, val_err__target_std=16.0)
    assert not _by_cat(snap, 'mean-collapse')


def test_calibration_flags_compressed_predictions():
    # r=0.8, std_y=16, std_pred=8 -> slope 1.6; best linear RMSE 16*0.6 = 9.6.
    snap = _age_snap(val_err__mae=9.0, val__pearson_r=0.8, val_err__rmse=12.0,
                     val_err__pred_std=8.0, val_err__target_std=16.0,
                     val_err__pred_mean=49.0, val_err__target_mean=49.0)
    ins = _by_cat(snap, 'calibration')
    assert ins and ins[0].severity == 'warning'
    assert ins[0].evidence['rmse_recalibrated'] == pytest.approx(9.6)
    assert ins[0].evidence['slope_true_on_pred'] == pytest.approx(1.6)
    assert 'compressed' in ins[0].message


def test_calibrated_predictions_have_no_calibration_insight():
    # std_pred = r * std_y and RMSE already at the linear optimum.
    snap = _age_snap(val_err__mae=7.5, val__pearson_r=0.8, val_err__rmse=9.6,
                     val_err__pred_std=12.8, val_err__target_std=16.0,
                     val_err__pred_mean=49.0, val_err__target_mean=49.0)
    assert not _by_cat(snap, 'calibration')


def test_age_bias_explained_by_correlation():
    # r=0.7 -> r^2 - 1 = -0.51, observed -0.52: expected, not a defect.
    snap = _age_snap(val_err__mae=9.0, val__age_bias_slope=-0.52,
                     val__pearson_r=0.7, val_err__target_std=16.0,
                     val_err__mae_corrected=7.1)
    ins = _by_cat(snap, 'age-bias')
    assert ins and ins[0].severity == 'warning'
    assert 'expected consequence' in ins[0].message
    assert 'val 7.10 vs raw 9.00' in ins[0].recommendation


def test_age_bias_beyond_correlation_points_at_calibration():
    snap = _age_snap(val_err__mae=9.0, val__age_bias_slope=-0.7,
                     val__pearson_r=0.8)                  # r^2 - 1 = -0.36
    ins = _by_cat(snap, 'age-bias')
    assert ins and 'compressed beyond' in ins[0].message


def test_val_test_gap_on_a_wider_test_cohort_is_info():
    # Same skill over each split's mean predictor, just wider test ages.
    snap = _age_snap(val_err__mae=8.0, test_err__mae=9.5,
                     val_err__target_std=15.8, test_err__target_std=17.8)
    ins = _by_cat(snap, 'val-test-gap')
    assert ins and ins[0].severity == 'info'


def test_val_test_gap_warns_when_skill_drops():
    snap = _age_snap(val_err__mae=8.0, test_err__mae=11.0,
                     val_err__target_std=16.0, test_err__target_std=16.0)
    ins = _by_cat(snap, 'val-test-gap')
    assert ins and ins[0].severity == 'warning'


def test_classification_val_test_gap():
    snap = _snap(va=_series('val', 'balanced_accuracy', [0.7, 0.82, 0.8]),
                 te=_series('test', 'balanced_accuracy', [0.68, 0.72, 0.74]))
    ins = _by_cat(snap, 'val-test-gap')
    assert ins and ins[0].evidence['test'] == pytest.approx(0.72)


def test_age_benchmark_gap():
    snap = _age_snap(val_err__mae=[13.0, 11.0, 10.5, 10.6, 10.7],
                     test_err__mae=11.2, val_window_err__mae=11.8)
    ins = _by_cat(snap, 'age-benchmark')
    assert ins and ins[0].severity == 'warning'
    assert '2.5 y above' in ins[0].message
    assert 'from 11.80 to 10.50' in ins[0].message


def test_age_benchmark_only_for_age_runs():
    hparams = dict(_AGE_HPARAMS, **{'config/data/dataset': 'OTHER'})
    snap = _age_snap(hparams=hparams, val_err__mae=10.5)
    snap.tags = []
    assert not _by_cat(snap, 'age-benchmark')


def test_huber_delta_in_z_units_flagged():
    snap = _age_snap(val_err__mae=9.0, val_err__target_std=16.0)
    ins = _by_cat(snap, 'loss-config')
    assert ins and '≈16 y' in ins[0].message
    snap.hyperparameters['config/loss/huber_delta'] = '0.3'
    assert not _by_cat(snap, 'loss-config')


def test_regression_generalization_gap_compares_windows():
    snap = _age_snap(train_err__mae=[10.0, 8.0, 6.0, 5.0, 4.0],
                     val_window_err__mae=[11.0, 10.0, 9.5, 9.4, 9.4],
                     val_err__mae=[10.0, 9.0, 8.5, 8.4, 8.4])
    ins = _by_cat(snap, 'generalization-gap')
    assert ins and ins[0].evidence['val_mae'] == pytest.approx(9.4)


def test_report_has_selected_epoch_table():
    snap = _age_snap(val_err__mae=[12.0, 10.0, 9.0, 9.6, 10.2],
                     test_err__mae=[12.5, 10.4, 9.8, 10.0, 10.5],
                     val__r2=[0.1, 0.3, 0.45, 0.4, 0.35])
    text = render_report(snap)
    assert '## Selected epoch' in text
    assert 'Best val `mae` (lower is better): 9 at epoch 2.' in text
    assert '| mae | 9 | 9.8 |' in text
    assert '| r2 | 0.45 | - |' in text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_task_id_from_clearml_url():
    from labram.eval.clearml_report import task_id_from
    url = ('https://app.clear.ml/projects/*/tasks/'
           '3700b75684684aba845480a50735018d/scalars?columns=name&deep=true')
    assert task_id_from(url) == '3700b75684684aba845480a50735018d'
    assert task_id_from(' abc ') == 'abc'


def test_cli_analyses_saved_snapshot(tmp_path, capsys):
    from labram.eval.clearml_report import main
    path = tmp_path / 'snapshot.json'
    _age_snap(val_err__mae=12.5, val_err__target_std=16.0).save_json(str(path))
    assert main(['--snapshot', str(path), '--print']) == 0
    assert 'mean-collapse' in capsys.readouterr().out
