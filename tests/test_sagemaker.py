"""SageMaker submission: pure spec/plan building and dry-run submission that
never import the ``sagemaker`` SDK or contact AWS."""

from labram.aws.sagemaker import SageMakerJobSpec, estimator_kwargs
from labram.configs.run_configs import FinetuneRunConfig
from labram.runs import submit_sagemaker as sub


# ------------------------------------------------------------------ estimator kwargs


def test_estimator_kwargs_managed_dlc():
    spec = SageMakerJobSpec(entry_point='e.py', role='arn:role',
                            framework_version='2.4.1', py_version='py311')
    kw = estimator_kwargs(spec)
    assert kw['framework_version'] == '2.4.1' and kw['py_version'] == 'py311'
    assert 'image_uri' not in kw
    assert kw['entry_point'] == 'e.py' and kw['role'] == 'arn:role'


def test_estimator_kwargs_image_uri_wins():
    spec = SageMakerJobSpec(entry_point='e.py', image_uri='123.dkr.ecr/img:tag')
    kw = estimator_kwargs(spec)
    assert kw['image_uri'] == '123.dkr.ecr/img:tag'
    # Managed selectors omitted when an explicit image is given (SDK rejects both).
    assert 'framework_version' not in kw and 'py_version' not in kw


def test_estimator_kwargs_spot_sets_max_wait():
    spec = SageMakerJobSpec(entry_point='e.py', use_spot=True, max_run_sec=100, max_wait_sec=0)
    kw = estimator_kwargs(spec)
    assert kw['use_spot_instances'] is True
    assert kw['max_wait'] == 100  # falls back to max_run when unset
    spec2 = SageMakerJobSpec(entry_point='e.py', use_spot=True, max_run_sec=100, max_wait_sec=200)
    assert estimator_kwargs(spec2)['max_wait'] == 200


def test_estimator_kwargs_tags_formatted():
    spec = SageMakerJobSpec(entry_point='e.py', tags={'a': 'b'})
    assert estimator_kwargs(spec)['tags'] == [{'Key': 'a', 'Value': 'b'}]


# ------------------------------------------------------------------ naming


def test_sanitize_and_fold_job_name():
    assert sub.sanitize_job_name('LaBraM/finetune_tuab cv') == 'LaBraM-finetune-tuab-cv'
    assert sub.fold_job_name('labram-finetune', 3) == 'labram-finetune-fold-3'
    assert sub.fold_job_name('labram-finetune', None) == 'labram-finetune'
    # 63-char cap, no trailing hyphen.
    assert len(sub.sanitize_job_name('x' * 100)) == 63


def test_container_config_path():
    assert sub.container_config_path('s3://b/p/run_config.yaml') == \
        '/opt/ml/input/data/config/run_config.yaml'


# ------------------------------------------------------------------ plan


def _cv_config(n_folds=4, fold=-1):
    c = FinetuneRunConfig()
    c.sagemaker.job_name_prefix = 'labram-tuab'
    c.cross_validation.enabled = True
    c.cross_validation.n_folds = n_folds
    c.cross_validation.fold = fold
    return c


def test_plan_jobs_cv_one_per_fold():
    plans = sub.plan_jobs(_cv_config(n_folds=5, fold=-1), 's3://b/run.yaml')
    assert [p.fold for p in plans] == [0, 1, 2, 3, 4]
    assert plans[2].job_name == 'labram-tuab-fold-2'
    assert plans[2].spec.hyperparameters['fold'] == 2
    assert plans[2].spec.hyperparameters['config'] == '/opt/ml/input/data/config/run.yaml'
    assert plans[2].spec.inputs == {'config': 's3://b/run.yaml'}
    assert plans[2].spec.tags['cv_fold'] == '2'


def test_plan_jobs_single_fold():
    plans = sub.plan_jobs(_cv_config(n_folds=5, fold=2), 's3://b/run.yaml')
    assert [p.fold for p in plans] == [2]


def test_plan_jobs_non_cv():
    c = FinetuneRunConfig()
    plans = sub.plan_jobs(c, 's3://b/run.yaml')
    assert len(plans) == 1 and plans[0].fold is None
    assert 'fold' not in plans[0].spec.hyperparameters


def test_submit_dry_run_no_sdk(monkeypatch):
    # Guarantee the SDK is never imported on the dry-run path.
    import builtins
    real_import = builtins.__import__

    def guard(name, *a, **k):
        if name == 'sagemaker' or name.startswith('sagemaker.'):
            raise AssertionError("sagemaker SDK must not be imported during dry-run")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, '__import__', guard)
    plans = sub.submit(_cv_config(n_folds=3, fold=-1), dry_run=True)
    assert [p.fold for p in plans] == [0, 1, 2]


def test_default_entry_point_is_sagemaker_entry():
    c = FinetuneRunConfig()
    assert c.sagemaker.entry_point == 'labram/runs/sagemaker_entry.py'


# ------------------------------------------------------------------ container entry


def test_entry_find_config_from_channel(tmp_path, monkeypatch):
    from labram.runs import sagemaker_entry as entry
    channel = tmp_path / 'config'
    channel.mkdir()
    (channel / 'run_config.yaml').write_text('x: 1\n')
    monkeypatch.setenv('SM_CHANNEL_CONFIG', str(channel))
    assert entry.find_config_path(None) == str(channel / 'run_config.yaml')


def test_entry_build_config_sets_fold_and_output(tmp_path, monkeypatch):
    from labram.runs import sagemaker_entry as entry
    cfg_path = tmp_path / 'cfg.yaml'
    FinetuneRunConfig().save_to(str(cfg_path))
    monkeypatch.setenv('SM_MODEL_DIR', str(tmp_path / 'model'))

    cli = entry.parse_cli(['--config', str(cfg_path), '--fold', '2'])
    config = entry.build_config(cli)
    assert config.cross_validation.enabled is True
    assert config.cross_validation.fold == 2
    assert config.output.output_dir.startswith(str(tmp_path / 'model'))
    assert config.cross_validation.base_dir.startswith(str(tmp_path / 'model'))
