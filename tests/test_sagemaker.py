"""SageMaker submission: pure spec/plan building, dry-run submission, and an
end-to-end debug submission that never imports the ``sagemaker`` SDK or contacts
AWS (a fake launcher simulates the container by running the real entry point)."""

import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest

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


def test_estimator_kwargs_input_mode():
    spec = SageMakerJobSpec(entry_point='e.py', input_mode='FastFile')
    assert estimator_kwargs(spec)['input_mode'] == 'FastFile'
    # Unset -> omitted so the SDK applies its own default.
    assert 'input_mode' not in estimator_kwargs(SageMakerJobSpec(entry_point='e.py'))


def test_input_mode_flows_from_config_to_spec():
    c = FinetuneRunConfig()
    c.sagemaker.input_mode = 'FastFile'
    plans = sub.plan_jobs(c, 's3://b/run.yaml')
    assert plans[0].spec.input_mode == 'FastFile'
    assert estimator_kwargs(plans[0].spec)['input_mode'] == 'FastFile'


def test_invalid_input_mode_rejected():
    c = FinetuneRunConfig()
    c.sagemaker.input_mode = 'Fastfile'  # wrong case
    with pytest.raises(ValueError, match='input_mode'):
        sub.submit(c, dry_run=True)


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
    assert plans[0].spec.hyperparameters['phase'] == 'finetune'


def test_plan_jobs_vqnsp_and_pretrain_single_job():
    from labram.configs.run_configs import PretrainRunConfig, VQNSPRunConfig
    for phase, cfg in (('vqnsp', VQNSPRunConfig()), ('pretrain', PretrainRunConfig())):
        plans = sub.plan_jobs(cfg, 's3://b/run.yaml', phase=phase)
        assert len(plans) == 1 and plans[0].fold is None
        assert plans[0].spec.hyperparameters['phase'] == phase
        assert 'fold' not in plans[0].spec.hyperparameters
        assert plans[0].spec.tags['phase'] == phase


def test_phase_configs_cover_all_trainers():
    assert set(sub.PHASE_CONFIGS) == {'vqnsp', 'pretrain', 'finetune'}


# ------------------------------------------------------------------ s3 channels


def _s3_config():
    c = FinetuneRunConfig()
    c.data.data_path = 's3://bucket/data/TUAB'
    c.finetune_checkpoint.finetune = 's3://bucket/ckpts/labram-base.pth'
    return c


def test_stage_s3_inputs_channels_data_and_checkpoint():
    c = _s3_config()
    c.data.split_json = 's3://bucket/runs/data_split.json'

    staged = sub.stage_s3_inputs(c, 'finetune')
    assert staged.channels['dataset'] == 's3://bucket/data/TUAB'
    assert staged.channels['pretrained'] == 's3://bucket/ckpts/labram-base.pth'
    assert staged.uploads == {}
    # Config rewritten to the in-container mounts.
    assert c.data.data_path == '/opt/ml/input/data/dataset'
    assert c.finetune_checkpoint.finetune == '/opt/ml/input/data/pretrained/labram-base.pth'
    # split_json stays an s3:// URI (read directly in-container), no channel.
    assert 'split' not in staged.channels
    assert c.data.split_json == 's3://bucket/runs/data_split.json'


def test_stage_s3_inputs_codebook_tokenizer():
    c = _s3_config()
    c.model.codebook_reg.tokenizer_weight = 's3://bucket/vqnsp.pth'
    staged = sub.stage_s3_inputs(c, 'finetune')
    assert staged.channels['tokenizer'] == 's3://bucket/vqnsp.pth'
    assert c.model.codebook_reg.tokenizer_weight == '/opt/ml/input/data/tokenizer/vqnsp.pth'


def test_stage_s3_inputs_rejects_local_data_path():
    c = FinetuneRunConfig()
    c.data.data_path = '/local/TUAB'
    with pytest.raises(ValueError, match='data.data_path'):
        sub.stage_s3_inputs(c, 'finetune')


def test_stage_s3_inputs_queues_local_checkpoint_for_upload(tmp_path):
    ckpt = tmp_path / 'labram-base.pth'
    ckpt.write_bytes(b'weights')
    c = _s3_config()
    c.finetune_checkpoint.finetune = str(ckpt)

    staged = sub.stage_s3_inputs(c, 'finetune')
    assert staged.uploads == {'pretrained': str(ckpt)}
    assert 'pretrained' not in staged.channels
    # The job still sees a mounted path, not the submitter's filesystem.
    assert c.finetune_checkpoint.finetune == '/opt/ml/input/data/pretrained/labram-base.pth'


def test_stage_s3_inputs_missing_local_checkpoint_fails_fast():
    c = _s3_config()
    c.finetune_checkpoint.finetune = './checkpoints/does-not-exist.pth'
    with pytest.raises(FileNotFoundError, match='finetune_checkpoint.finetune'):
        sub.stage_s3_inputs(c, 'finetune')


def test_stage_s3_inputs_leaves_http_checkpoint_alone():
    c = _s3_config()
    c.finetune_checkpoint.finetune = 'https://example.com/labram-base.pth'
    staged = sub.stage_s3_inputs(c, 'finetune')
    assert 'pretrained' not in staged.channels and staged.uploads == {}
    assert c.finetune_checkpoint.finetune == 'https://example.com/labram-base.pth'


# ------------------------------------------------------------ weight S3 mirrors


def test_weight_s3_uris_default_ships_the_public_checkpoints():
    # The shipped configs' local ./checkpoints/*.pth have known S3 copies, so a
    # fresh config carries the mirror map out of the box.
    c = FinetuneRunConfig()
    m = c.sagemaker.weight_s3_uris
    assert m['./checkpoints/labram-base.pth'] == \
        's3://eeg-data-public/models/labram/labram-base.pth'
    assert m['./checkpoints/vqnsp.pth'] == \
        's3://eeg-data-public/models/labram/vqnsp.pth'


def test_stage_s3_inputs_mirrors_local_weight_instead_of_uploading():
    # The shipped ./checkpoints/labram-base.pth is served from its S3 mirror as a
    # channel -- not uploaded -- so the heavy file is not re-shipped each run.
    c = _s3_config()
    c.finetune_checkpoint.finetune = './checkpoints/labram-base.pth'
    staged = sub.stage_s3_inputs(c, 'finetune')
    assert staged.channels['pretrained'] == \
        's3://eeg-data-public/models/labram/labram-base.pth'
    assert staged.uploads == {}
    assert c.finetune_checkpoint.finetune == \
        '/opt/ml/input/data/pretrained/labram-base.pth'


def test_weight_mirror_matches_by_normalized_path():
    # 'checkpoints/x.pth' (no ./) resolves to the same mirror as './checkpoints/x.pth'.
    c = _s3_config()
    c.finetune_checkpoint.finetune = 'checkpoints/labram-base.pth'
    staged = sub.stage_s3_inputs(c, 'finetune')
    assert staged.channels['pretrained'] == \
        's3://eeg-data-public/models/labram/labram-base.pth'
    assert staged.uploads == {}


def test_weight_mirror_skips_the_local_existence_check():
    # A mirrored path need not exist on the submitting machine -- the point is to
    # avoid the local file entirely -- so no FileNotFoundError is raised.
    c = _s3_config()
    c.sagemaker.weight_s3_uris = {'./ghost/labram-base.pth': 's3://mirror/base.pth'}
    c.finetune_checkpoint.finetune = './ghost/labram-base.pth'
    staged = sub.stage_s3_inputs(c, 'finetune')
    assert staged.channels['pretrained'] == 's3://mirror/base.pth'
    assert staged.uploads == {}
    assert c.finetune_checkpoint.finetune == '/opt/ml/input/data/pretrained/base.pth'


def test_cleared_mirror_falls_back_to_upload(tmp_path):
    # Clearing the map forces the local file to upload again (opt-out path).
    ckpt = tmp_path / 'labram-base.pth'
    ckpt.write_bytes(b'weights')
    c = _s3_config()
    c.sagemaker.weight_s3_uris = {}
    c.finetune_checkpoint.finetune = str(ckpt)
    staged = sub.stage_s3_inputs(c, 'finetune')
    assert staged.uploads == {'pretrained': str(ckpt)}
    assert 'pretrained' not in staged.channels


def test_stage_s3_inputs_pretrain_tokenizer_uses_mirror():
    from labram.configs.run_configs import PretrainRunConfig
    c = PretrainRunConfig()
    c.data.data_path = 's3://bucket/data/TUEG'
    c.model.tokenizer.tokenizer_weight = './checkpoints/vqnsp.pth'
    staged = sub.stage_s3_inputs(c, 'pretrain')
    assert staged.channels['tokenizer'] == \
        's3://eeg-data-public/models/labram/vqnsp.pth'
    assert staged.uploads == {}
    assert c.model.tokenizer.tokenizer_weight == \
        '/opt/ml/input/data/tokenizer/vqnsp.pth'


def test_submit_dry_run_shipped_config_ships_no_weight_uploads():
    # End-to-end plan for a shipped fine-tune config: the pretrained weight is the
    # S3 mirror, and nothing is queued for upload.
    c = FinetuneRunConfig.load_config('labram/configs/defaults/finetune_tuab.json')
    c.data.data_path = 's3://bucket/data/TUAB'
    plans = sub.submit(c, dry_run=True, phase='finetune')
    inputs = plans[0].spec.inputs
    assert inputs['pretrained'] == 's3://eeg-data-public/models/labram/labram-base.pth'
    assert not any(str(v).startswith('<upload') for v in inputs.values())


def test_submit_dry_run_includes_s3_channels_in_inputs():
    plans = sub.submit(_s3_config(), dry_run=True, phase='finetune')
    inputs = plans[0].spec.inputs
    assert inputs['dataset'] == 's3://bucket/data/TUAB'
    assert inputs['pretrained'] == 's3://bucket/ckpts/labram-base.pth'
    assert 'config' in inputs


def test_submit_dry_run_marks_pending_uploads(tmp_path):
    ckpt = tmp_path / 'labram-base.pth'
    ckpt.write_bytes(b'weights')
    c = _s3_config()
    c.finetune_checkpoint.finetune = str(ckpt)
    plans = sub.submit(c, dry_run=True, phase='finetune')
    assert plans[0].spec.inputs['pretrained'] == f'<upload {ckpt}>'


# ------------------------------------------------------------------ source dir


def test_staged_source_dir_excludes_weights_and_untracked_bulk(tmp_path):
    """The packaged code is the git-tracked tree minus model weights — the repo
    root itself also holds the venv, checkpoints and run outputs."""
    c = FinetuneRunConfig()
    with sub.staged_source_dir(c) as source_dir:
        assert source_dir != sub.repo_root()
        staged = {os.path.relpath(os.path.join(dirpath, f), source_dir)
                  for dirpath, _, files in os.walk(source_dir) for f in files}
        assert 'requirements.txt' in staged
        assert 'labram/runs/sagemaker_entry.py' in staged
        assert not [p for p in staged if p.endswith(('.pth', '.pt', '.ckpt'))]
        assert not [p for p in staged if p.startswith('.venv')]
    # Cleaned up on exit.
    assert not os.path.exists(source_dir)


def test_staged_source_dir_honours_explicit_setting(tmp_path):
    c = FinetuneRunConfig()
    c.sagemaker.source_dir = str(tmp_path)
    with sub.staged_source_dir(c) as source_dir:
        assert source_dir == str(tmp_path)
    assert os.path.exists(tmp_path)  # not ours to delete


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


def test_entry_build_config_phase_vqnsp(tmp_path, monkeypatch):
    from labram.configs.run_configs import VQNSPRunConfig
    from labram.runs import sagemaker_entry as entry
    cfg_path = tmp_path / 'vqnsp.yaml'
    VQNSPRunConfig().save_to(str(cfg_path))
    monkeypatch.setenv('SM_MODEL_DIR', str(tmp_path / 'model'))

    cli = entry.parse_cli(['--config', str(cfg_path), '--phase', 'vqnsp'])
    config = entry.build_config(cli)
    assert isinstance(config, VQNSPRunConfig)
    assert config.output.output_dir == str(tmp_path / 'model' / 'vqnsp')
    assert not hasattr(config, 'cross_validation')


# ------------------------------------------------------------------ clearml env


def test_forward_clearml_env(monkeypatch):
    monkeypatch.setenv('CLEARML_API_ACCESS_KEY', 'AK')
    monkeypatch.setenv('CLEARML_API_SECRET_KEY', 'SK')
    c = FinetuneRunConfig()
    c.clearml.enabled = True
    forwarded = sub.forward_clearml_env(c)
    assert forwarded['CLEARML_API_ACCESS_KEY'] == 'AK'
    assert c.sagemaker.environment['CLEARML_API_SECRET_KEY'] == 'SK'
    # Explicit values are not overwritten, and nothing forwards when disabled.
    c2 = FinetuneRunConfig()
    c2.clearml.enabled = False
    assert sub.forward_clearml_env(c2) == {}


# ------------------------------------------------------------------ e2e (faked SDK)


def _make_tuab_root(root: Path, subjects, wins=2):
    root.mkdir(parents=True, exist_ok=True)
    for s in subjects:
        for w in range(wins):
            with open(root / f"{s}_s001_t000_{w}.pkl", "wb") as f:
                pickle.dump({"X": np.random.randn(23, 200).astype("f4"), "y": w % 2}, f)


class _FakeSession:
    """Stands in for a sagemaker.Session: 'uploads' by copying locally."""

    def __init__(self, store: Path):
        self.store = store
        self.boto_region_name = "us-east-1"

    def default_bucket(self):
        return "fake-bucket"

    def upload_data(self, path, bucket=None, key_prefix=""):
        dest = self.store / key_prefix
        dest.mkdir(parents=True, exist_ok=True)
        out = dest / os.path.basename(path)
        shutil.copy(path, out)
        return str(out)


def test_submit_e2e_debug_runs_training(tmp_path, monkeypatch):
    """Full submit path with a faked SDK: config upload -> one job per fold ->
    the fake 'container' runs the real sagemaker_entry, which trains a debug fold
    on synthetic TUAB and writes the fold outputs SageMaker would ship to S3."""
    from labram.runs import sagemaker_entry as entry

    # 4 train + 2 val subjects -> 6-subject pool, 3 folds.
    data = tmp_path / "data"
    _make_tuab_root(data / "train", [f"s{i:02d}" for i in range(4)])
    _make_tuab_root(data / "val", [f"s{i:02d}" for i in range(4, 6)])
    _make_tuab_root(data / "test", [f"s{i:02d}" for i in range(6, 8)])

    config = FinetuneRunConfig.load_config("labram/configs/defaults/finetune_tuab_cv.json")
    config.update(**{
        # Submitted as an s3:// uri, so the run really goes through channel
        # staging; the fake container below maps the mount back to `data`.
        "data.dataset": "TUAB", "data.data_path": "s3://fake-bucket/data/TUAB",
        "finetune_checkpoint.finetune": "",   # train from scratch, no weights to ship
        "trainer.debug": True, "trainer.epochs": 1, "trainer.debug_samples": 4,
        "optimizer.warmup_epochs": 0, "distributed.device": "cpu",
        "model.model": "labram_base_patch200_200",
        "logging.log_model_graph": False, "logging.log_data_split": False,
        "cross_validation.enabled": True, "cross_validation.n_folds": 3,
        "cross_validation.fold": -1, "cross_validation.split_by": "subject",
        "sagemaker.enabled": True, "sagemaker.role": "arn:aws:iam::0:role/test",
        "sagemaker.job_name_prefix": "labram-e2e",
    })

    s3_store = tmp_path / "s3"
    submitted = []

    class _FakeLauncher:
        def __init__(self, region=None, default_role="", sagemaker_session=None):
            self.role = default_role

        def _get_session(self):
            return _FakeSession(s3_store)

        def resolve_role(self, role=""):
            return role or self.role or "arn:aws:iam::0:role/fallback"

        def resolve_image_uri(self, spec):
            return f"763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:{spec.framework_version}-gpu-{spec.py_version}"

        def submit(self, spec, wait=False, job_name=None):
            # Simulate the container: mount the config channel, point outputs at a
            # per-job model dir, and run the real entry point for this fold. The
            # dataset channel stands in for the /opt/ml/input/data/dataset mount.
            assert spec.inputs["dataset"] == "s3://fake-bucket/data/TUAB"
            assert os.path.isfile(os.path.join(spec.source_dir, "requirements.txt"))
            cfg_uri = spec.inputs["config"]
            channel_dir = os.path.dirname(cfg_uri)
            model_dir = tmp_path / "model" / spec.base_job_name
            fold = spec.hyperparameters.get("fold")
            phase = spec.hyperparameters.get("phase", "finetune")
            prev = {k: os.environ.get(k) for k in ("SM_CHANNEL_CONFIG", "SM_MODEL_DIR")}
            os.environ["SM_CHANNEL_CONFIG"] = channel_dir
            os.environ["SM_MODEL_DIR"] = str(model_dir)
            try:
                entry.main(["--phase", phase, "--fold", str(fold),
                            "--set", f"data.data_path={data}"])
            finally:
                for k, v in prev.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            submitted.append((spec.base_job_name, fold, str(model_dir),
                              self.resolve_image_uri(spec)))
            return spec.base_job_name

    monkeypatch.setattr(sub, "SageMakerLauncher", _FakeLauncher)
    plans = sub.submit(config, dry_run=False, phase="finetune")

    # One job per fold, fold number in the job name, image resolved.
    assert [p.fold for p in plans] == [0, 1, 2]
    assert {j[0] for j in submitted} == {"labram-e2e-fold-0", "labram-e2e-fold-1", "labram-e2e-fold-2"}
    for _name, fold, model_dir, image in submitted:
        assert "pytorch-training:2.4.0-gpu-py311" in image
        assert (Path(model_dir) / "cv" / f"fold_{fold}" / "fold_metrics.json").exists()
        assert (Path(model_dir) / "cv" / "cv_split.json").exists()
