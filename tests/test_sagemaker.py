"""SageMaker submission: pure spec/plan building, dry-run submission, and an
end-to-end debug submission that never imports the ``sagemaker`` SDK or contacts
AWS (a fake launcher simulates the container by running the real entry point)."""

import os
import pickle
import shutil
import sys
import tarfile
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


def test_estimator_kwargs_output_kms_key():
    spec = SageMakerJobSpec(entry_point='e.py',
                            output_kms_key='arn:aws:kms:us-east-1:0:key/abc')
    assert estimator_kwargs(spec)['output_kms_key'] == 'arn:aws:kms:us-east-1:0:key/abc'
    # Unset -> omitted so the SDK / account default applies (used by the
    # execution role for model output at runtime).
    assert 'output_kms_key' not in estimator_kwargs(SageMakerJobSpec(entry_point='e.py'))


def test_input_mode_flows_from_config_to_spec():
    c = FinetuneRunConfig()
    c.sagemaker.input_mode = 'FastFile'
    plans = sub.plan_jobs(c, 's3://b/run.yaml')
    assert plans[0].spec.input_mode == 'FastFile'
    assert estimator_kwargs(plans[0].spec)['input_mode'] == 'FastFile'


# --------------------------------------------- single-object channels vs FastFile


def test_channel_input_modes_only_overrides_for_streaming_modes():
    """FastFile/Pipe expose only the keys *under* a prefix, so single-object
    channels must be delivered as File; File needs no override."""
    assert sub.channel_input_modes('File', {'pretrained'}) == {}
    assert sub.channel_input_modes('', {'pretrained'}) == {}
    assert sub.channel_input_modes('FastFile', {'pretrained'}) == {
        'config': 'File', 'pretrained': 'File'}
    assert sub.channel_input_modes('Pipe', set()) == {'config': 'File'}
    # The bulk dataset channel is a prefix -> keeps streaming, never overridden.
    assert 'dataset' not in sub.channel_input_modes('FastFile', {'pretrained'})


def test_fastfile_forces_file_mode_on_config_and_weight_channels():
    c = _s3_config()
    c.sagemaker.input_mode = 'FastFile'
    staged = sub.stage_s3_inputs(c, 'finetune')
    assert staged.object_channels == {'pretrained'}          # dataset is a prefix
    plans = sub.plan_jobs(c, 's3://b/run.yaml', extra_inputs=staged.resolved(),
                          object_channels=staged.object_channels)
    modes = plans[0].spec.channel_input_modes
    assert modes == {'config': 'File', 'pretrained': 'File'}
    assert plans[0].spec.input_mode == 'FastFile'            # job-level unchanged


def test_channel_modes_restricted_to_channels_the_job_has():
    # No weights staged -> only the config override, never a phantom channel.
    c = FinetuneRunConfig()
    c.sagemaker.input_mode = 'FastFile'
    plans = sub.plan_jobs(c, 's3://b/run.yaml', object_channels={'tokenizer'})
    assert plans[0].spec.channel_input_modes == {'config': 'File'}


def test_uploaded_weight_channel_is_an_object_channel(tmp_path):
    ckpt = tmp_path / 'labram-base.pth'
    ckpt.write_bytes(b'weights')
    c = _s3_config()
    c.sagemaker.weight_s3_uris = {}          # force the upload path
    c.finetune_checkpoint.finetune = str(ckpt)
    staged = sub.stage_s3_inputs(c, 'finetune')
    assert staged.uploads == {'pretrained': str(ckpt)}
    assert 'pretrained' in staged.object_channels


def test_build_inputs_wraps_only_overridden_channels():
    """The launcher turns overridden channels into TrainingInput and leaves the
    rest as plain uris (no SDK import needed when there are no overrides)."""
    from labram.aws.sagemaker import SageMakerLauncher
    spec = SageMakerJobSpec(entry_point='e.py',
                            inputs={'config': 's3://b/run.yaml', 'dataset': 's3://b/data/'})
    assert SageMakerLauncher().build_inputs(spec) == {
        'config': 's3://b/run.yaml', 'dataset': 's3://b/data/'}


# ------------------------------------------------- runtime cap / detach / banner


def test_default_max_run_sec_is_24h():
    """A forgotten GPU job is capped at a day, not four."""
    c = FinetuneRunConfig()
    assert c.sagemaker.max_run_sec == 24 * 60 * 60
    assert sub.plan_jobs(c, 's3://b/run.yaml')[0].spec.max_run_sec == 86400
    assert estimator_kwargs(SageMakerJobSpec(entry_point='e.py'))['max_run'] == 86400


def test_spot_max_wait_follows_the_24h_cap():
    c = FinetuneRunConfig()
    c.sagemaker.use_spot = True
    kw = estimator_kwargs(sub.plan_jobs(c, 's3://b/run.yaml')[0].spec)
    assert kw['max_wait'] == 86400


def test_stream_logs_default_and_flow():
    c = FinetuneRunConfig()
    assert c.sagemaker.stream_logs is True


def test_submit_detach_does_not_wait_or_stream(tmp_path, monkeypatch):
    """--detach: create the job, then return — no wait, no log streaming."""
    calls = []

    class _L:
        def __init__(self, region=None, default_role='', sagemaker_session=None,
                     profile=None):
            pass

        def _get_session(self):
            return _FakeSession(tmp_path)

        def caller_identity(self):
            return {}

        def resolve_role(self, role=''):
            return role or 'arn:aws:iam::0:role/r'

        def resolve_image_uri(self, spec):
            return 'img'

        def submit(self, spec, wait=False, job_name=None, stream_logs=True,
                   on_submitted=None):
            calls.append({'wait': wait, 'stream_logs': stream_logs})
            if on_submitted:
                on_submitted(spec.base_job_name + '-ts')
            return spec.base_job_name + '-ts'

    monkeypatch.setattr(sub, 'SageMakerLauncher', _L)

    def _fresh(**sm):
        # submit() rewrites the config to the in-container mounts, so each
        # submission needs its own config object.
        c = _s3_config()
        c.sagemaker.wait = True      # deliberately on: --detach must override it
        for key, value in sm.items():
            setattr(c.sagemaker, key, value)
        return c

    sub.submit(_fresh(), dry_run=False, phase='finetune', detach=True)
    assert calls == [{'wait': False, 'stream_logs': False}]

    calls.clear()
    plans = sub.submit(_fresh(), dry_run=False, phase='finetune')  # waits+streams
    assert calls == [{'wait': True, 'stream_logs': True}]
    assert plans[0].submitted_name.endswith('-ts')

    # stream_logs=false waits quietly.
    calls.clear()
    sub.submit(_fresh(stream_logs=False), dry_run=False, phase='finetune')
    assert calls == [{'wait': True, 'stream_logs': False}]


def test_submitted_banner_says_the_job_survives_a_local_interrupt():
    c = _s3_config()
    c.sagemaker.instance_type = 'ml.g5.xlarge'
    c.sagemaker.use_spot = True
    plan = sub.plan_jobs(c, 's3://b/run.yaml')[0]
    plan.submitted_name = 'labram-abnormal-2026-08-02-20-32-23-168'

    text = sub.submitted_banner(plan, c, 'us-east-1', will_wait=True,
                                git_summary='commit abc123, branch main, clean')
    assert 'labram-abnormal-2026-08-02-20-32-23-168' in text
    # The point of the banner: Ctrl-C does not kill the training job.
    assert 'training continues' in text.lower()
    assert 'ml.g5.xlarge' in text and '(spot)' in text
    assert '24h' in text                                   # the runtime cap
    assert 'stop-training-job' in text                     # how to really stop it
    assert 'us-east-1.console.aws.amazon.com' in text
    assert 'commit abc123' in text

    # Detached: nothing is streaming, so no "interrupt is safe" wording needed.
    detached = sub.submitted_banner(plan, c, 'us-east-1', will_wait=False)
    assert 'this command is done' in detached.lower()


def test_interrupted_banner_lists_still_running_jobs():
    c = _s3_config()
    plan = sub.plan_jobs(c, 's3://b/run.yaml')[0]
    plan.submitted_name = 'labram-abnormal-ts'
    text = sub.interrupted_banner([plan], 'us-east-1')
    assert 'STILL RUNNING' in text
    assert 'labram-abnormal-ts' in text
    assert 'stop-training-job' in text


def test_console_url():
    assert sub.console_url('job-1', 'eu-west-1') == (
        'https://eu-west-1.console.aws.amazon.com/sagemaker/home'
        '?region=eu-west-1#/jobs/job-1')
    assert sub.console_url('job-1', '') == ''      # unknown region -> no link


# ------------------------------------------------------------------ git provenance


def test_collect_git_info_on_this_repo():
    from labram.utils import git_info as gi
    info = gi.collect_git_info(sub.repo_root())
    assert info is not None and len(info['commit']) == 40
    assert info['commit_short'] == info['commit'][:12]
    assert isinstance(info['dirty'], bool)
    assert 'labram' in info['remote'].lower() or info['remote'] == ''
    assert gi.format_git_summary(info).startswith('commit ')


def test_collect_git_info_outside_a_checkout(tmp_path):
    from labram.utils import git_info as gi
    assert gi.collect_git_info(str(tmp_path)) is None
    assert 'no git metadata' in gi.format_git_summary(None)


def test_git_remote_credentials_are_stripped():
    from labram.utils.git_info import _sanitize_remote
    assert _sanitize_remote('https://user:ghp_secret@github.com/o/r.git') == \
        'https://github.com/o/r.git'
    assert _sanitize_remote('git@github.com:o/r.git') == 'git@github.com:o/r.git'
    assert _sanitize_remote('https://github.com/o/r.git') == 'https://github.com/o/r.git'


def test_git_info_is_shipped_inside_the_source_tarball(tmp_path):
    """The container has no .git, so the metadata travels in the tarball."""
    from labram.utils import git_info as gi
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'requirements.txt').write_text('timm\n')

    session = _RecordingSession()
    c = FinetuneRunConfig()
    info = {'commit': 'a' * 40, 'commit_short': 'a' * 12, 'branch': 'feature/x',
            'remote': 'https://github.com/o/r.git', 'dirty': True,
            'modified_files': ['labram/x.py'], 'untracked_files': [],
            'diff': 'diff --git a/x b/x\n', 'diff_truncated': False}
    sub.package_and_upload_source(_RecordingLauncher(session), c, str(src), info)
    assert gi.GIT_INFO_FILENAME in session.calls[-1]['names']


def test_load_and_apply_git_info_to_clearml_task(tmp_path, monkeypatch):
    from labram.utils import git_info as gi
    info = {'commit': 'b' * 40, 'commit_short': 'b' * 12, 'branch': 'main',
            'remote': 'https://github.com/o/r.git', 'dirty': True,
            'modified_files': ['a.py'], 'untracked_files': ['b.py'],
            'diff': 'DIFF', 'diff_truncated': False}
    gi.write_git_info(str(tmp_path / gi.GIT_INFO_FILENAME), info)
    monkeypatch.chdir(tmp_path)
    assert gi.load_git_info()['commit'] == 'b' * 40

    class _Task:
        def __init__(self):
            self.script, self.connected = None, {}

        def set_script(self, **kw):
            self.script = kw

        def connect(self, d, name=None):
            self.connected[name] = d

    task = _Task()
    assert gi.apply_git_info_to_task(task) is True
    assert task.script['branch'] == 'main'
    assert task.script['commit'] == 'b' * 40
    assert task.script['repository'] == 'https://github.com/o/r.git'
    assert task.script['diff'] == 'DIFF'          # uncommitted changes recorded
    assert task.connected['git']['dirty'] is True
    assert task.connected['git']['untracked_files'] == 'b.py'


def test_apply_git_info_is_a_noop_without_shipped_metadata(tmp_path, monkeypatch):
    """A local run keeps ClearML's own git detection; nothing to replay."""
    from labram.utils import git_info as gi
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(gi.GIT_INFO_ENV_VAR, raising=False)
    assert gi.load_git_info() is None
    assert gi.apply_git_info_to_task(object()) is False
    assert gi.apply_git_info_to_task(None) is False


def test_apply_git_info_never_raises_on_a_broken_task(tmp_path, monkeypatch):
    from labram.utils import git_info as gi

    class _Boom:
        def set_script(self, **kw):
            raise RuntimeError('clearml server down')

    info = {'commit': 'c' * 40, 'branch': 'x', 'remote': '', 'diff': ''}
    assert gi.apply_git_info_to_task(_Boom(), info) is False   # warned, not raised


def test_git_diff_is_capped(monkeypatch):
    from labram.utils import git_info as gi
    monkeypatch.setattr(gi, '_git', lambda root, *a: (
        'x' * (gi.MAX_DIFF_BYTES * 2) if a[0] == 'diff' else
        'd' * 40 if a[0] == 'rev-parse' and a[1] == 'HEAD' else 'main'))
    info = gi.collect_git_info('/anywhere')
    assert info['diff_truncated'] is True
    assert len(info['diff'].encode()) < gi.MAX_DIFF_BYTES + 200


def test_entry_missing_config_in_channel_raises_diagnostic(tmp_path, monkeypatch):
    from labram.runs import sagemaker_entry as entry
    channel = tmp_path / 'config'
    channel.mkdir()                                   # mounted but empty (the bug)
    monkeypatch.setenv('SM_CHANNEL_CONFIG', str(channel))
    with pytest.raises(FileNotFoundError, match='FastFile/Pipe'):
        entry.find_config_path('/opt/ml/input/data/config/run_config.yaml')


def test_output_kms_key_flows_from_config_to_spec():
    c = FinetuneRunConfig()
    c.sagemaker.output_kms_key = 'arn:aws:kms:us-east-1:0:key/abc'
    plans = sub.plan_jobs(c, 's3://b/run.yaml')
    assert plans[0].spec.output_kms_key == 'arn:aws:kms:us-east-1:0:key/abc'
    assert estimator_kwargs(plans[0].spec)['output_kms_key'] == \
        'arn:aws:kms:us-east-1:0:key/abc'


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


class _RecordingSession:
    """Minimal sagemaker.Session stand-in that records the last upload."""

    def __init__(self):
        self.calls = []

    def upload_data(self, path, bucket=None, key_prefix="", extra_args=None):
        names = set(tarfile.open(path).getnames()) if path.endswith('.tar.gz') else None
        self.calls.append({'key_prefix': key_prefix, 'extra_args': extra_args,
                           'names': names})
        return f"s3://fake-bucket/{key_prefix}/{os.path.basename(path)}"


class _RecordingLauncher:
    def __init__(self, session):
        self._session = session

    def _get_session(self):
        return self._session


def test_source_upload_extra_args():
    c = FinetuneRunConfig()
    assert sub.source_upload_extra_args(c) is None            # plain upload by default
    c.sagemaker.output_kms_key = 'k-123'
    assert sub.source_upload_extra_args(c) == {
        'ServerSideEncryption': 'aws:kms', 'SSEKMSKeyId': 'k-123'}


def test_package_and_upload_source_tars_contents(tmp_path):
    """The code is packaged + uploaded by the CLI (not the estimator) as an
    s3:// sourcedir.tar.gz, so the estimator skips its KMS-inheriting upload."""
    src = tmp_path / "src"
    (src / "labram" / "runs").mkdir(parents=True)
    (src / "labram" / "runs" / "sagemaker_entry.py").write_text("# entry\n")
    (src / "requirements.txt").write_text("timm\n")

    session = _RecordingSession()
    c = FinetuneRunConfig()
    c.sagemaker.job_name_prefix = "labram-abnormal"
    uri = sub.package_and_upload_source(_RecordingLauncher(session), c, str(src))

    assert uri == "s3://fake-bucket/labram-abnormal/source/sourcedir.tar.gz"
    call = session.calls[-1]
    assert call['key_prefix'] == "labram-abnormal/source"
    assert call['extra_args'] is None            # plain upload — no KMS by default
    assert "requirements.txt" in call['names']   # contents at the archive root
    assert "labram/runs/sagemaker_entry.py" in call['names']


def test_package_and_upload_source_honours_kms_key(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "requirements.txt").write_text("timm\n")

    session = _RecordingSession()
    c = FinetuneRunConfig()
    c.sagemaker.output_kms_key = "arn:aws:kms:us-east-1:0:key/abc"
    sub.package_and_upload_source(_RecordingLauncher(session), c, str(src))
    assert session.calls[-1]['extra_args'] == {
        'ServerSideEncryption': 'aws:kms',
        'SSEKMSKeyId': 'arn:aws:kms:us-east-1:0:key/abc'}


def test_reraise_kms_access_denied_wraps_kms_errors():
    orig = Exception(
        "S3UploadFailedError: Failed to upload /tmp/x/source.tar.gz ...: An error "
        "occurred (AccessDenied) when calling the PutObject operation: User is not "
        "authorized to perform: kms:GenerateDataKey on resource: arn:aws:kms:...")
    with pytest.raises(RuntimeError, match="output_kms_key"):
        sub._reraise_kms_access_denied(orig)


def test_reraise_kms_access_denied_ignores_other_errors():
    # Non-KMS errors return None so the caller re-raises the original untouched.
    assert sub._reraise_kms_access_denied(ValueError("boom")) is None
    assert sub._reraise_kms_access_denied(
        Exception("AccessDenied on s3:PutObject")) is None  # S3-only, not KMS


# ------------------------------------------------------- cross-account preflight


def test_role_account_parses_the_arn():
    from labram.aws.sagemaker import role_account
    assert role_account('arn:aws:iam::574441342949:role/SageMakerExecutionRole') \
        == '574441342949'
    assert role_account('arn:aws-us-gov:iam::123456789012:role/x') == '123456789012'


def test_role_account_of_a_non_arn_is_empty():
    from labram.aws.sagemaker import role_account
    assert role_account('') == ''
    assert role_account('SageMakerExecutionRole') == ''
    assert role_account('arn:aws:s3:::bucket/key') == ''


def test_cross_account_role_error_names_both_accounts():
    msg = sub.cross_account_role_error(
        'arn:aws:iam::574441342949:role/SM',
        {'Account': '660185423351', 'Arn': 'arn:aws:iam::660185423351:user/leon'})
    assert msg is not None
    assert '574441342949' in msg and '660185423351' in msg
    assert 'user/leon' in msg
    assert 'sagemaker.profile' in msg          # points at the fix


def test_cross_account_role_error_mentions_the_configured_profile():
    msg = sub.cross_account_role_error(
        'arn:aws:iam::111:role/SM', {'Account': '222', 'Arn': 'a'}, profile='neuro')
    assert "profile 'neuro'" in msg


def test_same_account_role_passes():
    assert sub.cross_account_role_error(
        'arn:aws:iam::111:role/SM',
        {'Account': '111', 'Arn': 'arn:aws:iam::111:user/x'}) is None


def test_unknown_identity_or_role_never_blocks():
    # STS unreachable, or a role name the account cannot be read from: the check
    # cannot be made, so it must not stand in the way of a submission.
    assert sub.cross_account_role_error('arn:aws:iam::111:role/SM', {}) is None
    assert sub.cross_account_role_error('SomeRoleName', {'Account': '111'}) is None


def test_submit_aborts_on_cross_account_role_before_uploading(tmp_path, monkeypatch):
    """The preflight must fire before any S3 upload — otherwise the code, config
    and weights land in the *caller's* bucket for a job that can never start."""
    session = _RecordingSession()

    class _L:
        def __init__(self, region=None, default_role='', sagemaker_session=None,
                     profile=None):
            pass

        def _get_session(self):
            return session

        def caller_identity(self):
            return {'Account': '660185423351',
                    'Arn': 'arn:aws:iam::660185423351:user/leon'}

        def resolve_role(self, role=''):
            return role

        def submit(self, spec, **kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("submit() must not be called")

    monkeypatch.setattr(sub, 'SageMakerLauncher', _L)
    c = _s3_config()
    c.sagemaker.role = 'arn:aws:iam::574441342949:role/SageMakerExecutionRole'

    with pytest.raises(SystemExit) as excinfo:
        sub.submit(c, dry_run=False, phase='finetune')
    assert 'Cross-account' in str(excinfo.value)
    assert session.calls == []      # nothing was uploaded


def test_profile_flows_from_config_to_the_launcher(tmp_path, monkeypatch):
    seen = {}

    class _L:
        def __init__(self, region=None, default_role='', sagemaker_session=None,
                     profile=None):
            seen['profile'] = profile
            seen['role'] = default_role

        def _get_session(self):
            return _FakeSession(tmp_path)

        def caller_identity(self):
            return {}

        def resolve_role(self, role=''):
            return role or 'arn:aws:iam::0:role/r'

        def resolve_image_uri(self, spec):
            return 'img'

        def submit(self, spec, wait=False, job_name=None, stream_logs=True,
                   on_submitted=None):
            return 'job'

    monkeypatch.setattr(sub, 'SageMakerLauncher', _L)
    c = _s3_config()
    c.sagemaker.profile = 'neuro'
    sub.submit(c, dry_run=False, phase='finetune')
    assert seen['profile'] == 'neuro'


def test_launcher_passes_the_profile_to_boto3(monkeypatch):
    from labram.aws import sagemaker as sm_lib
    seen = {}

    class _FakeBoto3:
        @staticmethod
        def Session(**kwargs):
            seen.update(kwargs)
            return 'boto-session'

    class _FakeSdk:
        @staticmethod
        def Session(boto_session=None):
            return f'sm-session({boto_session})'

    monkeypatch.setitem(sys.modules, 'boto3', _FakeBoto3)
    monkeypatch.setitem(sys.modules, 'sagemaker', _FakeSdk)
    assert sm_lib.SageMakerLauncher(region='us-east-1', profile='neuro')._get_session() \
        == 'sm-session(boto-session)'
    assert seen == {'profile_name': 'neuro', 'region_name': 'us-east-1'}

    # No profile configured -> boto3 does its own resolution (AWS_PROFILE/default).
    seen.clear()
    sm_lib.SageMakerLauncher()._get_session()
    assert seen == {}


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

    def upload_data(self, path, bucket=None, key_prefix="", extra_args=None):
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
        def __init__(self, region=None, default_role="", sagemaker_session=None,
                     profile=None):
            self.role = default_role

        def _get_session(self):
            return _FakeSession(s3_store)

        def caller_identity(self):
            # Same account as the fallback role -> the preflight lets it through.
            return {"Account": "0", "Arn": "arn:aws:iam::0:user/tester"}

        def resolve_role(self, role=""):
            return role or self.role or "arn:aws:iam::0:role/fallback"

        def resolve_image_uri(self, spec):
            return f"763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:{spec.framework_version}-gpu-{spec.py_version}"

        def submit(self, spec, wait=False, job_name=None, stream_logs=True,
                   on_submitted=None):
            # Simulate the container: mount the config channel, point outputs at a
            # per-job model dir, and run the real entry point for this fold. The
            # dataset channel stands in for the /opt/ml/input/data/dataset mount.
            assert spec.inputs["dataset"] == "s3://fake-bucket/data/TUAB"
            # The CLI packages + "uploads" the code itself, so source_dir is a
            # sourcedir.tar.gz (an s3:// uri in real runs) rather than a local dir.
            assert spec.source_dir.endswith("sourcedir.tar.gz")
            with tarfile.open(spec.source_dir) as tar:
                names = set(tar.getnames())
            assert "requirements.txt" in names
            assert "labram/runs/sagemaker_entry.py" in names
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
            # SageMaker appends a timestamp to the base name; the caller learns the
            # real name through on_submitted, before any waiting.
            real_name = f"{spec.base_job_name}-2026-01-01-00-00-00-000"
            if on_submitted is not None:
                on_submitted(real_name)
            return real_name

    monkeypatch.setattr(sub, "SageMakerLauncher", _FakeLauncher)
    plans = sub.submit(config, dry_run=False, phase="finetune")

    # One job per fold, fold number in the job name, image resolved.
    assert [p.fold for p in plans] == [0, 1, 2]
    assert {j[0] for j in submitted} == {"labram-e2e-fold-0", "labram-e2e-fold-1", "labram-e2e-fold-2"}
    # The real submitted name is recorded on each plan (used by the banner/CLI).
    assert all(p.submitted_name.startswith(p.job_name) for p in plans)
    for _name, fold, model_dir, image in submitted:
        assert "pytorch-training:2.4.0-gpu-py311" in image
        assert (Path(model_dir) / "cv" / f"fold_{fold}" / "fold_metrics.json").exists()
        assert (Path(model_dir) / "cv" / "cv_split.json").exists()
