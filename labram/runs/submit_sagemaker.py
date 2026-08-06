# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Submit LaBraM fine-tuning to AWS SageMaker as managed training jobs. A single
# fine-tune becomes one job; a cross-validation study becomes one job per fold,
# each job name embedding the fold number so the folds are easy to track. The run
# config is uploaded to S3 and mounted in-container via the ``config`` input
# channel; labram/runs/sagemaker_entry.py runs it. See docs/sagemaker.md.
#
#   python -m labram.runs.submit_sagemaker --config <finetune_cv.json> \
#       --set sagemaker.role=arn:aws:iam::123:role/SM sagemaker.enabled=true
#
#   python -m labram.runs.submit_sagemaker --config <cfg> --dry_run   # plan only
# ---------------------------------------------------------

import argparse
import contextlib
import io
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import labram.utils as utils
from labram.aws.sagemaker import SageMakerJobSpec, SageMakerLauncher, role_account
from labram.configs.defaults import SAGEMAKER_INPUT_MODES
from labram.configs.run_configs import (
    FinetuneRunConfig,
    PretrainRunConfig,
    RunConfig,
    VQNSPRunConfig,
)
from labram.configs.utils_conf import add_override_arg, parse_overrides

logger = utils.get_logger(__name__)

# Trainer phase -> RunConfig class. Any phase can be dispatched to SageMaker;
# only 'finetune' additionally supports cross-validation (one job per fold).
PHASE_CONFIGS = {
    'vqnsp': VQNSPRunConfig,
    'pretrain': PretrainRunConfig,
    'finetune': FinetuneRunConfig,
}

# Where SageMaker mounts input channels inside the container.
INPUT_MOUNT = '/opt/ml/input/data'
CONFIG_CHANNEL_MOUNT = f'{INPUT_MOUNT}/config'


def _channel_mount(channel: str) -> str:
    return f'{INPUT_MOUNT}/{channel}'


def _basename(uri: str) -> str:
    return uri.rstrip('/').split('/')[-1]


def _is_s3(value: Any) -> bool:
    return isinstance(value, str) and value.startswith('s3://')


@dataclass
class StagedInputs:
    """The input channels for a job, split by what still has to happen.

    ``channels`` are already in S3 and can go straight onto the estimator;
    ``uploads`` are local weight files that must be uploaded first (done in
    :func:`submit`, skipped on a dry run). The run config has already been
    rewritten to the in-container mount paths for both.

    ``object_channels`` names the channels whose S3 uri is a *single object*
    rather than a prefix (every weight file; the dataset is always a prefix).
    They are tracked because ``FastFile``/``Pipe`` expose only the keys *beneath*
    the given prefix, so an object uri yields an empty mount — those channels
    have to be delivered as ``File`` (see :func:`channel_input_modes`).
    """

    channels: Dict[str, str] = field(default_factory=dict)
    uploads: Dict[str, str] = field(default_factory=dict)
    object_channels: Set[str] = field(default_factory=set)

    def resolved(self, uploaded: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        return {**self.channels, **(uploaded or {})}


def stage_s3_inputs(config: RunConfig, phase: str = 'finetune') -> StagedInputs:
    """Turn the run config's data/weight paths into SageMaker input channels and
    rewrite the config to the in-container mount paths, so the job reads them as
    local files.

    Handled: ``data.data_path`` -> ``dataset`` channel (TUAB/TUEV loaders use
    ``os.listdir``, so the data must be a mount, not an S3 URI);
    ``finetune_checkpoint.finetune`` -> ``pretrained`` and
    ``model.codebook_reg.tokenizer_weight`` -> ``tokenizer`` (finetune); and
    ``model.tokenizer.tokenizer_weight`` -> ``tokenizer`` (pretrain, the frozen
    VQNSP). ``data.split_json`` is left as an ``s3://`` URI — it is read directly
    via the shared FileSystem in-container, no channel needed.

    A *local* weight file is staged too: nothing in the submitting machine's
    filesystem exists inside the container, so it is queued for upload rather
    than silently handed to the job as a path that will not resolve — unless the
    path has a configured S3 mirror in ``sagemaker.weight_s3_uris`` (the shipped
    ``./checkpoints/labram-base.pth`` / ``./checkpoints/vqnsp.pth`` do), in which
    case the mirror is used as a channel and nothing is uploaded. A local
    ``data.data_path`` is rejected instead — corpora are far too large to upload
    as part of a submission.
    """
    staged = StagedInputs()

    data = config.data
    if _is_s3(data.data_path):
        staged.channels['dataset'] = data.data_path
        data.data_path = _channel_mount('dataset')
    elif data.data_path:
        raise ValueError(
            f"data.data_path={data.data_path!r} is a local path, which does not "
            "exist inside the SageMaker container. Upload the preprocessed "
            "dataset to S3 and pass the uri, e.g. "
            "--set data.data_path=s3://<bucket>/<prefix>.")

    mirrors = dict(getattr(config.sagemaker, 'weight_s3_uris', {}) or {})
    if phase == 'finetune':
        ck = getattr(config, 'finetune_checkpoint', None)
        if ck is not None and getattr(ck, 'finetune', ''):
            ck.finetune = _stage_weight_file(staged, 'pretrained', ck.finetune,
                                             'finetune_checkpoint.finetune', mirrors)
        cr = getattr(getattr(config, 'model', None), 'codebook_reg', None)
        if cr is not None and getattr(cr, 'tokenizer_weight', ''):
            cr.tokenizer_weight = _stage_weight_file(
                staged, 'tokenizer', cr.tokenizer_weight,
                'model.codebook_reg.tokenizer_weight', mirrors)
    elif phase == 'pretrain':
        # Masked pre-training loads a frozen VQNSP tokenizer via torch.load, so
        # its weight needs the same staging as the fine-tune checkpoints.
        tok = getattr(getattr(config, 'model', None), 'tokenizer', None)
        if tok is not None and getattr(tok, 'tokenizer_weight', ''):
            tok.tokenizer_weight = _stage_weight_file(
                staged, 'tokenizer', tok.tokenizer_weight,
                'model.tokenizer.tokenizer_weight', mirrors)

    if staged.channels:
        logger.info("Staging %d S3 input channel(s): %s",
                    len(staged.channels), staged.channels)
    if staged.uploads:
        logger.info("Staging %d local weight file(s) for upload: %s",
                    len(staged.uploads), staged.uploads)
    return staged


def _normalize_path(value: str) -> str:
    return os.path.normpath(os.path.expanduser(value))


def _weight_mirror(value: str, mirrors: Dict[str, str]) -> str:
    """Return the configured S3 mirror for a local weight path, or ``''``.

    Keys in ``mirrors`` are matched by normalized path, so the shipped
    ``./checkpoints/labram-base.pth`` matches whether written with or without the
    ``./`` prefix (and ``~`` is expanded on both sides).
    """
    if not mirrors:
        return ''
    target = _normalize_path(value)
    for key, uri in mirrors.items():
        if _normalize_path(key) == target:
            return uri
    return ''


def _stage_weight_file(staged: StagedInputs, channel: str, value: str,
                       field_name: str, mirrors: Optional[Dict[str, str]] = None) -> str:
    """Route one weight path onto ``channel`` and return its in-container path.

    ``s3://`` and ``https://`` values need no work beyond the channel (the latter
    is fetched by ``torch.hub`` in-container). A local path that has a configured
    S3 mirror (``sagemaker.weight_s3_uris``) is served from that mirror as a
    channel instead of being uploaded -- so the version-controlled
    ``./checkpoints/*.pth`` are not re-shipped on every submission. Any other
    existing local file is queued for upload; anything else is a path that would
    only fail once the job is running on a GPU, so it fails here instead.

    Every weight channel is a *single object*, so it is recorded in
    ``staged.object_channels`` — the returned mount path is
    ``<mount>/<basename>``, which only holds when the channel is delivered as
    ``File`` (see :func:`channel_input_modes`).
    """
    if value.startswith('https://') or value.startswith('http://'):
        return value
    if _is_s3(value):
        staged.channels[channel] = value
        staged.object_channels.add(channel)
        return f"{_channel_mount(channel)}/{_basename(value)}"
    mirror = _weight_mirror(value, mirrors or {})
    if mirror:
        staged.channels[channel] = mirror
        staged.object_channels.add(channel)
        logger.info("Using S3 mirror for %s=%r instead of uploading the local "
                    "file: %s", field_name, value, mirror)
        return f"{_channel_mount(channel)}/{_basename(mirror)}"
    local = os.path.abspath(os.path.expanduser(value))
    if not os.path.isfile(local):
        raise FileNotFoundError(
            f"{field_name}={value!r} is neither an s3:// uri nor an existing "
            f"local file ({local}). Point it at the checkpoint you want the job "
            "to start from.")
    staged.uploads[channel] = local
    staged.object_channels.add(channel)
    return f"{_channel_mount(channel)}/{os.path.basename(local)}"


# ------------------------------------------------------------------ source dir

# The source_dir is tarred and uploaded before the job starts. The repo root also
# holds the virtualenv, downloaded checkpoints and local run outputs (gigabytes),
# so the packaged code is built from the git-tracked files instead, minus the
# model weights — those travel as input channels, not as code.
CODE_EXCLUDED_SUFFIXES = ('.pth', '.pt', '.ckpt', '.h5', '.hdf5', '.pkl')


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


AWS_PROFILE_FILE = '.aws-profile'


def discover_aws_profile(start: str) -> Tuple[str, str]:
    """Find the AWS profile a directory belongs to: ``(profile, source_path)``.

    Walks up from ``start`` for an :data:`AWS_PROFILE_FILE` holding a profile
    name — the same convention as the shell hook that exports ``AWS_PROFILE``
    per repository, so a checkout wired to one account keeps submitting to it.
    Reading it here (rather than relying on the exported variable) also covers
    the contexts where that hook never runs — IDE run configurations, cron,
    non-interactive shells — and, unlike ``AWS_PROFILE``, a profile passed
    explicitly to boto3 is not overridden by ambient ``AWS_ACCESS_KEY_ID``
    credentials from a different account.

    Returns ``('', '')`` when there is no such file, it is empty, or it cannot
    be read.
    """
    path = os.path.abspath(start)
    while True:
        candidate = os.path.join(path, AWS_PROFILE_FILE)
        if os.path.isfile(candidate):
            try:
                with open(candidate) as fh:
                    profile = fh.read().strip()
            except OSError:
                return '', ''
            return (profile, candidate) if profile else ('', '')
        parent = os.path.dirname(path)
        if parent == path:            # reached the filesystem root
            return '', ''
        path = parent


def resolve_aws_profile(config: RunConfig) -> str:
    """The AWS profile to submit with: ``sagemaker.profile`` if set, else the
    one the checkout is wired to. Logs a discovered profile — picking an account
    out of a file must never be silent."""
    if config.sagemaker.profile:
        return config.sagemaker.profile
    profile, source = discover_aws_profile(repo_root())
    if profile:
        logger.info("Using AWS profile %r from %s (set sagemaker.profile to override).",
                    profile, source)
    return profile


def git_tracked_files(root: str) -> Optional[List[str]]:
    """Repo-relative paths of the git-tracked files, or None outside a checkout."""
    try:
        out = subprocess.run(['git', 'ls-files', '-z'], cwd=root, check=True,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None
    return [p for p in out.stdout.decode().split('\0') if p]


def populate_source_dir(root: str, dest: str) -> int:
    """Copy the packageable git-tracked files from *root* into *dest*.

    Returns the number of files copied. Working-tree content is used, so
    uncommitted edits ship; untracked files do not, and are warned about.
    """
    tracked = git_tracked_files(root)
    if tracked is None:
        raise RuntimeError(f"{root} is not a git checkout")
    copied = 0
    for rel in tracked:
        if rel.endswith(CODE_EXCLUDED_SUFFIXES):
            continue
        src = os.path.join(root, rel)
        if not os.path.isfile(src):        # tracked but deleted in the worktree
            continue
        dst = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied


def warn_untracked_sources(root: str) -> List[str]:
    """Warn about untracked ``.py`` files under ``labram/`` — they are invisible
    to the packaged source dir, so the job would run without them."""
    try:
        out = subprocess.run(['git', 'ls-files', '--others', '--exclude-standard', '-z',
                              '--', 'labram'], cwd=root, check=True,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return []
    untracked = [p for p in out.stdout.decode().split('\0') if p.endswith('.py')]
    if untracked:
        logger.warning(
            "%d untracked python file(s) under labram/ will NOT be packaged into "
            "the SageMaker job: %s. `git add` them first if the job needs them.",
            len(untracked), untracked)
    return untracked


@contextlib.contextmanager
def staged_source_dir(config: RunConfig) -> Iterator[str]:
    """Yield the directory the estimator should package as ``source_dir``.

    An explicit ``sagemaker.source_dir`` is used verbatim. Otherwise a temp copy
    of the git-tracked repo files is built (and removed on exit) so the upload
    stays a few MB instead of the whole working directory.
    """
    explicit = config.sagemaker.source_dir
    if explicit:
        yield explicit
        return
    root = repo_root()
    with tempfile.TemporaryDirectory(prefix='labram-sagemaker-src-') as tmp:
        try:
            copied = populate_source_dir(root, tmp)
        except RuntimeError as exc:
            logger.warning("Packaging the repo root verbatim (%s); set "
                           "sagemaker.source_dir to control what is uploaded.", exc)
            yield root
            return
        warn_untracked_sources(root)
        logger.info("Packaged %d git-tracked file(s) from %s as the job source dir.",
                    copied, root)
        yield tmp


# ------------------------------------------------------------------ clearml

# ClearML credential env vars forwarded into the training container so that
# clearml.enabled runs can talk to the ClearML server from inside SageMaker,
# mapped to where the same setting lives in a local ``clearml.conf``.
CLEARML_ENV_VARS = {
    'CLEARML_API_ACCESS_KEY': 'api.credentials.access_key',
    'CLEARML_API_SECRET_KEY': 'api.credentials.secret_key',
    'CLEARML_API_HOST': 'api.api_server',
    'CLEARML_WEB_HOST': 'api.web_server',
    'CLEARML_FILES_HOST': 'api.files_server',
}
# Without these the in-container Task.init() has no server to talk to.
CLEARML_REQUIRED_ENV_VARS = ('CLEARML_API_ACCESS_KEY', 'CLEARML_API_SECRET_KEY',
                             'CLEARML_API_HOST')


def clearml_conf_credentials() -> Dict[str, str]:
    """Read the submitter's ClearML settings out of their ``clearml.conf``.

    `clearml init` writes credentials to a config file, not to the environment,
    so the env vars alone miss the common setup. Returns ``{ENV_VAR: value}``
    for whatever the local ClearML config resolves; empty if clearml is not
    installed or not configured.
    """
    try:
        from clearml.config import config_obj
    except Exception as exc:  # clearml is an optional dependency
        logger.debug("Cannot read clearml.conf (%s)", exc)
        return {}
    found = {}
    for env_name, conf_path in CLEARML_ENV_VARS.items():
        try:
            value = config_obj.get(conf_path, None)
        except Exception:  # pragma: no cover - malformed local config
            value = None
        if value:
            found[env_name] = str(value)
    return found


def forward_clearml_env(config: RunConfig) -> Dict[str, str]:
    """When ClearML tracking is on, copy the submitter's ClearML credentials into
    ``sagemaker.environment`` so the in-container run logs to the same server.

    Precedence: values already in ``sagemaker.environment`` win, then the
    submitter's ``CLEARML_*`` env vars, then their ``clearml.conf``. Returns the
    names forwarded. No-op when clearml is disabled.
    """
    if not config.clearml.enabled:
        return {}
    env = config.sagemaker.environment
    from_conf = clearml_conf_credentials()
    forwarded = {}
    for name in CLEARML_ENV_VARS:
        value = os.environ.get(name) or from_conf.get(name)
        if name not in env and value:
            env[name] = value
            forwarded[name] = value
    if forwarded:
        logger.info("Forwarding %d ClearML credential var(s) to the SageMaker job: %s",
                    len(forwarded), sorted(forwarded))
    missing = [n for n in CLEARML_REQUIRED_ENV_VARS if not env.get(n)]
    if missing:
        logger.warning(
            "clearml.enabled is set but %s could not be resolved from the "
            "environment or clearml.conf — the job will not be able to report. "
            "Run `clearml-init` on this machine, or export them before "
            "submitting.", missing)
    return forwarded


# ------------------------------------------------------------------ naming


def sanitize_job_name(name: str) -> str:
    """Coerce to the SageMaker job-name grammar: ``[a-zA-Z0-9-]`` up to 63 chars,
    no leading/trailing hyphen."""
    name = re.sub(r'[^a-zA-Z0-9-]', '-', name)
    name = re.sub(r'-+', '-', name).strip('-')
    return name[:63].rstrip('-') or 'labram-job'


def fold_job_name(prefix: str, fold: Optional[int]) -> str:
    return sanitize_job_name(prefix if fold is None else f'{prefix}-fold-{fold}')


def container_config_path(config_uri: str) -> str:
    """In-container path of the uploaded config file (config channel mount)."""
    return os.path.join(CONFIG_CHANNEL_MOUNT, os.path.basename(config_uri))


# ------------------------------------------------------------------ plan


def build_hyperparameters(config_uri: str, fold: Optional[int],
                          phase: str = 'finetune') -> Dict[str, Any]:
    hp: Dict[str, Any] = {'config': container_config_path(config_uri), 'phase': phase}
    if fold is not None:
        hp['fold'] = fold
    return hp


def validate_input_mode(input_mode: str) -> str:
    """Reject an unusable ``sagemaker.input_mode`` here rather than in the
    CreateTrainingJob API response."""
    if input_mode and input_mode not in SAGEMAKER_INPUT_MODES:
        raise ValueError(
            f"sagemaker.input_mode={input_mode!r} is not one of "
            f"{list(SAGEMAKER_INPUT_MODES)}.")
    return input_mode


# Channels whose S3 uri addresses one object rather than a prefix. The config is
# always one uploaded file; weight channels are registered as they are staged.
SINGLE_OBJECT_CHANNELS = ('config',)


def channel_input_modes(input_mode: str,
                        object_channels: Optional[Set[str]] = None) -> Dict[str, str]:
    """Per-channel ``File`` overrides for the channels that address a single S3
    object, when the job-level mode is ``FastFile``/``Pipe``.

    ``FastFile`` "supports S3 prefixes only": it mounts the uri as a prefix and
    exposes the keys *beneath* it, so a channel pointing at one object (the run
    config, a checkpoint) mounts as an empty directory and the job dies on a
    missing file. Those channels are small, so delivering just them as ``File``
    costs one quick download and restores ``<mount>/<basename>``, while the big
    ``dataset`` channel keeps streaming. ``File`` (or unset) needs no override.
    """
    if input_mode in ('', 'File'):
        return {}
    channels = set(SINGLE_OBJECT_CHANNELS) | set(object_channels or ())
    return {channel: 'File' for channel in sorted(channels)}


def build_job_spec(config: RunConfig, config_uri: str, fold: Optional[int],
                   phase: str = 'finetune',
                   extra_inputs: Optional[Dict[str, str]] = None,
                   source_dir: str = '',
                   object_channels: Optional[Set[str]] = None) -> SageMakerJobSpec:
    """Build the :class:`SageMakerJobSpec` for one job (a fold, or the whole run)."""
    sm = config.sagemaker
    tags = dict(sm.tags)
    tags.setdefault('project', config.clearml.project_name or 'LaBraM')
    tags['phase'] = phase
    if fold is not None:
        tags['cv_fold'] = str(fold)
    inputs = {'config': config_uri, **(extra_inputs or {})}
    modes = channel_input_modes(sm.input_mode, object_channels)
    # Only channels this job actually has.
    modes = {c: m for c, m in modes.items() if c in inputs}
    return SageMakerJobSpec(
        entry_point=sm.entry_point,
        source_dir=source_dir or sm.source_dir or '.',
        role=sm.role,
        instance_type=sm.instance_type,
        instance_count=sm.instance_count,
        volume_size_gb=sm.volume_size_gb,
        max_run_sec=sm.max_run_sec,
        use_spot=sm.use_spot,
        max_wait_sec=sm.max_wait_sec,
        framework_version=sm.framework_version,
        py_version=sm.py_version,
        image_uri=sm.image_uri,
        hyperparameters={**sm.hyperparameters, **build_hyperparameters(config_uri, fold, phase)},
        environment=dict(sm.environment),
        tags=tags,
        inputs=inputs,
        input_mode=validate_input_mode(sm.input_mode),
        channel_input_modes=modes,
        output_path=sm.output_path,
        code_location=sm.code_location,
        output_kms_key=sm.output_kms_key,
        base_job_name=fold_job_name(sm.job_name_prefix, fold),
    )


@dataclass
class JobPlan:
    fold: Optional[int]
    job_name: str
    spec: SageMakerJobSpec
    # The real, timestamped name SageMaker assigned (known only after submission);
    # ``job_name`` is just the base prefix.
    submitted_name: str = ''


def plan_jobs(config: RunConfig, config_uri: str, phase: str = 'finetune',
              extra_inputs: Optional[Dict[str, str]] = None,
              source_dir: str = '',
              object_channels: Optional[Set[str]] = None) -> List[JobPlan]:
    """Enumerate the jobs to submit without touching AWS.

    A fine-tune with cross-validation enabled -> one job per fold (or a single
    fold when ``cross_validation.fold >= 0``); any other trainer (vqnsp,
    pretrain, plain fine-tune) -> a single job.
    """
    cv = getattr(config, 'cross_validation', None)
    if phase == 'finetune' and cv is not None and cv.enabled:
        if cv.fold is not None and cv.fold >= 0:
            folds: List[Optional[int]] = [cv.fold]
        else:
            folds = list(range(cv.n_folds))
    else:
        folds = [None]

    plans: List[JobPlan] = []
    for fold in folds:
        spec = build_job_spec(config, config_uri, fold, phase, extra_inputs, source_dir,
                              object_channels)
        plans.append(JobPlan(fold=fold, job_name=spec.base_job_name, spec=spec))
    return plans


# ------------------------------------------------------------------ submit


def source_upload_extra_args(config: RunConfig) -> Optional[Dict[str, str]]:
    """SSE-KMS ``ExtraArgs`` for the submit-side S3 uploads, or ``None`` for a
    plain upload.

    Only set when ``sagemaker.output_kms_key`` is given. The default is a plain
    ``PutObject`` so the submitting identity never needs ``kms:GenerateDataKey``
    on a key it may be denied — which is exactly what breaks the estimator's own
    code upload on an MFA-enforced account (see :func:`package_and_upload_source`
    and docs/sagemaker.md).
    """
    key = config.sagemaker.output_kms_key
    if not key:
        return None
    return {'ServerSideEncryption': 'aws:kms', 'SSEKMSKeyId': key}


def _upload_data(session, local: str, key_prefix: str,
                 extra_args: Optional[Dict[str, str]]) -> str:
    """``session.upload_data`` that only forwards ``extra_args`` when there is
    something to forward, so the common no-KMS path stays a plain upload."""
    if extra_args:
        return session.upload_data(local, key_prefix=key_prefix, extra_args=extra_args)
    return session.upload_data(local, key_prefix=key_prefix)


def upload_run_config(launcher: SageMakerLauncher, config: RunConfig,
                      config_channel: str = '') -> str:
    """Return the S3 uri of the run config, uploading it if not pre-supplied.

    When ``config_channel`` is set it is used verbatim (the caller pre-uploaded
    it). Otherwise the resolved config is written to a temp YAML and uploaded to
    the SageMaker session's default bucket under the job-name prefix.
    """
    if config_channel:
        return config_channel
    session = launcher._get_session()
    sm = config.sagemaker
    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, 'run_config.yaml')
        config.save_to(local)
        key_prefix = sanitize_job_name(sm.job_name_prefix) + '/config'
        uri = _upload_data(session, local, key_prefix, source_upload_extra_args(config))
    logger.info("Uploaded run config -> %s", uri)
    return uri


def upload_staged_weights(launcher: SageMakerLauncher, config: RunConfig,
                          staged: StagedInputs) -> Dict[str, str]:
    """Upload the local weight files queued by :func:`stage_s3_inputs` and return
    ``{channel: s3_uri}``. Nothing to do when every weight is already in S3."""
    if not staged.uploads:
        return {}
    session = launcher._get_session()
    prefix = sanitize_job_name(config.sagemaker.job_name_prefix)
    extra_args = source_upload_extra_args(config)
    uploaded = {}
    for channel, local in staged.uploads.items():
        logger.info("Uploading %s (%.0f MB) as the %r input channel...",
                    local, os.path.getsize(local) / 1e6, channel)
        uploaded[channel] = _upload_data(
            session, local, f'{prefix}/{channel}', extra_args)
        logger.info("  -> %s", uploaded[channel])
    return uploaded


def package_and_upload_source(launcher: SageMakerLauncher, config: RunConfig,
                              source_dir: str,
                              git_info: Optional[Dict[str, Any]] = None) -> str:
    """Tar ``source_dir`` and upload it as ``sourcedir.tar.gz``, returning the
    ``s3://`` uri to hand the estimator as its ``source_dir``.

    The estimator would otherwise tar-and-upload the code itself inside
    ``fit()`` — but that upload inherits the estimator's ``output_kms_key`` (which
    an account-level ``sagemaker.config`` default can set to a customer KMS key),
    so on an MFA-enforced account the submitting identity is denied
    ``kms:GenerateDataKey`` and the whole submission fails. Uploading the code
    ourselves keeps it on the same plain path already used for the config and
    weights (or SSE-KMS with ``sagemaker.output_kms_key`` when set), and the SDK
    skips its own upload when ``source_dir`` is an ``s3://`` tarball. The job's
    model output still honours ``output_kms_key`` at runtime under the execution
    role. See docs/sagemaker.md.
    """
    session = launcher._get_session()
    prefix = sanitize_job_name(config.sagemaker.job_name_prefix)
    extra_args = source_upload_extra_args(config)
    with tempfile.TemporaryDirectory(prefix='labram-sagemaker-tar-') as tmp:
        tar_path = os.path.join(tmp, 'sourcedir.tar.gz')
        # Tar the *contents* of source_dir at the archive root, matching the SDK's
        # layout so the entry point and requirements.txt resolve in-container.
        with tarfile.open(tar_path, 'w:gz') as tar:
            for name in sorted(os.listdir(source_dir)):
                tar.add(os.path.join(source_dir, name), arcname=name)
            if git_info:
                _add_git_info_member(tar, git_info)
        uri = _upload_data(session, tar_path, f'{prefix}/source', extra_args)
    logger.info("Uploaded source dir -> %s", uri)
    return uri


def _add_git_info_member(tar: tarfile.TarFile, info: Dict[str, Any]) -> None:
    """Embed the checkout's git metadata in the source tarball.

    Added from memory rather than written into ``source_dir``, so an explicit
    ``sagemaker.source_dir`` (a directory the user owns) is never mutated. The
    in-container ClearML task reads it back to fill the experiment's code section,
    which it otherwise cannot: the packaged tree has no ``.git``.
    """
    payload = utils.git_info_bytes(info)
    entry = tarfile.TarInfo(name=utils.GIT_INFO_FILENAME)
    entry.size = len(payload)
    entry.mtime = 0            # keep the tarball byte-stable for identical trees
    tar.addfile(entry, io.BytesIO(payload))


def cross_account_role_error(role: str, identity: Dict[str, str],
                             profile: str = '') -> Optional[str]:
    """Message describing a cross-account execution role, or ``None`` if fine.

    ``CreateTrainingJob`` refuses to pass a ``RoleArn`` from an account other
    than the caller's ("Cross-account pass role is not allowed") — no trust
    policy can grant it. AWS only says so after the submission has already
    uploaded the code, config and weights, and names neither account, so check
    it up front. Pure, so the wording is unit-testable without AWS.

    Returns ``None`` when the accounts match or either is unknown (an
    unparseable role, or STS unreachable) — never block a submission on a check
    that could not be made.
    """
    want = role_account(role)
    have = identity.get('Account', '')
    if not want or not have or want == have:
        return None
    where = f"profile {profile!r}" if profile else "your current credentials"
    return (
        f"Cross-account SageMaker execution role.\n"
        f"  role    {role}\n"
        f"          -> account {want}\n"
        f"  caller  {identity.get('Arn', '<unknown>')}\n"
        f"          -> account {have} (from {where})\n"
        f"CreateTrainingJob cannot pass a role across accounts. Either submit "
        f"with credentials in {want} (e.g. AWS_PROFILE=<profile> ... or "
        f"--set sagemaker.profile=<profile>), or pass a sagemaker.role from "
        f"{have}.")


def _reraise_kms_access_denied(exc: Exception) -> None:
    """If ``exc`` is an S3/KMS ``AccessDenied`` on ``kms:GenerateDataKey``, raise a
    ``RuntimeError`` with an actionable message; otherwise return so the caller
    re-raises the original."""
    text = str(exc)
    if 'AccessDenied' not in text or 'kms:' not in text:
        return
    raise RuntimeError(
        "An S3 upload was denied a KMS data key (kms:GenerateDataKey). The "
        "submitting identity is not allowed to use the KMS key that encrypts the "
        "target bucket — often an MFA-enforcement policy that denies KMS without "
        "an MFA-backed session. LaBraM uploads the code, config and weights "
        "itself without a customer key, so this most likely comes from the "
        "output/session bucket enforcing one. Fixes: submit with MFA-backed "
        "credentials; set sagemaker.output_kms_key to a key you are allowed to "
        "use; or point sagemaker.output_path at a bucket without a mandatory "
        "CMK. See docs/sagemaker.md > KMS-encrypted buckets.") from exc


# ------------------------------------------------------------------ reporting

_RULE = '=' * 78
_THIN = '-' * 78


def console_url(job_name: str, region: str) -> str:
    if not region:
        return ''
    return (f"https://{region}.console.aws.amazon.com/sagemaker/home"
            f"?region={region}#/jobs/{job_name}")


def _fmt_duration(seconds: int) -> str:
    hours, rem = divmod(int(seconds), 3600)
    return f"{hours}h{rem // 60:02d}m" if rem // 60 else f"{hours}h"


def submitted_banner(plan: JobPlan, config: RunConfig, region: str,
                     will_wait: bool, git_summary: str = '') -> str:
    """A deliberately loud block confirming the job is running on AWS.

    Printed as soon as the job exists — before any log streaming — because the
    single most useful fact at that moment is that the *remote* job is now
    independent of this terminal: interrupting the local process only stops the
    log tail, never the training. Without saying so, Ctrl-C looks destructive and
    a dropped SSH session looks like a lost run.
    """
    sm = config.sagemaker
    spot = ' (spot)' if sm.use_spot else ''
    lines = [
        '', _RULE,
        '  SAGEMAKER TRAINING JOB SUBMITTED — NOW RUNNING ON AWS',
        _RULE,
        f"  job name    : {plan.submitted_name or plan.job_name}",
        f"  phase/fold  : {plan.spec.hyperparameters.get('phase', '?')}"
        f"{'' if plan.fold is None else f' / fold {plan.fold}'}",
        f"  instance    : {sm.instance_count} x {sm.instance_type}{spot}",
        f"  max runtime : {_fmt_duration(sm.max_run_sec)} (job is stopped at the cap)",
    ]
    if git_summary:
        lines.append(f"  code        : {git_summary}")
    if config.clearml.enabled:
        lines.append(f"  clearml     : {config.clearml.project_name or 'LaBraM'}"
                     f" / {config.clearml.task_name or '<derived>'}")
    url = console_url(plan.submitted_name or plan.job_name, region)
    if url:
        lines.append(f"  console     : {url}")
    lines += [
        _THIN,
        '  The job runs on AWS, NOT in this terminal.',
    ]
    if will_wait:
        lines += [
            '  This process is only STREAMING its logs from CloudWatch.',
            '  Ctrl-C (or closing this terminal, or losing the connection) stops',
            '  the log tail ONLY — training continues and still writes its',
            '  checkpoints to S3 and its metrics to ClearML.',
        ]
    else:
        lines.append('  Nothing left to wait for locally — this command is done.')
    lines += [
        _THIN,
        '  monitor : aws sagemaker describe-training-job --training-job-name \\',
        f"                {plan.submitted_name or plan.job_name}",
        '  stop    : aws sagemaker stop-training-job --training-job-name \\',
        f"                {plan.submitted_name or plan.job_name}",
        _RULE, '',
    ]
    return '\n'.join(lines)


def interrupted_banner(plans: List[JobPlan], region: str) -> str:
    """Shown when the user interrupts a wait: the jobs are still running."""
    names = [p.submitted_name or p.job_name for p in plans if p.submitted_name]
    lines = ['', _RULE,
             '  LOCAL WAIT INTERRUPTED — THE SAGEMAKER JOB(S) ARE STILL RUNNING',
             _RULE,
             '  Only the log stream stopped. Training continues on AWS and its',
             '  outputs still land in S3 / ClearML.', _THIN]
    for name in names:
        lines.append(f"  {name}")
        url = console_url(name, region)
        if url:
            lines.append(f"      {url}")
    lines += [_THIN,
              '  To actually stop a job:',
              '      aws sagemaker stop-training-job --training-job-name <name>',
              _RULE, '']
    return '\n'.join(lines)


def submit(config: RunConfig, dry_run: bool = False, phase: str = 'finetune',
           detach: bool = False) -> List[JobPlan]:
    """Submit the planned SageMaker job(s) for the given trainer ``phase``. With
    ``dry_run`` only the plan is returned (no AWS calls, no SDK import), which is
    what the tests exercise."""
    sm = config.sagemaker
    validate_input_mode(sm.input_mode)
    # Turn the data/checkpoint paths into input channels + rewrite the config to
    # the in-container mounts (done before upload so the job sees local paths).
    staged = stage_s3_inputs(config, phase)

    if dry_run:
        placeholder = sm.config_channel or 's3://<bucket>/<prefix>/run_config.yaml'
        pending = {c: f'<upload {p}>' for c, p in staged.uploads.items()}
        plans = plan_jobs(config, placeholder, phase, staged.resolved(pending),
                          object_channels=staged.object_channels)
        for p in plans:
            logger.info("[dry-run] phase=%s job=%s fold=%s inputs=%s hp=%s modes=%s", phase,
                        p.job_name, p.fold, p.spec.inputs, p.spec.hyperparameters,
                        p.spec.channel_input_modes)
        return plans

    # Make ClearML credentials available in-container before the config/env is
    # packaged, so clearml.enabled runs can log from inside SageMaker.
    forward_clearml_env(config)

    profile = resolve_aws_profile(config)
    launcher = SageMakerLauncher(region=sm.region or None, default_role=sm.role,
                                 profile=profile or None)
    # Fail fast with an actionable message if no usable execution role.
    role = launcher.resolve_role(sm.role)
    logger.info("SageMaker execution role: %s", role)
    # Which credentials are we actually submitting with? Logged unconditionally
    # because everything below (bucket names, the role) is account-scoped, and
    # checked here so a cross-account role fails before any upload rather than
    # after the code/config/weights have gone to the wrong account's bucket.
    identity = launcher.caller_identity()
    if identity:
        logger.info("AWS caller identity: %s (account %s%s)",
                    identity.get('Arn', '<unknown>'), identity.get('Account', '?'),
                    f", profile {profile}" if profile else '')
    else:
        logger.warning("Could not read the AWS caller identity (sts:GetCallerIdentity); "
                       "skipping the cross-account role check.")
    mismatch = cross_account_role_error(role, identity, profile)
    if mismatch:
        raise SystemExit(mismatch)

    # Record what code this run *is*: the container has no .git, so this is the
    # only way branch/commit/uncommitted changes reach the ClearML experiment.
    git_info = utils.collect_git_info(repo_root())
    git_summary = utils.format_git_summary(git_info)
    if git_info:
        logger.info("Shipping git provenance to the job: %s", git_summary)
        if git_info.get('dirty'):
            logger.warning(
                "The working tree is dirty — the job runs the *working tree* "
                "content, which does not match commit %s. The diff is recorded "
                "on the ClearML experiment.", git_info.get('commit_short'))
    else:
        logger.warning("No git metadata for %s — the ClearML experiment will not "
                       "record a branch/commit for this run.", repo_root())

    try:
        uploaded = upload_staged_weights(launcher, config, staged)
        config_uri = upload_run_config(launcher, config, sm.config_channel)
        # Package + upload the code ourselves and pass the estimator an s3://
        # sourcedir.tar.gz, so it skips its own KMS-inheriting code upload.
        with staged_source_dir(config) as source_dir:
            source_uri = package_and_upload_source(launcher, config, source_dir,
                                                   git_info)
        plans = plan_jobs(config, config_uri, phase, staged.resolved(uploaded), source_uri,
                          object_channels=staged.object_channels)
        if plans and plans[0].spec.channel_input_modes:
            # Loud, because it silently rescues an otherwise-baffling failure.
            logger.info(
                "Delivering single-object channel(s) %s as File (input_mode=%s "
                "exposes only keys *under* a prefix, so an object uri would mount "
                "empty); the dataset channel still streams.",
                sorted(plans[0].spec.channel_input_modes), sm.input_mode)
        if plans:
            # Log the training image that will actually be used (verify it resolves).
            try:
                logger.info("SageMaker training image: %s",
                            launcher.resolve_image_uri(plans[0].spec))
            except Exception as exc:  # pragma: no cover - depends on SDK/region
                logger.warning("Could not resolve training image URI: %s", exc)
        region = sm.region or getattr(launcher._get_session(), 'boto_region_name', '')
        # --detach: create the job(s) and return, streaming nothing.
        stream_logs = sm.stream_logs and not detach
        for p in plans:
            logger.info("Submitting SageMaker job %s (fold=%s)", p.job_name, p.fold)
            # Only the final job blocks when wait is requested, so earlier folds are
            # dispatched without waiting on each other.
            wait = sm.wait and (p is plans[-1]) and not detach

            def announce(name: str, plan: JobPlan = p, waiting: bool = wait) -> None:
                # Runs the moment the job exists, before any log streaming, so the
                # real (timestamped) job name is on screen even if the wait is
                # later interrupted or the connection drops.
                plan.submitted_name = name
                logger.info(submitted_banner(plan, config, region, waiting,
                                             git_summary))
            try:
                launcher.submit(p.spec, wait=wait, stream_logs=stream_logs,
                                on_submitted=announce)
            except KeyboardInterrupt:
                logger.warning(interrupted_banner(plans, region))
                raise SystemExit(130) from None
    except Exception as exc:
        _reraise_kms_access_denied(exc)  # raises a clearer error, or returns
        raise
    logger.info("Submitted %d SageMaker job(s).", len(plans))
    return plans


# ------------------------------------------------------------------ cli


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Submit LaBraM training to AWS SageMaker')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--phase', choices=sorted(PHASE_CONFIGS), default='finetune',
                        help='Which trainer to submit: vqnsp | pretrain | finetune.')
    add_override_arg(parser)
    parser.add_argument('--dry_run', action='store_true',
                        help='Print the job plan without contacting AWS.')
    parser.add_argument('--detach', action='store_true',
                        help='Submit and exit immediately: do not wait for the '
                             'job and do not stream its logs (overrides '
                             'sagemaker.wait). The job keeps running on AWS.')
    return parser.parse_args()


def main() -> None:
    cli = parse_cli()
    # Submission is a one-shot CLI: what it stages, uploads and resolves is the
    # only feedback the user gets before a GPU job starts costing money.
    utils.configure_logging()
    overrides = parse_overrides(cli.overrides)
    config_cls = PHASE_CONFIGS[cli.phase]
    config = config_cls.load_config(cli.config, **overrides)
    if not config.sagemaker.enabled and not cli.dry_run:
        raise SystemExit(
            "sagemaker.enabled is false; set it (e.g. --set sagemaker.enabled=true) "
            "or use --dry_run to preview the job plan.")
    plans = submit(config, dry_run=cli.dry_run, phase=cli.phase, detach=cli.detach)
    for p in plans:
        name = p.submitted_name or p.job_name
        print(f"{'[dry-run] ' if cli.dry_run else ''}{name} (fold={p.fold})")


if __name__ == '__main__':
    main()
