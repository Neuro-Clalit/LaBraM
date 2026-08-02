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
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import labram.utils as utils
from labram.aws.sagemaker import SageMakerJobSpec, SageMakerLauncher
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
    """

    channels: Dict[str, str] = field(default_factory=dict)
    uploads: Dict[str, str] = field(default_factory=dict)

    def resolved(self, uploaded: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        return {**self.channels, **(uploaded or {})}


def stage_s3_inputs(config: RunConfig, phase: str = 'finetune') -> StagedInputs:
    """Turn the run config's data/weight paths into SageMaker input channels and
    rewrite the config to the in-container mount paths, so the job reads them as
    local files.

    Handled: ``data.data_path`` -> ``dataset`` channel (TUAB/TUEV loaders use
    ``os.listdir``, so the data must be a mount, not an S3 URI);
    ``finetune_checkpoint.finetune`` -> ``pretrained``; and
    ``model.codebook_reg.tokenizer_weight`` -> ``tokenizer`` (finetune only).
    ``data.split_json`` is left as an ``s3://`` URI — it is read directly via the
    shared FileSystem in-container, no channel needed.

    A *local* weight file is staged too: nothing in the submitting machine's
    filesystem exists inside the container, so it is queued for upload rather
    than silently handed to the job as a path that will not resolve. A local
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

    if phase == 'finetune':
        ck = getattr(config, 'finetune_checkpoint', None)
        if ck is not None and getattr(ck, 'finetune', ''):
            ck.finetune = _stage_weight_file(staged, 'pretrained', ck.finetune,
                                             'finetune_checkpoint.finetune')
        cr = getattr(getattr(config, 'model', None), 'codebook_reg', None)
        if cr is not None and getattr(cr, 'tokenizer_weight', ''):
            cr.tokenizer_weight = _stage_weight_file(
                staged, 'tokenizer', cr.tokenizer_weight,
                'model.codebook_reg.tokenizer_weight')

    if staged.channels:
        logger.info("Staging %d S3 input channel(s): %s",
                    len(staged.channels), staged.channels)
    if staged.uploads:
        logger.info("Staging %d local weight file(s) for upload: %s",
                    len(staged.uploads), staged.uploads)
    return staged


def _stage_weight_file(staged: StagedInputs, channel: str, value: str,
                       field_name: str) -> str:
    """Route one weight path onto ``channel`` and return its in-container path.

    ``s3://`` and ``https://`` values need no work beyond the channel (the latter
    is fetched by ``torch.hub`` in-container); an existing local file is queued
    for upload; anything else is a path that would only fail once the job is
    running on a GPU, so it fails here instead.
    """
    if value.startswith('https://') or value.startswith('http://'):
        return value
    if _is_s3(value):
        staged.channels[channel] = value
        return f"{_channel_mount(channel)}/{_basename(value)}"
    local = os.path.abspath(os.path.expanduser(value))
    if not os.path.isfile(local):
        raise FileNotFoundError(
            f"{field_name}={value!r} is neither an s3:// uri nor an existing "
            f"local file ({local}). Point it at the checkpoint you want the job "
            "to start from.")
    staged.uploads[channel] = local
    return f"{_channel_mount(channel)}/{os.path.basename(local)}"


# ------------------------------------------------------------------ source dir

# The source_dir is tarred and uploaded before the job starts. The repo root also
# holds the virtualenv, downloaded checkpoints and local run outputs (gigabytes),
# so the packaged code is built from the git-tracked files instead, minus the
# model weights — those travel as input channels, not as code.
CODE_EXCLUDED_SUFFIXES = ('.pth', '.pt', '.ckpt', '.h5', '.hdf5', '.pkl')


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def build_job_spec(config: RunConfig, config_uri: str, fold: Optional[int],
                   phase: str = 'finetune',
                   extra_inputs: Optional[Dict[str, str]] = None,
                   source_dir: str = '') -> SageMakerJobSpec:
    """Build the :class:`SageMakerJobSpec` for one job (a fold, or the whole run)."""
    sm = config.sagemaker
    tags = dict(sm.tags)
    tags.setdefault('project', config.clearml.project_name or 'LaBraM')
    tags['phase'] = phase
    if fold is not None:
        tags['cv_fold'] = str(fold)
    inputs = {'config': config_uri, **(extra_inputs or {})}
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


def plan_jobs(config: RunConfig, config_uri: str, phase: str = 'finetune',
              extra_inputs: Optional[Dict[str, str]] = None,
              source_dir: str = '') -> List[JobPlan]:
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
        spec = build_job_spec(config, config_uri, fold, phase, extra_inputs, source_dir)
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
                              source_dir: str) -> str:
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
        uri = _upload_data(session, tar_path, f'{prefix}/source', extra_args)
    logger.info("Uploaded source dir -> %s", uri)
    return uri


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


def submit(config: RunConfig, dry_run: bool = False, phase: str = 'finetune') -> List[JobPlan]:
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
        plans = plan_jobs(config, placeholder, phase, staged.resolved(pending))
        for p in plans:
            logger.info("[dry-run] phase=%s job=%s fold=%s inputs=%s hp=%s", phase,
                        p.job_name, p.fold, p.spec.inputs, p.spec.hyperparameters)
        return plans

    # Make ClearML credentials available in-container before the config/env is
    # packaged, so clearml.enabled runs can log from inside SageMaker.
    forward_clearml_env(config)

    launcher = SageMakerLauncher(region=sm.region or None, default_role=sm.role)
    # Fail fast with an actionable message if no usable execution role.
    role = launcher.resolve_role(sm.role)
    logger.info("SageMaker execution role: %s", role)

    try:
        uploaded = upload_staged_weights(launcher, config, staged)
        config_uri = upload_run_config(launcher, config, sm.config_channel)
        # Package + upload the code ourselves and pass the estimator an s3://
        # sourcedir.tar.gz, so it skips its own KMS-inheriting code upload.
        with staged_source_dir(config) as source_dir:
            source_uri = package_and_upload_source(launcher, config, source_dir)
        plans = plan_jobs(config, config_uri, phase, staged.resolved(uploaded), source_uri)
        if plans:
            # Log the training image that will actually be used (verify it resolves).
            try:
                logger.info("SageMaker training image: %s",
                            launcher.resolve_image_uri(plans[0].spec))
            except Exception as exc:  # pragma: no cover - depends on SDK/region
                logger.warning("Could not resolve training image URI: %s", exc)
        for p in plans:
            logger.info("Submitting SageMaker job %s (fold=%s)", p.job_name, p.fold)
            # Only the final job blocks when wait is requested, so earlier folds are
            # dispatched without waiting on each other.
            wait = sm.wait and (p is plans[-1])
            launcher.submit(p.spec, wait=wait)
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
    plans = submit(config, dry_run=cli.dry_run, phase=cli.phase)
    for p in plans:
        print(f"{'[dry-run] ' if cli.dry_run else ''}{p.job_name} (fold={p.fold})")


if __name__ == '__main__':
    main()
