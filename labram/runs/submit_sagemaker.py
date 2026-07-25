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
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import labram.utils as utils
from labram.aws.sagemaker import SageMakerJobSpec, SageMakerLauncher
from labram.configs.run_configs import FinetuneRunConfig
from labram.configs.utils_conf import parse_overrides

logger = utils.get_logger(__name__)

# Where SageMaker mounts the ``config`` input channel inside the container.
CONFIG_CHANNEL_MOUNT = '/opt/ml/input/data/config'

# ClearML credential env vars forwarded into the training container so that
# clearml.enabled runs can talk to the ClearML server from inside SageMaker.
CLEARML_ENV_VARS = (
    'CLEARML_API_ACCESS_KEY', 'CLEARML_API_SECRET_KEY', 'CLEARML_API_HOST',
    'CLEARML_WEB_HOST', 'CLEARML_FILES_HOST',
)


def forward_clearml_env(config: FinetuneRunConfig) -> Dict[str, str]:
    """When ClearML tracking is on, copy the submitter's ``CLEARML_*`` credential
    env vars into ``sagemaker.environment`` (without overwriting explicit ones) so
    the in-container run can log to the same ClearML server. Returns the names
    forwarded. No-op when clearml is disabled."""
    if not config.clearml.enabled:
        return {}
    env = config.sagemaker.environment
    forwarded = {}
    for name in CLEARML_ENV_VARS:
        if name not in env and os.environ.get(name):
            env[name] = os.environ[name]
            forwarded[name] = env[name]
    if forwarded:
        logger.info("Forwarding %d ClearML credential var(s) to the SageMaker job: %s",
                    len(forwarded), sorted(forwarded))
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


def build_hyperparameters(config_uri: str, fold: Optional[int]) -> Dict[str, Any]:
    hp: Dict[str, Any] = {'config': container_config_path(config_uri)}
    if fold is not None:
        hp['fold'] = fold
    return hp


def build_job_spec(config: FinetuneRunConfig, config_uri: str,
                   fold: Optional[int]) -> SageMakerJobSpec:
    """Build the :class:`SageMakerJobSpec` for one job (a fold, or the whole run)."""
    sm = config.sagemaker
    tags = dict(sm.tags)
    tags.setdefault('project', config.clearml.project_name or 'LaBraM')
    if fold is not None:
        tags['cv_fold'] = str(fold)
    return SageMakerJobSpec(
        entry_point=sm.entry_point,
        source_dir=sm.source_dir or '.',
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
        hyperparameters={**sm.hyperparameters, **build_hyperparameters(config_uri, fold)},
        environment=dict(sm.environment),
        tags=tags,
        inputs={'config': config_uri},
        output_path=sm.output_path,
        code_location=sm.code_location,
        base_job_name=fold_job_name(sm.job_name_prefix, fold),
    )


@dataclass
class JobPlan:
    fold: Optional[int]
    job_name: str
    spec: SageMakerJobSpec


def plan_jobs(config: FinetuneRunConfig, config_uri: str) -> List[JobPlan]:
    """Enumerate the jobs to submit without touching AWS.

    Cross-validation -> one job per fold (or a single fold when
    ``cross_validation.fold >= 0``); otherwise a single fine-tune job.
    """
    cv = config.cross_validation
    if cv.enabled:
        if cv.fold is not None and cv.fold >= 0:
            folds: List[Optional[int]] = [cv.fold]
        else:
            folds = list(range(cv.n_folds))
    else:
        folds = [None]

    plans: List[JobPlan] = []
    for fold in folds:
        spec = build_job_spec(config, config_uri, fold)
        plans.append(JobPlan(fold=fold, job_name=spec.base_job_name, spec=spec))
    return plans


# ------------------------------------------------------------------ submit


def upload_run_config(launcher: SageMakerLauncher, config: FinetuneRunConfig,
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
        uri = session.upload_data(local, key_prefix=key_prefix)
    logger.info("Uploaded run config -> %s", uri)
    return uri


def submit(config: FinetuneRunConfig, dry_run: bool = False) -> List[JobPlan]:
    """Submit the planned SageMaker job(s). With ``dry_run`` only the plan is
    returned (no AWS calls, no SDK import), which is what the tests exercise."""
    sm = config.sagemaker
    if dry_run:
        placeholder = sm.config_channel or 's3://<bucket>/<prefix>/run_config.yaml'
        plans = plan_jobs(config, placeholder)
        for p in plans:
            logger.info("[dry-run] job=%s fold=%s hp=%s", p.job_name, p.fold,
                        p.spec.hyperparameters)
        return plans

    # Make ClearML credentials available in-container before the config/env is
    # packaged, so clearml.enabled runs can log from inside SageMaker.
    forward_clearml_env(config)

    launcher = SageMakerLauncher(region=sm.region or None, default_role=sm.role)
    # Fail fast with an actionable message if no usable execution role.
    role = launcher.resolve_role(sm.role)
    logger.info("SageMaker execution role: %s", role)

    config_uri = upload_run_config(launcher, config, sm.config_channel)
    plans = plan_jobs(config, config_uri)
    if plans:
        # Log the training image that will actually be used (verify it resolves).
        try:
            logger.info("SageMaker training image: %s", launcher.resolve_image_uri(plans[0].spec))
        except Exception as exc:  # pragma: no cover - depends on SDK/region
            logger.warning("Could not resolve training image URI: %s", exc)
    for p in plans:
        logger.info("Submitting SageMaker job %s (fold=%s)", p.job_name, p.fold)
        # Only the final job blocks when wait is requested, so earlier folds are
        # dispatched without waiting on each other.
        wait = sm.wait and (p is plans[-1])
        launcher.submit(p.spec, wait=wait)
    logger.info("Submitted %d SageMaker job(s).", len(plans))
    return plans


# ------------------------------------------------------------------ cli


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Submit LaBraM fine-tuning to AWS SageMaker')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--set', dest='overrides', nargs='*', default=[], metavar='KEY=VALUE')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print the job plan without contacting AWS.')
    return parser.parse_args()


def main() -> None:
    cli = parse_cli()
    overrides = parse_overrides(cli.overrides)
    config = FinetuneRunConfig.load_config(cli.config, **overrides)
    if not config.sagemaker.enabled and not cli.dry_run:
        raise SystemExit(
            "sagemaker.enabled is false; set it (e.g. --set sagemaker.enabled=true) "
            "or use --dry_run to preview the job plan.")
    plans = submit(config, dry_run=cli.dry_run)
    for p in plans:
        print(f"{'[dry-run] ' if cli.dry_run else ''}{p.job_name} (fold={p.fold})")


if __name__ == '__main__':
    main()
