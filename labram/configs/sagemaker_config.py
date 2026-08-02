"""AWS SageMaker training-job submission configuration.

``SageMakerConfig`` describes *how* to dispatch a LaBraM training run (a single
fine-tune, or one job per cross-validation fold) as a managed SageMaker training
job. It is consumed by :mod:`labram.aws.sagemaker` /
``labram.runs.submit_sagemaker`` on the submitting machine; it never affects the
in-container training itself, so it is off by default.
"""

from dataclasses import dataclass, field
from typing import Dict

from labram.configs.base_configs import ConfigBase
from labram.configs.defaults import (
    DEFAULT_SAGEMAKER_CODE_LOCATION,
    DEFAULT_SAGEMAKER_CONFIG_CHANNEL,
    DEFAULT_SAGEMAKER_ENABLED,
    DEFAULT_SAGEMAKER_ENTRY_POINT,
    DEFAULT_SAGEMAKER_FRAMEWORK_VERSION,
    DEFAULT_SAGEMAKER_IMAGE_URI,
    DEFAULT_SAGEMAKER_INPUT_MODE,
    DEFAULT_SAGEMAKER_INSTANCE_COUNT,
    DEFAULT_SAGEMAKER_INSTANCE_TYPE,
    DEFAULT_SAGEMAKER_JOB_NAME_PREFIX,
    DEFAULT_SAGEMAKER_MAX_RUN_SEC,
    DEFAULT_SAGEMAKER_MAX_WAIT_SEC,
    DEFAULT_SAGEMAKER_OUTPUT_PATH,
    DEFAULT_SAGEMAKER_PY_VERSION,
    DEFAULT_SAGEMAKER_REGION,
    DEFAULT_SAGEMAKER_ROLE,
    DEFAULT_SAGEMAKER_SOURCE_DIR,
    DEFAULT_SAGEMAKER_USE_SPOT,
    DEFAULT_SAGEMAKER_VOLUME_SIZE_GB,
    DEFAULT_SAGEMAKER_WAIT,
    DEFAULT_SAGEMAKER_WEIGHT_S3_URIS,
)


@dataclass
class SageMakerConfig(ConfigBase):
    """SageMaker training-job launch settings.

    Attributes:
        enabled: Master switch (only consulted by the submit entry point).
        role: SageMaker execution role ARN. Empty -> resolved on the submitting
            machine via ``sagemaker.get_execution_role()``.
        instance_type / instance_count / volume_size_gb: Compute for each job.
        max_run_sec: Hard wall-clock cap per training job.
        use_spot / max_wait_sec: Managed spot training and its max wait (0 ->
            reuse ``max_run_sec`` when spot is enabled).
        framework_version / py_version: Managed PyTorch DLC selectors.
        image_uri: Explicit training image; empty -> managed DLC for
            ``framework_version``.
        entry_point / source_dir: Training script (relative to ``source_dir``)
            and the code directory packaged & uploaded by the SDK. Empty
            ``source_dir`` -> repo root.
        job_name_prefix: Prefix for generated job names; the fold number is
            appended for CV (``<prefix>-fold-<k>``).
        region: AWS region; empty -> boto3 default.
        output_path / code_location: S3 prefixes for model artifacts and the
            packaged source.
        config_channel: S3 uri of the run config uploaded as the ``config``
            input channel (mounted at ``/opt/ml/input/data/config`` in-container).
        weight_s3_uris: ``{local_weight_path: s3_uri}`` mirrors for weight files
            whose bytes already live in S3. A weight field
            (``finetune_checkpoint.finetune`` / ``model.codebook_reg.tokenizer_weight``)
            pointing at one of these local paths is served from the S3 mirror as
            an input channel instead of being re-uploaded on every submission.
            Keys are matched by normalized path. Clear it (``{}``) to force the
            local file to upload; add entries for your own weights. Never affects
            local (non-SageMaker) training.
        input_mode: How every input channel is delivered — ``File`` (download to
            the EBS volume first), ``FastFile`` (stream from S3 through a FUSE
            mount; recommended for the many-small-files TUH corpora) or ``Pipe``.
        environment: Extra environment variables for the training container.
        hyperparameters: Extra hyperparameters merged into every job.
        tags: ``{Key: Value}`` tags applied to each job.
        wait: Block until the (last) job finishes.
    """

    enabled: bool = DEFAULT_SAGEMAKER_ENABLED
    role: str = DEFAULT_SAGEMAKER_ROLE
    instance_type: str = DEFAULT_SAGEMAKER_INSTANCE_TYPE
    instance_count: int = DEFAULT_SAGEMAKER_INSTANCE_COUNT
    volume_size_gb: int = DEFAULT_SAGEMAKER_VOLUME_SIZE_GB
    max_run_sec: int = DEFAULT_SAGEMAKER_MAX_RUN_SEC
    use_spot: bool = DEFAULT_SAGEMAKER_USE_SPOT
    max_wait_sec: int = DEFAULT_SAGEMAKER_MAX_WAIT_SEC
    framework_version: str = DEFAULT_SAGEMAKER_FRAMEWORK_VERSION
    py_version: str = DEFAULT_SAGEMAKER_PY_VERSION
    image_uri: str = DEFAULT_SAGEMAKER_IMAGE_URI
    entry_point: str = DEFAULT_SAGEMAKER_ENTRY_POINT
    source_dir: str = DEFAULT_SAGEMAKER_SOURCE_DIR
    job_name_prefix: str = DEFAULT_SAGEMAKER_JOB_NAME_PREFIX
    region: str = DEFAULT_SAGEMAKER_REGION
    output_path: str = DEFAULT_SAGEMAKER_OUTPUT_PATH
    code_location: str = DEFAULT_SAGEMAKER_CODE_LOCATION
    config_channel: str = DEFAULT_SAGEMAKER_CONFIG_CHANNEL
    weight_s3_uris: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_SAGEMAKER_WEIGHT_S3_URIS))
    input_mode: str = DEFAULT_SAGEMAKER_INPUT_MODE
    environment: Dict[str, str] = field(default_factory=dict)
    hyperparameters: Dict[str, str] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    wait: bool = DEFAULT_SAGEMAKER_WAIT
