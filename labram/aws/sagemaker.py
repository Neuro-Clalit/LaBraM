"""Generic AWS SageMaker training-job submission helpers.

Vendored from the shared ``common`` repo (mirroring ``labram/file_system``) so
LaBraM stays self-contained. The LaBraM-specific layer that builds specs from a
:class:`~labram.configs.run_configs.FinetuneRunConfig` lives in
``labram.runs.submit_sagemaker``.

A thin, dependency-light wrapper around the ``sagemaker`` Python SDK that turns a
declarative :class:`SageMakerJobSpec` into a managed PyTorch training job. The
spec -> estimator-kwargs mapping (:func:`estimator_kwargs`) is a pure function so
it can be unit-tested without the SDK or AWS credentials installed; the SDK is
imported lazily inside :class:`SageMakerLauncher` only when a job is actually
launched.

Typical use (from a submitting machine with AWS credentials)::

    spec = SageMakerJobSpec(
        entry_point='train.py', source_dir='.', role='arn:aws:iam::...:role/SM',
        instance_type='ml.g4dn.xlarge',
        hyperparameters={'config': '/opt/ml/input/data/config/run.yaml'},
        inputs={'config': 's3://bucket/run.yaml'},
        base_job_name='my-training',
    )
    launcher = SageMakerLauncher(region='eu-west-1')
    job_name = launcher.submit(spec, wait=False)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class SageMakerJobSpec:
    """Everything needed to launch one SageMaker PyTorch training job.

    ``inputs`` maps an input-channel name to an S3 uri; each channel is mounted
    in-container at ``/opt/ml/input/data/<channel>``. ``input_mode`` selects how
    they are delivered (``File`` / ``FastFile`` / ``Pipe``); empty keeps the SDK
    default (``File``). ``channel_input_modes`` overrides the mode for individual
    channels (``{channel: mode}``) — needed because ``FastFile``/``Pipe`` treat an
    S3 uri as a *prefix* and expose only the keys beneath it, so a channel whose
    uri is a single object must be delivered as ``File``. ``hyperparameters`` are
    passed to the entry point as ``--key value`` CLI args by the SDK.
    """

    entry_point: str
    source_dir: str = ""
    role: str = ""
    instance_type: str = "ml.g4dn.xlarge"
    instance_count: int = 1
    volume_size_gb: int = 100
    max_run_sec: int = 24 * 60 * 60      # SageMaker stops the job at this cap
    use_spot: bool = False
    max_wait_min: float = 0.0
    framework_version: str = "2.4.1"
    py_version: str = "py311"
    image_uri: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    inputs: Dict[str, str] = field(default_factory=dict)
    input_mode: str = ""
    channel_input_modes: Dict[str, str] = field(default_factory=dict)
    output_path: str = ""
    code_location: str = ""
    output_kms_key: str = ""
    base_job_name: str = "training-job"


def estimator_kwargs(spec: SageMakerJobSpec) -> Dict[str, Any]:
    """Translate a :class:`SageMakerJobSpec` into ``sagemaker.pytorch.PyTorch``
    keyword arguments. Pure — no SDK import, no network — so it is unit-testable.

    An explicit ``image_uri`` takes precedence over the managed-DLC
    ``framework_version``/``py_version`` selectors (the SDK rejects both at once).
    Managed spot training sets ``max_wait`` in seconds from ``max_wait_min``
    (falling back to ``max_run_sec``).
    An empty ``input_mode`` is omitted so the SDK applies its own default.
    """
    kwargs: Dict[str, Any] = {
        "entry_point": spec.entry_point,
        "role": spec.role or None,
        "instance_type": spec.instance_type,
        "instance_count": spec.instance_count,
        "volume_size": spec.volume_size_gb,
        "max_run": spec.max_run_sec,
        "base_job_name": spec.base_job_name,
        "hyperparameters": dict(spec.hyperparameters),
    }
    if spec.source_dir:
        kwargs["source_dir"] = spec.source_dir
    if spec.image_uri:
        kwargs["image_uri"] = spec.image_uri
    else:
        kwargs["framework_version"] = spec.framework_version
        kwargs["py_version"] = spec.py_version
    if spec.input_mode:
        kwargs["input_mode"] = spec.input_mode
    if spec.environment:
        kwargs["environment"] = dict(spec.environment)
    if spec.output_path:
        kwargs["output_path"] = spec.output_path
    if spec.code_location:
        kwargs["code_location"] = spec.code_location
    if spec.output_kms_key:
        kwargs["output_kms_key"] = spec.output_kms_key
    if spec.tags:
        kwargs["tags"] = [{"Key": k, "Value": v} for k, v in spec.tags.items()]
    if spec.use_spot:
        kwargs["use_spot_instances"] = True
        kwargs["max_wait"] = (int(spec.max_wait_min * 60)
                              if spec.max_wait_min > 0 else spec.max_run_sec)
    return kwargs


def role_account(role_arn: str) -> str:
    """The AWS account id embedded in an IAM role ARN; ``''`` if unparseable.

    ``arn:aws:iam::123456789012:role/Name`` -> ``'123456789012'``.
    """
    parts = role_arn.split(':')
    if len(parts) >= 6 and parts[0] == 'arn' and parts[2] == 'iam':
        return parts[4]
    return ''


class SageMakerLauncher:
    """Builds and submits SageMaker PyTorch estimators from :class:`SageMakerJobSpec`."""

    def __init__(self, region: Optional[str] = None, sagemaker_session: Any = None,
                 default_role: str = "", profile: Optional[str] = None):
        self._region = region or None
        self._session = sagemaker_session
        self._default_role = default_role
        self._profile = profile or None

    # -- lazy SDK access ---------------------------------------------------

    def _get_session(self):
        if self._session is not None:
            return self._session
        import boto3
        try:
            import sagemaker
        except ImportError as exc:
            raise ImportError(
                "The `sagemaker` SDK is required to submit training jobs but is "
                "not installed. Install it with "
                "`pip install -r requirements-sagemaker.txt`, or preview the plan "
                "with --dry_run, which needs neither the SDK nor AWS credentials."
            ) from exc
        # An explicit profile pins which account the job is submitted from; empty
        # leaves boto3 to its own resolution (AWS_PROFILE / 'default' / instance role).
        kwargs = {}
        if self._profile:
            kwargs['profile_name'] = self._profile
        if self._region:
            kwargs['region_name'] = self._region
        boto_session = boto3.Session(**kwargs)
        self._session = sagemaker.Session(boto_session=boto_session)
        return self._session

    def caller_identity(self) -> Dict[str, str]:
        """``sts:GetCallerIdentity`` for the submitting credentials.

        Returns ``{'Account': ..., 'Arn': ..., 'UserId': ...}``, or ``{}`` when
        STS cannot be reached. Never raises: this is used for a *diagnostic*
        preflight, and a credential problem should surface on the real API call
        with its own error rather than here.
        """
        try:
            boto_session = self._get_session().boto_session
            return dict(boto_session.client('sts').get_caller_identity())
        except Exception:  # pragma: no cover - depends on live credentials
            return {}

    def resolve_role(self, role: str = "") -> str:
        role = role or self._default_role
        if role:
            return role
        # get_execution_role() only works where an execution role is attached
        # (a SageMaker notebook / training job). On a plain EC2 box or laptop it
        # raises, so surface an actionable message: an explicit role is required.
        import sagemaker
        try:
            return sagemaker.get_execution_role(sagemaker_session=self._get_session())
        except Exception as exc:  # ValueError / ClientError depending on context
            raise ValueError(
                "No SageMaker execution role provided and get_execution_role() "
                "failed (you are not inside a SageMaker-managed environment). "
                "Set an IAM role ARN with SageMaker + S3 permissions (e.g. "
                "--set sagemaker.role=arn:aws:iam::<acct>:role/<name>)."
            ) from exc

    def resolve_image_uri(self, spec: SageMakerJobSpec) -> str:
        """The training image that will actually be used: an explicit
        ``image_uri`` when set, otherwise the managed PyTorch DLC the SDK resolves
        for ``(framework_version, py_version, instance_type)``. Useful to log/verify
        the image before launching."""
        if spec.image_uri:
            return spec.image_uri
        from sagemaker import image_uris
        session = self._get_session()
        region = self._region or getattr(session, "boto_region_name", None)
        return image_uris.retrieve(
            framework="pytorch", region=region, version=spec.framework_version,
            py_version=spec.py_version, instance_type=spec.instance_type,
            image_scope="training")

    # -- build / submit ----------------------------------------------------

    def build_estimator(self, spec: SageMakerJobSpec):
        """Instantiate a ``sagemaker.pytorch.PyTorch`` estimator for ``spec``."""
        from sagemaker.pytorch import PyTorch
        kwargs = estimator_kwargs(spec)
        kwargs["role"] = self.resolve_role(spec.role)
        kwargs["sagemaker_session"] = self._get_session()
        return PyTorch(**kwargs)

    def build_inputs(self, spec: SageMakerJobSpec) -> Dict[str, Any]:
        """The ``fit(inputs=...)`` mapping: a plain uri per channel, or a
        ``TrainingInput`` for channels that override the job-level input mode."""
        modes = {c: m for c, m in (spec.channel_input_modes or {}).items() if m}
        if not modes:
            return dict(spec.inputs)
        from sagemaker.inputs import TrainingInput
        return {
            channel: (TrainingInput(s3_data=uri, input_mode=modes[channel])
                      if channel in modes else uri)
            for channel, uri in spec.inputs.items()
        }

    def submit(self, spec: SageMakerJobSpec, wait: bool = False,
               job_name: Optional[str] = None, stream_logs: bool = True,
               on_submitted: Optional[Callable[[str], None]] = None) -> str:
        """Launch the training job; returns the (possibly SDK-generated) job name.

        The job is always *created* without blocking so its real name is known
        immediately; ``on_submitted(name)`` is then called (the caller uses it to
        report the job before any waiting), and only afterwards does ``wait``
        block on completion. ``stream_logs=False`` waits without pulling the
        container's CloudWatch log stream into this process.
        """
        estimator = self.build_estimator(spec)
        fit_kwargs: Dict[str, Any] = {"wait": False}
        if spec.inputs:
            fit_kwargs["inputs"] = self.build_inputs(spec)
        if job_name:
            fit_kwargs["job_name"] = job_name
        estimator.fit(**fit_kwargs)
        name = estimator.latest_training_job.name
        if on_submitted is not None:
            on_submitted(name)
        if wait:
            estimator.latest_training_job.wait(logs="All" if stream_logs else "None")
        return name
