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
from typing import Any, Dict, Optional


@dataclass
class SageMakerJobSpec:
    """Everything needed to launch one SageMaker PyTorch training job.

    ``inputs`` maps an input-channel name to an S3 uri; each channel is mounted
    in-container at ``/opt/ml/input/data/<channel>``. ``hyperparameters`` are
    passed to the entry point as ``--key value`` CLI args by the SDK.
    """

    entry_point: str
    source_dir: str = ""
    role: str = ""
    instance_type: str = "ml.g4dn.xlarge"
    instance_count: int = 1
    volume_size_gb: int = 100
    max_run_sec: int = 4 * 24 * 60 * 60
    use_spot: bool = False
    max_wait_sec: int = 0
    framework_version: str = "2.4.1"
    py_version: str = "py311"
    image_uri: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    inputs: Dict[str, str] = field(default_factory=dict)
    output_path: str = ""
    code_location: str = ""
    base_job_name: str = "training-job"


def estimator_kwargs(spec: SageMakerJobSpec) -> Dict[str, Any]:
    """Translate a :class:`SageMakerJobSpec` into ``sagemaker.pytorch.PyTorch``
    keyword arguments. Pure — no SDK import, no network — so it is unit-testable.

    An explicit ``image_uri`` takes precedence over the managed-DLC
    ``framework_version``/``py_version`` selectors (the SDK rejects both at once).
    Managed spot training sets ``max_wait`` (falling back to ``max_run_sec``).
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
    if spec.environment:
        kwargs["environment"] = dict(spec.environment)
    if spec.output_path:
        kwargs["output_path"] = spec.output_path
    if spec.code_location:
        kwargs["code_location"] = spec.code_location
    if spec.tags:
        kwargs["tags"] = [{"Key": k, "Value": v} for k, v in spec.tags.items()]
    if spec.use_spot:
        kwargs["use_spot_instances"] = True
        kwargs["max_wait"] = spec.max_wait_sec or spec.max_run_sec
    return kwargs


class SageMakerLauncher:
    """Builds and submits SageMaker PyTorch estimators from :class:`SageMakerJobSpec`."""

    def __init__(self, region: Optional[str] = None, sagemaker_session: Any = None,
                 default_role: str = ""):
        self._region = region or None
        self._session = sagemaker_session
        self._default_role = default_role

    # -- lazy SDK access ---------------------------------------------------

    def _get_session(self):
        if self._session is not None:
            return self._session
        import boto3
        import sagemaker
        boto_session = boto3.Session(region_name=self._region) if self._region else boto3.Session()
        self._session = sagemaker.Session(boto_session=boto_session)
        return self._session

    def resolve_role(self, role: str = "") -> str:
        role = role or self._default_role
        if role:
            return role
        import sagemaker
        return sagemaker.get_execution_role(sagemaker_session=self._get_session())

    # -- build / submit ----------------------------------------------------

    def build_estimator(self, spec: SageMakerJobSpec):
        """Instantiate a ``sagemaker.pytorch.PyTorch`` estimator for ``spec``."""
        from sagemaker.pytorch import PyTorch
        kwargs = estimator_kwargs(spec)
        kwargs["role"] = self.resolve_role(spec.role)
        kwargs["sagemaker_session"] = self._get_session()
        return PyTorch(**kwargs)

    def submit(self, spec: SageMakerJobSpec, wait: bool = False,
               job_name: Optional[str] = None) -> str:
        """Launch the training job; returns the (possibly SDK-generated) job name."""
        estimator = self.build_estimator(spec)
        fit_kwargs: Dict[str, Any] = {"wait": wait}
        if spec.inputs:
            fit_kwargs["inputs"] = dict(spec.inputs)
        if job_name:
            fit_kwargs["job_name"] = job_name
        estimator.fit(**fit_kwargs)
        return estimator.latest_training_job.name
