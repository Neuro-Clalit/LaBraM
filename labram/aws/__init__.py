# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# AWS helpers. ``sagemaker`` holds a dependency-light wrapper around the SageMaker
# Python SDK for submitting LaBraM training runs (a single fine-tune, or one job
# per cross-validation fold) as managed training jobs. Vendored from the shared
# `common` repo, mirroring labram/file_system. See docs/sagemaker.md.
# ---------------------------------------------------------

from labram.aws.sagemaker import (
    SageMakerJobSpec,
    SageMakerLauncher,
    estimator_kwargs,
)

__all__ = [
    'SageMakerJobSpec',
    'SageMakerLauncher',
    'estimator_kwargs',
]
