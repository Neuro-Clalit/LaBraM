"""Named constants for every default value used as an init/argparse argument
across the runner scripts, the optimizer factory, and the data-processor.

Single source of truth — config dataclass fields reference these constants
instead of re-stating literals. Inventory was swept from:

* labram/runs/run_pretrain.py argparse + main()
* labram/runs/run_vqnsp.py argparse + main()
* labram/runs/finetune_args.py argparse
* labram/runs/finetune_setup.py (debug overrides)
* labram/optim_factory.py (per-optimizer hyperparam fallbacks)
* labram/data/hdf5_datasets.py SingleShockDataset.__init__
* labram/utils/training.py cosine_scheduler
"""

from typing import Dict

# ---------- Trainer ----------
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_PRETRAIN_EPOCHS: int = 300
DEFAULT_VQNSP_EPOCHS: int = 100
DEFAULT_FINETUNE_EPOCHS: int = 30
DEFAULT_PRETRAIN_SAVE_CKPT_FREQ: int = 20
DEFAULT_VQNSP_SAVE_CKPT_FREQ: int = 20
DEFAULT_FINETUNE_SAVE_CKPT_FREQ: int = 5
DEFAULT_START_EPOCH: int = 0
DEFAULT_GRADIENT_ACCUMULATION_STEPS: int = 1
DEFAULT_UPDATE_FREQ: int = 1  # finetune update_freq

# ---------- Backbone / model (shared) ----------
DEFAULT_LAYER_SCALE_INIT_VALUE: float = 0.1   # 0.1 base, 1e-5 large; 0 disables
DEFAULT_DROP_PATH: float = 0.1
DEFAULT_DROP: float = 0.0                     # finetune dropout
DEFAULT_ATTN_DROP_RATE: float = 0.0

# ---------- Pretrain model ----------
DEFAULT_PRETRAIN_MODEL: str = 'labram_base_patch200_1600_8k_vocab'
DEFAULT_PRETRAIN_INPUT_SIZE: int = 1600
DEFAULT_PRETRAIN_REL_POS_BIAS: bool = False
DEFAULT_PRETRAIN_ABS_POS_EMB: bool = True

# ---------- VQNSP model ----------
DEFAULT_VQNSP_MODEL: str = 'vqnsp_encoder_base_decoder_3x200x12'
DEFAULT_VQNSP_INPUT_SIZE: int = 1600
DEFAULT_VQNSP_EMA_DECAY: float = 0.99
DEFAULT_VQNSP_QUANTIZE_KMEANS_INIT: bool = False

# ---------- Tokenizer (shared between pretrain + vqnsp model heads) ----------
DEFAULT_TOKENIZER_MODEL: str = 'vqnsp_encoder_base_decoder_3x200x12'
DEFAULT_TOKENIZER_WEIGHT: str = ''
DEFAULT_CODEBOOK_SIZE: int = 8192
DEFAULT_QUANTIZER_DIM: int = 32

# ---------- Codebook-regularized fine-tuning ----------
DEFAULT_FEATURES_EMB_DIM: int = 128

# ---------- Finetune model ----------
DEFAULT_FINETUNE_MODEL: str = 'labram_base_patch200_200'
DEFAULT_FINETUNE_INPUT_SIZE: int = 200
DEFAULT_FINETUNE_QKV_BIAS: bool = True
DEFAULT_FINETUNE_REL_POS_BIAS: bool = True
DEFAULT_FINETUNE_ABS_POS_EMB: bool = False
DEFAULT_FINETUNE_USE_MEAN_POOLING: bool = True
DEFAULT_FINETUNE_INIT_SCALE: float = 0.001
DEFAULT_FINETUNE_MODEL_KEY: str = 'model|module'
DEFAULT_FINETUNE_MODEL_PREFIX: str = ''
DEFAULT_FINETUNE_MODEL_FILTER_NAME: str = 'gzp'
DEFAULT_FINETUNE_CHECKPOINT: str = ''
DEFAULT_FINETUNE_NB_CLASSES: int = 0
DEFAULT_FINETUNE_TASK: str = 'classification'
DEFAULT_FINETUNE_LABEL_SMOOTHING: float = 0.1
DEFAULT_FINETUNE_MODEL_EMA: bool = False
DEFAULT_FINETUNE_MODEL_EMA_DECAY: float = 0.9999
DEFAULT_FINETUNE_MODEL_EMA_FORCE_CPU: bool = False
DEFAULT_FINETUNE_DISABLE_EVAL: bool = False
DEFAULT_FINETUNE_DISABLE_WD_ON_REL_POS_BIAS: bool = False
DEFAULT_FINETUNE_RANDOM_ERASE_PROB: float = 0.25
DEFAULT_FINETUNE_RANDOM_ERASE_MODE: str = 'pixel'
DEFAULT_FINETUNE_RANDOM_ERASE_COUNT: int = 1
DEFAULT_FINETUNE_RANDOM_ERASE_SPLIT: bool = False
DEFAULT_FINETUNE_DATASET: str = 'TUAB'
DEFAULT_FINETUNE_DATA_PATH: str = ''
DEFAULT_FINETUNE_DEBUG: bool = False
DEFAULT_FINETUNE_DEBUG_SAMPLES: int = 16
DEFAULT_FINETUNE_ENABLE_DEEPSPEED: bool = False
DEFAULT_FINETUNE_ROBUST_TEST: str = ''

# ---------- Optimizer ----------
DEFAULT_OPTIMIZER: str = 'adamw'
DEFAULT_OPT_EPS: float = 1e-8
DEFAULT_MOMENTUM: float = 0.9
DEFAULT_WEIGHT_DECAY: float = 0.05
DEFAULT_VQNSP_WEIGHT_DECAY: float = 1e-4
DEFAULT_PRETRAIN_LR: float = 5e-4
DEFAULT_FINETUNE_LR: float = 5e-4
DEFAULT_VQNSP_LR: float = 5e-5
DEFAULT_WARMUP_LR: float = 1e-6
DEFAULT_MIN_LR: float = 1e-5
DEFAULT_FINETUNE_MIN_LR: float = 1e-6
DEFAULT_WARMUP_EPOCHS: int = 5
DEFAULT_WARMUP_STEPS: int = -1
DEFAULT_FINETUNE_LAYER_DECAY: float = 0.9
# create_optimizer / nested optimizer defaults
DEFAULT_RMSPROP_ALPHA: float = 0.9
DEFAULT_ADAMP_WD_RATIO: float = 0.01
DEFAULT_FUSED_NOVOGRAD_BETAS: tuple = (0.95, 0.98)

# ---------- Data ----------
DEFAULT_NUM_WORKERS: int = 10
DEFAULT_PIN_MEM: bool = True
# '' -> build the dataset's default train/val/test split. Otherwise reuse the
# exact case assignment recorded in this data_split.json (local path or s3://),
# e.g. to run several models on an identical split. Fine-tuning only.
DEFAULT_DATA_SPLIT_JSON: str = ''
# build_pretraining_dataset arguments
DEFAULT_PRETRAIN_STRIDE: int = 800        # run_pretrain.py
DEFAULT_VQNSP_STRIDE: int = 200           # run_vqnsp.py
DEFAULT_DATASET_DEFAULT_STRIDE: int = 200  # build_pretraining_dataset signature
DEFAULT_DATASET_START_PERCENTAGE: float = 0.0
DEFAULT_DATASET_END_PERCENTAGE: float = 1.0
DEFAULT_TIME_WINDOWS: tuple = (4, 8)
DEFAULT_EVAL_BATCH_SCALE: float = 1.5
# SingleShockDataset (data_processor)
DEFAULT_SINGLE_SHOCK_WINDOW_SIZE: int = 200
DEFAULT_SINGLE_SHOCK_STRIDE: int = 1

# ---------- Output / checkpoint ----------
DEFAULT_OUTPUT_DIR: str = ''
DEFAULT_LOG_DIR: str = ''     # '' rather than None so the YAML round-trips through safe_load
DEFAULT_RESUME: str = ''
DEFAULT_AUTO_RESUME: bool = True
DEFAULT_SAVE_CKPT: bool = True
# When True, skip the periodic/rolling per-epoch checkpoints and save only the
# final trained model (one file at the end of training).
DEFAULT_SAVE_ONLY_FINAL_MODEL: bool = False

# ---------- Logging / visualization (LoggingConfig) ----------
DEFAULT_LOG_MODEL_GRAPH: bool = True
DEFAULT_MODEL_GRAPH_FORMAT: str = 'svg'   # svg (vector) | png
DEFAULT_LOG_DATA_SPLIT: bool = True
# Relative (scale-free) metric logging. Per-component losses / gradient norms
# are reported as their share of the component total, and every metric is
# plotted against normalized training progress instead of the raw iteration
# count, so runs of different length / batch size overlay directly.
DEFAULT_RELATIVE_LOSS_COMPONENTS: bool = True
DEFAULT_RELATIVE_STEP_AXIS: bool = True
DEFAULT_RELATIVE_STEP_SCALE: int = 1000   # progress 0..1 -> 0..scale (per-mille)

# ---------- ClearML experiment tracking (labram/utils/logging.ClearMLLogger) ----------
DEFAULT_CLEARML_ENABLED: bool = False
DEFAULT_CLEARML_PROJECT_NAME: str = 'LaBraM'
DEFAULT_CLEARML_TASK_NAME: str = ''      # '' -> derived from output_dir / run at task-init time
# Append a millisecond-precision timestamp (YYYYmmdd_HHMMSS_fff) to the task name
# so every run is uniquely identifiable in the ClearML UI.
DEFAULT_CLEARML_APPEND_TIMESTAMP: bool = True
DEFAULT_CLEARML_OUTPUT_URI: str = ''     # '' -> ClearML default (no artifact upload target)
DEFAULT_CLEARML_OFFLINE: bool = False    # run without a ClearML server; store locally
DEFAULT_CLEARML_CONTINUE_LAST_TASK: bool = False
DEFAULT_CLEARML_AUTO_CONNECT_FRAMEWORKS: bool = True

# ---------- Distributed ----------
DEFAULT_WORLD_SIZE: int = 1
DEFAULT_LOCAL_RANK: int = -1
DEFAULT_DIST_ON_ITP: bool = False
DEFAULT_DIST_URL: str = 'env://'
DEFAULT_DEVICE: str = 'cuda'           # pretrain / vqnsp
DEFAULT_FINETUNE_DEVICE: str = 'auto'  # finetune resolves auto -> cuda/mps/cpu
DEFAULT_SEED: int = 0
DEFAULT_DIST_EVAL: bool = False        # finetune; vqnsp uses True separately
DEFAULT_VQNSP_DIST_EVAL: bool = True

# ---------- LR/WD scheduler (utils/training.cosine_scheduler) ----------
DEFAULT_COSINE_START_WARMUP_VALUE: float = 0.0
# Learning-rate schedule selector (labram/utils/training.build_lr_schedule).
# 'cosine' reproduces the historical behaviour; 'step'/'multistep'/'linear'/
# 'constant' are the alternatives, all sharing the same linear warmup prefix.
DEFAULT_LR_SCHED: str = 'cosine'
DEFAULT_LR_DECAY_EPOCHS: int = 30      # 'step': decay LR by decay_rate every N epochs
DEFAULT_LR_DECAY_RATE: float = 0.1     # multiplicative decay for 'step'/'multistep'

# ---------- Evaluation / detailed metrics / window aggregation ----------
# Detailed classification metrics (confusion matrix, ROC/PR curves, F1,
# sensitivity/specificity) reported for train/val/test; window aggregation for
# inference. See labram/utils/eval_metrics.py and labram/train/train_finetune.py.
DEFAULT_EVAL_DETAILED_METRICS: bool = True
DEFAULT_EVAL_LOG_CONFUSION_MATRIX: bool = True
DEFAULT_EVAL_LOG_CURVES: bool = True
DEFAULT_EVAL_LOG_GRAD_COMPONENTS: bool = False  # per-loss-component grad norms (extra backward)
DEFAULT_EVAL_LOG_GRAD_FREQ: int = 50            # steps between grad-component logging
DEFAULT_EVAL_AGG_WINDOWS: str = 'none'          # none|mean|vote|max
DEFAULT_EVAL_AGG_CASE_BY: str = 'recording'     # recording|subject

# ---------- Post-training shutdown (labram/utils/shutdown.py) ----------
DEFAULT_STOP_INSTANCE_ON_FINISH: bool = False
DEFAULT_STOP_DELAY_MINUTES: int = 5
DEFAULT_STOP_METHOD: str = 'ec2'                # ec2|os

# ---------- ClearML model artifact upload ----------
DEFAULT_CLEARML_UPLOAD_MODEL_ARTIFACT: bool = True
DEFAULT_CLEARML_ARTIFACT_NAME: str = ''         # '' -> derived from the run/task name

# ---------- Cross-validation fine-tuning (labram/data/cross_validation.py) ----------
DEFAULT_CV_ENABLED: bool = False
DEFAULT_CV_N_FOLDS: int = 5
# -1 -> the CV runner iterates every fold; >=0 -> run only that single fold (used
# when each fold is dispatched as its own process / SageMaker job).
DEFAULT_CV_FOLD: int = -1
# How a "group" is defined when partitioning folds so the same subject/recording
# never straddles train/val/test: subject|recording|window.
DEFAULT_CV_SPLIT_BY: str = 'subject'
DEFAULT_CV_SHUFFLE: bool = True
DEFAULT_CV_SEED: int = 42
# Which data pool is re-partitioned into folds: 'train_val' keeps the original
# test split untouched and only cross-validates over train+val; 'all' pools
# train+val+test. The held-out fold is the fold's test set; the next fold
# (cyclically) is its validation set; the remainder is training.
DEFAULT_CV_POOL: str = 'train_val'
# '' -> compute folds from (split_by, seed, shuffle); otherwise reuse the folds
# recorded in this cv_split.json so every fold job sees an identical partition.
DEFAULT_CV_SPLIT_JSON: str = ''
# Base experiment/output folder for the fold sub-runs. '' -> derived from the
# fine-tune output_dir. Each fold lives in ``<base>/fold_<k>``.
DEFAULT_CV_BASE_DIR: str = ''

# ---------- AWS SageMaker training-job submission (labram/aws/sagemaker.py) ----------
DEFAULT_SAGEMAKER_ENABLED: bool = False
DEFAULT_SAGEMAKER_ROLE: str = ''                # '' -> resolve via sagemaker.get_execution_role()
# GPU instance for training. ml.g5.2xlarge (1x A10G, 24 GB) matches the g5.2xl
# EC2 box used for local runs; ml.g5.xlarge is the cheaper single-GPU option.
DEFAULT_SAGEMAKER_INSTANCE_TYPE: str = 'ml.g5.2xlarge'
DEFAULT_SAGEMAKER_INSTANCE_COUNT: int = 1
DEFAULT_SAGEMAKER_VOLUME_SIZE_GB: int = 100
# Hard wall-clock cap per training job: SageMaker stops the job when it is hit,
# so this is the ceiling on what a single submission can cost if training hangs,
# diverges, or is simply slower than expected. 24h covers the LaBraM fine-tunes
# and keeps a forgotten GPU job from running for days; raise it explicitly
# (--set sagemaker.max_run_sec=...) for long pre-training runs.
DEFAULT_SAGEMAKER_MAX_RUN_SEC: int = 24 * 60 * 60   # 24 hours
DEFAULT_SAGEMAKER_USE_SPOT: bool = False
DEFAULT_SAGEMAKER_MAX_WAIT_SEC: int = 0         # 0 -> falls back to max_run_sec when spot is on
# Managed PyTorch Deep Learning Container selector. 2.4.0 + py311 is a published
# SageMaker training DLC (CUDA 12.4, compatible with g5/A10G); the SDK resolves
# the GPU image for the chosen instance type. requirements.txt does NOT pin torch,
# so the container keeps this DLC's torch build. (Note: 2.4.1 has no managed DLC.)
DEFAULT_SAGEMAKER_FRAMEWORK_VERSION: str = '2.4.0'
DEFAULT_SAGEMAKER_PY_VERSION: str = 'py311'
DEFAULT_SAGEMAKER_IMAGE_URI: str = ''           # '' -> managed PyTorch DLC for framework_version
DEFAULT_SAGEMAKER_ENTRY_POINT: str = 'labram/runs/sagemaker_entry.py'
DEFAULT_SAGEMAKER_SOURCE_DIR: str = ''          # '' -> repo root (packaged & uploaded by the SDK)
DEFAULT_SAGEMAKER_JOB_NAME_PREFIX: str = 'labram-finetune'
DEFAULT_SAGEMAKER_REGION: str = ''              # '' -> boto3 default region
DEFAULT_SAGEMAKER_OUTPUT_PATH: str = ''         # S3 prefix for model artifacts
DEFAULT_SAGEMAKER_CODE_LOCATION: str = ''       # S3 prefix for the packaged source
# KMS key for the S3 objects this submission writes (model output + the code /
# config / weight objects uploaded at submit time). '' -> plain uploads and the
# SDK/account default for the job output, so the submitting identity never needs
# kms:GenerateDataKey it may not have (e.g. an MFA-enforced account).
DEFAULT_SAGEMAKER_OUTPUT_KMS_KEY: str = ''
DEFAULT_SAGEMAKER_CONFIG_CHANNEL: str = ''      # S3 uri of the config uploaded as an input channel
DEFAULT_SAGEMAKER_WAIT: bool = False            # block until the job finishes
# Stream the job's CloudWatch logs into the submitting terminal while waiting.
# False waits quietly; --detach turns both this and `wait` off (submit and exit).
DEFAULT_SAGEMAKER_STREAM_LOGS: bool = True
# How input channels are delivered to the container. 'File' copies every object
# onto the EBS volume before training starts (simple, but the TUH corpora are
# ~400k small files and would need both the wait and the disk); 'FastFile'
# streams them through a FUSE mount, which the pickle loaders' os.listdir/open
# access pattern supports and which starts training immediately.
DEFAULT_SAGEMAKER_INPUT_MODE: str = 'File'
SAGEMAKER_INPUT_MODES = ('File', 'FastFile', 'Pipe')
# Local weight files whose bytes already live in S3. When submitting to
# SageMaker, a weight field (finetune_checkpoint.finetune /
# model.codebook_reg.tokenizer_weight) pointing at one of these local paths is
# served from its S3 mirror as an input channel instead of being re-uploaded on
# every submission -- the shipped ./checkpoints/*.pth are ~95 MB each and version
# controlled, so uploading them on each run is pure waste. Keyed by the path as
# written in the configs (matched by normalized path, so './checkpoints/x.pth'
# and 'checkpoints/x.pth' are the same); clear it (weight_s3_uris={}) to force the
# local file to upload, or add entries for your own weights. Consulted only by the
# submit CLI -- it never affects local (non-SageMaker) training.
DEFAULT_SAGEMAKER_WEIGHT_S3_URIS: Dict[str, str] = {
    './checkpoints/labram-base.pth': 's3://eeg-data-public/models/labram/labram-base.pth',
    './checkpoints/vqnsp.pth': 's3://eeg-data-public/models/labram/vqnsp.pth',
}

# ---------- Transformer architecture (NeuralTransformerBase) ----------
# Defaults match the "base" variant used in almost all production factories.
DEFAULT_ARCH_EEG_WINDOW_SIZE: int = 1600
DEFAULT_ARCH_PATCH_SIZE: int = 200
DEFAULT_ARCH_IN_CHANS: int = 1
DEFAULT_ARCH_OUT_CHANS: int = 8
DEFAULT_ARCH_NUM_CLASSES: int = 0
DEFAULT_ARCH_EMBED_DIM: int = 200
DEFAULT_ARCH_DEPTH: int = 12
DEFAULT_ARCH_NUM_HEADS: int = 10
DEFAULT_ARCH_MLP_RATIO: float = 4.0
DEFAULT_ARCH_QKV_BIAS: bool = False
DEFAULT_ARCH_DROP_RATE: float = 0.0
DEFAULT_ARCH_ATTN_DROP_RATE: float = 0.0
DEFAULT_ARCH_DROP_PATH_RATE: float = 0.0
DEFAULT_ARCH_USE_ABS_POS_EMB: bool = True
DEFAULT_ARCH_USE_REL_POS_BIAS: bool = False
DEFAULT_ARCH_USE_SHARED_REL_POS_BIAS: bool = False
DEFAULT_ARCH_USE_MEAN_POOLING: bool = True
DEFAULT_ARCH_INIT_STD: float = 0.02
DEFAULT_ARCH_INIT_SCALE: float = 0.001
DEFAULT_ARCH_USE_NORM: bool = True
# NeuralTransformerForMEM (pre-train head)
DEFAULT_ARCH_VOCAB_SIZE: int = DEFAULT_CODEBOOK_SIZE  # 8192
# VQNSP tokenizer
DEFAULT_ARCH_DECODER_OUT_DIM: int = 200
DEFAULT_ARCH_SMOOTH_L1_LOSS: bool = False
DEFAULT_QUANTIZER_BETA: float = 1.0
