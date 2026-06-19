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
