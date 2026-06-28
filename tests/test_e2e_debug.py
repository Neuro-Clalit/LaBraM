import os
import torch
import numpy as np
import h5py
import pickle
import pytest
import argparse
from pathlib import Path
from labram.runs import run_vqnsp, run_pretrain, run_finetune
from labram.configs.base_configs import ConfigBase
from labram.data.eeg_constants import TUH_EEG_CH_NAMES, normalize_ch_names

def create_synthetic_hdf5(path, n_subjects=2, n_samples=160000, n_channels=23):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as f:
        for i in range(n_subjects):
            subj = f.create_group(f'subject_{i}')
            eeg = np.random.randn(n_channels, n_samples).astype(np.float32)
            ds = subj.create_dataset('eeg', data=eeg)
            ds.attrs['chOrder'] = normalize_ch_names(TUH_EEG_CH_NAMES)

def create_synthetic_tuab(root, n_samples=32):
    for split in ['train', 'val', 'test']:
        split_dir = Path(root) / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_samples):
            data = {
                'X': np.random.randn(23, 200).astype(np.float32),
                'y': i % 2
            }
            with open(split_dir / f'sample_{i}.pkl', 'wb') as f:
                pickle.dump(data, f)

def run_pipeline(base_dir, device="cpu", real_tuab_path=None, verbose=False):
    base_dir = Path(base_dir)

    def print_command(phase, module, config):
        if not verbose:
            return
        
        cmd = [f"python -m {module}"]
        
        def _flatten(obj, prefix=""):
            for k, v in obj.__dict__.items():
                full_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, ConfigBase):
                    _flatten(v, full_key)
                else:
                    if v is not None:
                        # Simple representation of the value
                        if isinstance(v, list):
                            if len(v) > 0 and isinstance(v[0], list):
                                # Handle list of lists (datasets_train)
                                val_str = ",".join([",".join(map(str, inner)) for inner in v])
                            else:
                                val_str = ",".join(map(str, v))
                        else:
                            val_str = str(v)
                        cmd.append(f"    --set {full_key}={val_str}")
        
        _flatten(config)
        
        print(f"\n[VERBOSE] {phase} command equivalent:")
        print(" \\\n".join(cmd))

    # 1. Setup Data
    pretrain_h5 = base_dir / "data" / "pretrain.h5"
    if not pretrain_h5.exists():
        create_synthetic_hdf5(pretrain_h5)
    
    if real_tuab_path:
        tuab_root = Path(real_tuab_path)
    else:
        tuab_root = base_dir / "data" / "TUAB"
        if not tuab_root.exists():
            create_synthetic_tuab(tuab_root)
    
    vqnsp_out = base_dir / "vqnsp_out"
    pretrain_out = base_dir / "pretrain_out"
    finetune_out = base_dir / "finetune_out"
    
    for d in [vqnsp_out, pretrain_out, finetune_out]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 2. Run VQNSP
    print("\n>>> Running VQNSP stage")
    vqnsp_config = run_vqnsp.VQNSPRunConfig()
    vqnsp_config.update(**{
        "data.datasets_train": [[str(pretrain_h5)]],
        "data.datasets_val": [[str(pretrain_h5)]],
        "data.time_window": [1],
        "data.val_time_window": [1],
        "data.num_workers": 0,
        "trainer.debug": True,
        "trainer.epochs": 2,
        "optimizer.warmup_epochs": 1,
        "output.save_ckpt_freq": 1,
        "distributed.device": device,
        "output.output_dir": str(vqnsp_out),
        "model.model": "vqnsp_encoder_base_decoder_3x200x12",
    })
    print_command("VQNSP", "labram.runs.run_vqnsp", vqnsp_config)
    run_vqnsp.main(vqnsp_config)
    
    vqnsp_checkpoint = vqnsp_out / "checkpoint-1.pth"
    
    # 3. Run Pre-training
    print("\n>>> Running Pre-training stage")
    pretrain_config = run_pretrain.PretrainRunConfig()
    pretrain_config.update(**{
        "data.datasets_train": [[str(pretrain_h5)]],
        "data.time_window": [1],
        "data.num_workers": 0,
        "trainer.debug": True,
        "trainer.epochs": 2,
        "optimizer.warmup_epochs": 1,
        "output.save_ckpt_freq": 1,
        "distributed.device": device,
        "output.output_dir": str(pretrain_out),
        "model.model": "labram_base_patch200_1600_8k_vocab",
        "model.tokenizer.tokenizer_weight": str(vqnsp_checkpoint),
    })
    print_command("Pre-training", "labram.runs.run_pretrain", pretrain_config)
    run_pretrain.main(pretrain_config)
    
    pretrain_checkpoint = pretrain_out / "checkpoint-1.pth"
    
    # 4. Run Fine-tuning
    print("\n>>> Running Fine-tuning stage")
    finetune_config = run_finetune.FinetuneRunConfig()
    finetune_config.update(**{
        "data.dataset": "TUAB",
        "data.data_path": str(tuab_root),
        "trainer.debug": True,
        "trainer.epochs": 2,
        "optimizer.warmup_epochs": 1,
        "trainer.debug_samples": 4,
        "distributed.device": device,
        "output.output_dir": str(finetune_out),
        "model.model": "labram_base_patch200_200",
        "finetune_checkpoint.finetune": str(pretrain_checkpoint),
    })
    print_command("Fine-tuning", "labram.runs.run_finetune", finetune_config)
    run_finetune.main(finetune_config)
    
    print(f"\n>>> E2E Pipeline finished. Results in {base_dir}")
    return finetune_out

@pytest.mark.parametrize("device", ["cpu"])
def test_e2e_debug_run(tmp_path, device, pytestconfig):
    verbose = pytestconfig.getoption("verbose") > 0
    finetune_out = run_pipeline(tmp_path, device, verbose=verbose)
    assert (finetune_out / "log.txt").exists()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LaBraM E2E debug pipeline.")
    parser.add_argument("--data_path", type=str, help="Path to real TUAB data (pickle format).")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu, cuda, mps).")
    parser.add_argument("--output_dir", type=str, default="./debug_e2e_out", help="Directory for checkpoints and logs.")
    parser.add_argument("--verbose", action="store_true", help="Print training commands.")
    args = parser.parse_args()
    
    run_pipeline(args.output_dir, args.device, args.data_path, verbose=args.verbose)
