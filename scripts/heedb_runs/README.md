# HEEDB run launchers (as used, 2026-11-03 – 2026-11-06)

The exact commands behind the HEEDB experiments in `docs/heedb_onboarding_plan.md`. They use absolute
paths of the GPU box (`/data/datasets/...`, `/data/runs/heedb/...`) and the worktree they ran from; adapt
the paths before reuse.

| Script | Run |
|---|---|
| `run_e2c.sh` | E2c, local: E2 + clinical condition inputs |
| `run_e2l.sh` | E2L, local: 16 epochs, batch 512 / lr 4e-4 (epochs 1–6) |
| `resume_e2l.sh`, `e2l_switch_to_pread.sh` | E2L resume with the pread loader |
| `resume_e2l_bs256.sh` | E2L epochs 7–16: batch 256 / lr 2.8e-4, 12 workers |
| `submit_e2p.sh` | E2P on ml.g5.12xlarge, 4-GPU DDP (hung in NCCL, see docs/multi_gpu.md) |
| `submit_e2p_1gpu.sh` | E2P on ml.g5.16xlarge, 1 GPU (the completed run) |
| `score_val_test.sh` | val + test scoring of E0 / E1 / E2 for the bias-correction report |
| `score_e2p.sh` | E2P per-class scoring (`docs/data/e2p_per_class.csv`) |
| `make_epoch_csv.sh` | per-epoch metrics from the run logs |
