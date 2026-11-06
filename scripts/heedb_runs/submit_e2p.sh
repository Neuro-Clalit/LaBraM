#!/bin/bash
# E2P: E2L scenario (TUH + HEEDB, identical split, 16 epochs / 4 warm-up, batch 256,
# lr 2.8e-4, patience 5) as a status + 17-group pathology classifier without age
# prediction (main's Exp4cls recipe: multi-task head, class-balanced pathology
# loss, loss.age_weight=0, epoch selected on pathology macro ROC-AUC), on one
# ml.g5.12xlarge with 4-GPU DDP (global batch 256 = 4 x 64).
cd /data/code/LaBraM/.claude/worktrees/bridge-cse_01CjxPDyZr6ZFF9cF4BmbpuA
/data/code/LaBraM/.venv/bin/python -m labram.runs.submit_sagemaker --phase finetune \
  --config labram/configs/defaults/finetune_tuab_age.json "$@" --set \
  data.data_path=s3://eeg-data-public/HEEDB/v4.1/cohorts/tuh_heedb_v1_compact data.data_format=npy \
  'data.drop_channels=["A1","A2","T1","T2"]' channel_drop.enabled=true channel_drop.prob=0.0 channel_drop.consistency_weight=0.0 \
  data.num_workers=10 data.train_minutes_per_epoch=5 mixup.enabled=true model.fused_attention=true model.use_mean_pooling=false \
  model.head_type=multitask data.multitask_targets=multitask_targets_v2.npz \
  'model.multitask_head.shared_dims=[256,256]' 'model.multitask_head.age_dims=[128,64]' 'model.multitask_head.abnormal_dims=[128,64]' \
  loss.abnormal_weight=0.5 loss.pathology_weight=0.5 loss.pathology_reduction=present loss.age_weight=0.0 \
  evaluation.select_metric=pathology_macro_roc_auc evaluation.train_eval_every=4 \
  optimizer.lr=0.00028 optimizer.warmup_epochs=4 trainer.epochs=16 trainer.batch_size=64 trainer.global_batch_size=256 \
  trainer.early_stopping=true trainer.early_stopping_patience=5 \
  distributed.nproc_per_node=-1 output.output_dir= output.log_dir= \
  clearml.enabled=true clearml.project_name=LaBraM/heedb clearml.task_name=heedb_E2P_pathology_cls_ddp4 \
  'clearml.tags=["heedb","E2P","pathology","classification-only","ddp4"]' \
  sagemaker.enabled=true sagemaker.role=arn:aws:iam::574441342949:role/SageMakerExecutionRole sagemaker.input_mode=File \
  sagemaker.instance_type=ml.g5.12xlarge sagemaker.volume_size_gb=1000 sagemaker.job_name_prefix=labram-heedb-e2p-pathology
