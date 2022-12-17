#!/bin/bash

# go to appropriate directory
cd $HOME/code_projects/inr4ssh/
export PYTHONPATH=$HOME/code_projects/inr4ssh/:${PYTHONPATH}

# loading of modules
conda activate torch_py39

## run script (smoke-test: QG)
#python experiments/qg/main.py \
#    --experiment="simulation" \
#    --my_config=experiments/qg/config.py \
#    --my_config.log.mode="disabled" \
#    --my_config.optim.num_epochs=5 \
#    --my_config.optim.warmup=1 \
#    --my_config.trainer.grad_batches=2 \
#    --my_config.data.data_dir="/Users/eman/code_projects/torchqg/data/qgsim_simple_128x128.nc" \
#    --my_config.log.log_dir="/Users/eman/code_projects/logs/" \
#    --my_config.data.res="128x128" \
#    --my_config.dl.batchsize_train=1024 \
#    --my_config.dl.batchsize_val=2048 \
#    --my_config.dl.num_workers=1 \
#    --my_config.dl.pin_memory=False \
#    --my_config.trainer.accelerator="cpu" \
#    --my_config.pre.time_max=1

## ====================================================================
## EXPERIMENT - SUBSET TEMPORAL (NADIR)
## ====================================================================
#python experiments/dc21a/main.py \
#    --stage="train" \
#    --my_config=experiments/dc21a/configs/config.py \
#    --my_config.datadir.staging.staging_dir="/Volumes/EMANS_HDD/data/dc21b_ose/test_2/ml_ready" \
#    --my_config.log.log_dir="/Users/eman/code_projects/logs/" \
#    --my_config.log.mode="disabled" \
#    --my_config.dataloader.batchsize_train=32 \
#    --my_config.trainer.grad_batches=128 \
#    --my_config.trainer.num_epochs=2 \
#    --my_config.lr_scheduler.warmup_epochs=1 \
#    --my_config.lr_scheduler.warmup_lr=1e-6 \
#    --my_config.lr_scheduler.eta_min=1e-5 \
#    --my_config.preprocess.subset_time.time_min="2017-01-01" \
#    --my_config.preprocess.subset_time.time_max="2017-02-01" \
#    --my_config.evaluation.time_min="2017-01-01" \
#    --my_config.evaluation.time_max="2017-02-01" \
#    --my_config.trainer.accelerator="mps" \
#    --my_config.model.hidden_dim=256 \
#    --my_config.traintest.subset_random=0.25

## ====================================================================
## SMOKE TEST - SUBSET TEMPORAL
## ====================================================================
#python experiments/dc21a/main.py \
#    --stage="train" \
#    --my_config=experiments/dc21a/configs/config.py \
#    --my_config.datadir.staging.staging_dir="/Volumes/EMANS_HDD/data/dc21b_ose/test_2/ml_ready" \
#    --my_config.log.log_dir="/Users/eman/code_projects/logs/" \
#    --my_config.log.mode="online" \
#    --my_config.dataloader.batchsize_train=4092 \
#    --my_config.trainer.grad_batches=2 \
#    --my_config.trainer.num_epochs=1000 \
#    --my_config.lr_scheduler.warmup_epochs=50 \
#    --my_config.optimizer.learning_rate=1e-3 \
#    --my_config.lr_scheduler.warmup_lr=1e-5 \
#    --my_config.lr_scheduler.eta_min=1e-5 \
#    --my_config.preprocess.subset_time.time_min="2017-01-15" \
#    --my_config.preprocess.subset_time.time_max="2017-03-15" \
#    --my_config.evaluation.time_min="2017-02-01" \
#    --my_config.evaluation.time_max="2017-03-01" \
#    --my_config.trainer.accelerator="mps" \
#    --my_config.model.hidden_dim=256 \
#    --my_config.traintest.subset_random=0.0 \
#    --my_config.model.pretrain=False
#

# ====================================================================
# SMOKE TEST - SUBSET TEMPORAL (PRETRAINED)
# ====================================================================
python experiments/dc21a/main.py \
    --stage="train" \
    --my_config=experiments/dc21a/configs/config.py \
    --my_config.datadir.staging.staging_dir="/Volumes/EMANS_HDD/data/dc21b_ose/test_2/ml_ready" \
    --my_config.log.log_dir="/Users/eman/code_projects/logs/" \
    --my_config.log.mode="online" \
    --my_config.dataloader.batchsize_train=4092 \
    --my_config.trainer.grad_batches=2 \
    --my_config.trainer.num_epochs=1000 \
    --my_config.lr_scheduler.warmup_epochs=50 \
    --my_config.optimizer.learning_rate=1e-4 \
    --my_config.lr_scheduler.warmup_lr=1e-6 \
    --my_config.lr_scheduler.eta_min=1e-6 \
    --my_config.preprocess.subset_time.time_min="2017-01-15" \
    --my_config.preprocess.subset_time.time_max="2017-03-15" \
    --my_config.evaluation.time_min="2017-02-01" \
    --my_config.evaluation.time_max="2017-03-01" \
    --my_config.trainer.accelerator="mps" \
    --my_config.model.hidden_dim=256 \
    --my_config.traintest.subset_random=0.0 \
    --my_config.model.pretrain=True \
    --my_config.model.pretrain_reference="experiment-ckpts:v35" \
    --my_config.model.pretrain_checkpoint="last.ckpt" \
    --my_config.model.pretrain_id="3snro276" \
    --my_config.callbacks.wandb_artifact=True

# experiment-ckpts:v35 | 3snro276
# experiment-ckpts:v30 | 3o1s9zze
