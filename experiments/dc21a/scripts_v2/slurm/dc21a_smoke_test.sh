#!/bin/bash

# loading of modules
module purge

#module load cuda/11.2
module load cudnn/9.2-v7.5.1.10
module load git/2.31.1
module load github-cli/1.13.1
module load git-lfs/3.0.2
module load ffmpeg/4.2.2

# go to appropriate directory
cd $WORK/projects/inr4ssh
export PYTHONPATH=$WORK/projects/inr4ssh:${PYTHONPATH}

# loading of modules
source activate torch_py39

# run script (smoke test)
python experiments/dc21a/main.py \
    --experiment="dc21a" \
    --my_config=experiments/dc21a/configs/config_v2.py \
    --my_config.log.mode="offline" \
    --my_config.trainer.accelerator="gpu" \
    --my_config.optimizer.num_epochs=50 \
    --my_config.lr_scheduler.warmup_epochs=10 \
    --my_config.dataloader.num_workers=10 \
    --my_config.dataloader.pin_memory=True \
    --my_config.trainer.dev_run=False \
    --my_config.preprocess.time_min="2017-01-01" \
    --my_config.preprocess.time_max="2017-02-01" \
    --my_config.eval_data.time_min="2017-01-01" \
    --my_config.eval_data.time_max="2017-02-01" \
    --my_config.trainer.devices=1 \
    --my_config.trainer.num_nodes=1
