#!/bin/bash

module purge

#module load cuda/11.2
#module load cudnn/9.2-v7.5.1.10
module load cudnn/10.1-v7.5.1.10
module load git/2.31.1
module load github-cli/1.13.1
module load git-lfs/3.0.2
module load ffmpeg/4.2.2

# go to appropriate directory
cd /gpfswork/rech/cli/uvo53rl/projects/inr4ssh/
export PYTHONPATH=/gpfswork/rech/cli/uvo53rl/projects/inr4ssh:${PYTHONPATH}

# loading of modules
source activate torch_py39

python experiments/dc20a/main.py \
    --stage="train" \
    --my_config=experiments/dc20a/configs/config.py \
    --my_config.experiment="swot1nadir5" \
    --my_config.trainer.num_epochs=10000 \
    --my_config.lr_scheduler.warmup_epochs=50 \
    --my_config.lr_scheduler.max_epochs=100 \
    --my_config.lr_scheduler.eta_min=1e-5 \
    --my_config.log.mode="disabled" \
    --my_config.model.hidden_dim=256 \
    --my_config.trainer.accelerator="gpu" \
    --my_config.trainer.strategy="dp" \
    --my_config.trainer.devices=4 \
    --my_config.trainer.grad_batches=1
#    --my_config.evaluation.lon_coarsen=5 \
#    --my_config.evaluation.lat_coarsen=5 \
