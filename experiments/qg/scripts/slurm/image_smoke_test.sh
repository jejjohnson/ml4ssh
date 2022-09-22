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

# run script
python experiments/qg/main.py \
    --experiment="image" \
    --my_config=experiments/qg/configs/config_image.py \
    --my_config.log.mode="disabled" \
    --my_config.optim.num_epochs=20 \
    --my_config.optim.warmup=5 \
    --my_config.optim_qg.num_epochs=20 \
    --my_config.optim_qg.warmup=5 \
    --my_config.trainer.grad_batches=10
