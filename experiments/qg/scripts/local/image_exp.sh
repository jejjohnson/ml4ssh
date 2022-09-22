#!/bin/bash

# go to appropriate directory
cd $HOME/code_projects/inr4ssh/
export PYTHONPATH=$HOME/code_projects/inr4ssh/:${PYTHONPATH}

# loading of modules
conda activate torch_py39

# run script (smoke-test)
python experiments/qg/main.py \
    --experiment="image" \
    --my_config=experiments/qg/configs/config_image.py \
    --my_config.log.mode="disabled" \
    --my_config.optim.num_epochs=2 \
    --my_config.optim.warmup=1 \
    --my_config.optim_qg.num_epochs=2 \
    --my_config.optim_qg.warmup=1 \
    --my_config.trainer.grad_batches=2 \
    --my_config.data.data_dir="/Users/eman/code_projects/torchqg/data/qgsim_simple_128x128.nc" \
    --my_config.log.log_dir="/Users/eman/code_projects/logs/" \
    --my_config.data.res="128x128" \
    --my_config.dl.batchsize_train=1024 \
    --my_config.dl.batchsize_val=2048 \
    --my_config.dl.num_workers=1 \
    --my_config.dl.pin_memory=False \
    --my_config.trainer.accelerator="mps" \
    --my_config.trainer_qg.accelerator="cpu" \
    --my_config.pre.time_max=1
