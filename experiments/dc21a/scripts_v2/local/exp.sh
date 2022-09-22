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

# run script (smoke-test: no QG)
python experiments/dc21a/main.py \
    --experiment="simulation" \
    --my_config=experiments/dc21a/configs/config_v2.py \
    --my_config.log.log_dir="/Users/eman/code_projects/logs/" \
    --my_config.log.mode="online" \
    --my_config.data.train_data_dir="/Users/eman/.CMVolumes/cal1_workdir/data/dc_2021/raw/train" \
    --my_config.data.ref_data_dir="/Users/eman/.CMVolumes/cal1_workdir/data/dc_2021/raw/ref" \
    --my_config.data.test_data_dir="/Users/eman/.CMVolumes/cal1_workdir/data/dc_2021/raw/test" \
    --my_config.preprocess.time_min="2017-01-01" \
    --my_config.preprocess.time_max="2017-02-01" \
    --my_config.eval_data.time_min="2017-01-01" \
    --my_config.eval_data.time_max="2017-02-01" \
    --my_config.trainer.accelerator="mps" \
    --my_config.optimizer.num_epochs=2 \
    --my_config.lr_scheduler.warmup_epochs=1 \
    --my_config.lr_scheduler.max_epochs=2 \
    --my_config.dataloader.num_workers=0 \
    --my_config.dataloader.pin_memory=False \
    --my_config.trainer.dev_run=False
