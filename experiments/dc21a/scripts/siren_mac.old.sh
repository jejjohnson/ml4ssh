#!/bin/bash

# go to appropriate directory
cd ~/code_projects/inr4ssh
export PYTHONPATH=$WORK/code_projects/inr4ssh:${PYTHONPATH}

# loading of modules
source activate torch_py39

# run script
python experiments/expv2/train.py \
  --num_epochs 10 \
  --wandb_mode online \
  --wandb_log_dir "/Users/eman/code_projects/logs" \
  --device mps \
  --train_data_dir "/Users/eman/.CMVolumes/cal1_workdir/data/dc_2021/raw/train" \
  --ref_data_dir "/Users/eman/.CMVolumes/cal1_workdir/data/dc_2021/raw/ref" \
  --test_data_dir "/Users/eman/.CMVolumes/cal1_workdir/data/dc_2021/raw/test" \
  --preprocess.time_min "2017-01-01" \
  --preprocess.time_max "2017-02-01" \
  --eval.time_min "2017-01-01" \
  --eval.time_max "2017-02-01" \
  --eval.dtime_freq 12 \
  --eval.dtime_unit "h"
