#!/bin/bash

# go to appropriate directory
cd ~/code_projects/inr4ssh
export PYTHONPATH=$WORK/code_projects/inr4ssh:${PYTHONPATH}

# loading of modules
source activate torch_py39

# run script
python experiments/expv2/train_pl.py \
  --num_epochs 2 \
  --wandb_mode disabled \
  --wandb_log_dir "/Users/eman/code_projects/logs" \
  --device mps \
  --gpus 0 \
  --train_data_dir "/Users/eman/.CMVolumes/cal1_workdir/data/dc_2021/raw/train" \
  --ref_data_dir "/Users/eman/.CMVolumes/cal1_workdir/data/dc_2021/raw/ref" \
  --test_data_dir "/Users/eman/.CMVolumes/cal1_workdir/data/dc_2021/raw/test" \
  --preprocess.time_min "2017-01-01" \
  --preprocess.time_max "2017-02-01" \
  --eval.time_min "2017-01-01" \
  --eval.time_max "2017-02-01" \
  --eval.dtime_freq 12 \
  --eval.dtime_unit "h" \
  --cartesian True \
  --minmax_spatial True \
  --minmax_temporal True \
  --abs_time_min 2016-11-01 \
  --abs_time_max 2018-02-01
