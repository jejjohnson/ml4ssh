#!/bin/bash

# go to appropriate directory
cd ~/code_projects/inr4ssh
export PYTHONPATH=$WORK/code_projects/inr4ssh:${PYTHONPATH}

# loading of modules
source activate torch_py39

# "/Users/eman/.CMVolumes/cal1_workdir/data/dc_2021/raw/train" \
# "/Users/eman/.CMVolumes/cal1_workdir/data/dc_2021/raw/ref"
# "/Users/eman/.CMVolumes/cal1_workdir/data/dc_2021/raw/test"

# run script
python experiments/dc21a/train.py \
  --num_epochs 2 \
  --mode offline \
  --log_dir "/Users/eman/code_projects/logs" \
  --device "mps" \
  --gpus 0 \
  --train_data_dir "/Volumes/EMANS_HDD/data/dc21b/train" \
  --ref_data_dir "/Volumes/EMANS_HDD/data/dc21b/ref" \
  --test_data_dir "/Volumes/EMANS_HDD/data/dc21b/test" \
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
  --abs_time_max 2018-02-01 \
  --model "fouriernet" \
  --latent_dim 512 \
  --lr_scheduler "cosine"
