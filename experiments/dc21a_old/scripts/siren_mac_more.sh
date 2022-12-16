#!/bin/bash

# go to appropriate directory
cd ~/code_projects/inr4ssh
export PYTHONPATH=$WORK/code_projects/inr4ssh:${PYTHONPATH}

# loading of modules
source activate torch_py39
#
#  --train_data_dir "/Users/eman/.CMVolumes/cal1_workdir/data/dc_2021/raw/train" \
#  --ref_data_dir "/Users/eman/.CMVolumes/cal1_workdir/data/dc_2021/raw/ref" \
#  --test_data_dir "/Users/eman/.CMVolumes/cal1_workdir/data/dc_2021/raw/test" \

# run script
python experiments/dc21a/train_more.py \
  --run_path "ige/inr4ssh/2z8tsrfn" \
  --model_path "checkpoints/epoch=836-step=329778.ckpt" \
  --num_epochs 2 \
  --mode offline \
  --log_dir "/Users/eman/code_projects/logs" \
  --device mps \
  --gpus 0 \
  --train_data_dir "/Volumes/EMANS_HDD/data/dc21b/train" \
  --ref_data_dir "/Volumes/EMANS_HDD/data/dc21b/ref" \
  --test_data_dir "/Volumes/EMANS_HDD/data/dc21b/test" \
  --learning_rate 1e-4 \
  --factor 0.025 \
  --lr_scheduler.patience 20 \
  --callbacks.patience 100
