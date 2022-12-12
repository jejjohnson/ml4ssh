#!/bin/bash

# go to appropriate directory
cd $HOME/code_projects/inr4ssh/
export PYTHONPATH=$HOME/code_projects/inr4ssh/:${PYTHONPATH}

# loading of modules
conda activate torch_py39

python experiments/dc20a/main.py \
    --stage="train" \
    --my_config=experiments/dc20a/configs/config.py \
    --my_config.experiment="nadir4" \
    --my_config.trainer.num_epochs=10000 \
    --my_config.lr_scheduler.warmup_epochs=50 \
    --my_config.lr_scheduler.max_epochs=100 \
    --my_config.lr_scheduler.eta_min=1e-5 \
    --my_config.preprocess.subset_time.time_max="2012-11-01" \
    --my_config.evaluation.time_max="2012-11-01" \
    --my_config.log.mode="offline" \
    --my_config.model.hidden_dim=256 \
    --my_config.evaluation.lon_coarsen=5 \
    --my_config.evaluation.lat_coarsen=5
