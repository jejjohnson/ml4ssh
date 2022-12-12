#!/bin/bash

# go to appropriate directory
cd $HOME/code_projects/inr4ssh/
export PYTHONPATH=$HOME/code_projects/inr4ssh/:${PYTHONPATH}

# loading of modules
conda activate torch_py39

# run script (smoke-test: subset)
python experiments/dc20a/main.py \
    --stage="inference" \
    --results_name="/Users/eman/code_projects/logs/saved_data/test_res_sub.nc" \
    --my_config=experiments/dc20a/configs/config_local.py \
    --my_config.datadir.staging.staging_dir="/Volumes/EMANS_HDD/data/dc20a_osse/test_2/ml_ready/" \
    --my_config.model.pretrain=True \
    --my_config.model.pretrain_reference="experiment-ckpts:v21" \
    --my_config.model.pretrain_checkpoint="last.ckpt" \
    --my_config.model.pretrain_id="o6515b3e" \
    --my_config.model.pretrain_entity="ige" \
    --my_config.model.pretrain_project="inr4ssh" \
    --my_config.log.mode="disabled" \
    --my_config.evaluation.lon_coarsen=5 \
    --my_config.evaluation.lat_coarsen=5 \
    --my_config.preprocess.subset_time.time_max="2012-11-01" \
    --my_config.evaluation.time_max="2012-11-01"
