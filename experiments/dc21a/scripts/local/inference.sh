#!/bin/bash

# go to appropriate directory
cd $HOME/code_projects/inr4ssh/
export PYTHONPATH=$HOME/code_projects/inr4ssh/:${PYTHONPATH}

# loading of modules
conda activate torch_py39

# run script (smoke-test: subset)
python experiments/dc21a/main.py \
    --stage="inference" \
    --results_name="/Users/eman/code_projects/logs/saved_data/test_dc21b_feb_pretrain.nc" \
    --my_config=experiments/dc21a/configs/config.py \
    --my_config.datadir.staging.staging_dir="/Volumes/EMANS_HDD/data/dc21b_ose/test_2/ml_ready" \
    --my_config.model.pretrain=True \
    --my_config.model.pretrain_reference="experiment-ckpts:v35" \
    --my_config.model.pretrain_checkpoint="last.ckpt" \
    --my_config.model.pretrain_id="3snro276" \
    --my_config.model.pretrain_entity="ige" \
    --my_config.model.pretrain_project="inr4ssh" \
    --my_config.log.mode="disabled" \
    --my_config.evaluation.time_min="2017-02-01" \
    --my_config.evaluation.time_max="2017-03-01" \
    --my_config.trainer.accelerator="mps"


# experiment-ckpts:v35 | 3snro276
# experiment-ckpts:v36 | 3ts6zjs8
