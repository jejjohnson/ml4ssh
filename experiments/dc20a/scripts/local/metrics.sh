#!/bin/bash

# go to appropriate directory
cd $HOME/code_projects/inr4ssh/
export PYTHONPATH=$HOME/code_projects/inr4ssh/:${PYTHONPATH}

# loading of modules
conda activate torch_py39

## run script
#python experiments/dc20a/main.py \
#    --stage="metrics" \
#    --results_name="/Users/eman/code_projects/logs/saved_data/test_res.nc" \
#    --variable_name="gssh" \
#    --my_config=experiments/dc20a/configs/config_local.py \
#    --my_config.datadir.staging.staging_dir="/Volumes/EMANS_HDD/data/dc20a_osse/test_2/ml_ready/" \
#    --my_config.log.mode="disabled"


# run script
python experiments/dc20a/main.py \
    --stage="metrics" \
    --results_name="/Users/eman/code_projects/logs/saved_data/test_res.nc" \
    --variable_name="ssh_model_predict" \
    --my_config=experiments/dc20a/configs/config_local.py \
    --my_config.datadir.staging.staging_dir="/Volumes/EMANS_HDD/data/dc20a_osse/test_2/ml_ready/" \
    --my_config.log.mode="disabled"


# /Volumes/EMANS_HDD/data/dc20a_osse/results/nadir4/2020a_SSH_mapping_NATL60_DUACS_en_j1_tpn_g2.nc | gssh
# /Volumes/EMANS_HDD/data/dc20a_osse/results/nadir4/2020a_SSH_mapping_NATL60_MIOST_en_j1_tpn_g2.nc
# /Volumes/EMANS_HDD/data/dc20a_osse/results/swot1nadir5/2020a_SSH_mapping_NATL60_DUACS_swot_en_j1_tpn_g2.nc
# /Volumes/EMANS_HDD/data/dc20a_osse/results/swot1nadir5/2020a_SSH_mapping_NATL60_MIOST_swot_en_j1_tpn_g2.nc
