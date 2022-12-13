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


## run script
#python experiments/dc20a/main.py \
#    --stage="metrics" \
#    --results_name="/Volumes/EMANS_HDD/data/dc20a_osse/results/nadir4/2020a_SSH_mapping_NATL60_MIOST_en_j1_tpn_g2.nc" \
#    --variable_name="gssh" \
#    --my_config=experiments/dc20a/configs/config_local.py \
#    --my_config.datadir.staging.staging_dir="/Volumes/EMANS_HDD/data/dc20a_osse/test_2/ml_ready/" \
#    --my_config.log.mode="disabled" \
#    --my_config.preprocess.subset_time.time_max="2012-11-01" \
#    --my_config.evaluation.time_max="2012-11-01"

# run script
python experiments/dc20a/main.py \
    --stage="viz" \
    --figure="psd_iso" \
    --results_name="/Volumes/EMANS_HDD/data/dc20a_osse/results/swot1nadir5/2020a_SSH_mapping_NATL60_MIOST_swot_en_j1_tpn_g2.nc" \
    --variable_name="gssh" \
    --my_config=experiments/dc20a/configs/config_local.py \
    --my_config.datadir.staging.staging_dir="/Volumes/EMANS_HDD/data/dc20a_osse/test_2/ml_ready/" \
    --my_config.log.mode="disabled" \
    --my_config.preprocess.subset_time.time_max="2012-12-02" \
    --my_config.evaluation.time_max="2012-12-02"

# /Users/eman/code_projects/logs/saved_data/test_res.nc | ssh_model_predict
# /Volumes/EMANS_HDD/data/dc20a_osse/results/nadir4/2020a_SSH_mapping_NATL60_DUACS_en_j1_tpn_g2.nc | gssh
# /Volumes/EMANS_HDD/data/dc20a_osse/results/nadir4/2020a_SSH_mapping_NATL60_MIOST_en_j1_tpn_g2.nc
# /Volumes/EMANS_HDD/data/dc20a_osse/results/swot1nadir5/2020a_SSH_mapping_NATL60_DUACS_swot_en_j1_tpn_g2.nc
# /Volumes/EMANS_HDD/data/dc20a_osse/results/swot1nadir5/2020a_SSH_mapping_NATL60_MIOST_swot_en_j1_tpn_g2.nc
