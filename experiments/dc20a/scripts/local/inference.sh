#!/bin/bash

# go to appropriate directory
cd $HOME/code_projects/inr4ssh/
export PYTHONPATH=$HOME/code_projects/inr4ssh/:${PYTHONPATH}

# loading of modules
conda activate torch_py39

##############################################
## EXPERIMENT: NADIR4
##############################################
# experiment-ckpts:v28 | 3dw5swoo
# experiment-ckpts:v30 | 3o1s9zze
# experiment-ckpts:v40 | jmkerjxt | epoch=9999-step=80000.ckpt
# experiment-ckpts:v44 | ige/inr4ssh/vnogg4dj | epoch=19901-step=159216.ckpt
# experiment-ckpts:v47 | ige/inr4ssh/qwhp4vot | last.ckpt
REFERENCE="experiment-ckpts:v47"
ID="qwhp4vot"
CHECKPOINT="last.ckpt"

python experiments/dc20a/main.py \
    --stage="inference" \
    --results_name="/Users/eman/code_projects/logs/saved_data/test_res_nadir4_jz_v40.nc" \
    --my_config=experiments/dc20a/configs/config_local.py \
    --my_config.datadir.staging.staging_dir="/Volumes/EMANS_HDD/data/dc20a_osse/test_2/ml_ready/" \
    --my_config.model.pretrain=True \
    --my_config.model.pretrain_reference=$REFERENCE \
    --my_config.model.pretrain_checkpoint=$CHECKPOINT \
    --my_config.model.pretrain_id=$ID \
    --my_config.model.pretrain_entity="ige" \
    --my_config.model.pretrain_project="inr4ssh" \
    --my_config.log.mode="disabled" \
    --my_config.evaluation.lon_coarsen=2 \
    --my_config.evaluation.lat_coarsen=2 \
    --my_config.preprocess.subset_time.time_max="2012-12-02" \
    --my_config.evaluation.time_max="2012-12-02"




###############################################
### EXPERIMENT: SWOT1+NADIR5
###############################################
#
## experiment-ckpts:v42 | epoch=9717-step=3216658.ckpt | ige/inr4ssh/69rrhkdg
#
#python experiments/dc20a/main.py \
#    --stage="inference" \
#    --results_name="/Users/eman/code_projects/logs/saved_data/test_res_swot1nadir5_jz.nc" \
#    --my_config=experiments/dc20a/configs/config_local.py \
#    --my_config.datadir.staging.staging_dir="/Volumes/EMANS_HDD/data/dc20a_osse/test_2/ml_ready/" \
#    --my_config.model.pretrain=True \
#    --my_config.model.pretrain_reference="experiment-ckpts:v42" \
#    --my_config.model.pretrain_checkpoint="epoch=9717-step=3216658.ckpt" \
#    --my_config.model.pretrain_id="69rrhkdg" \
#    --my_config.model.pretrain_entity="ige" \
#    --my_config.model.pretrain_project="inr4ssh" \
#    --my_config.log.mode="disabled" \
#    --my_config.evaluation.lon_coarsen=2 \
#    --my_config.evaluation.lat_coarsen=2 \
#    --my_config.preprocess.subset_time.time_max="2012-12-02" \
#    --my_config.evaluation.time_max="2012-12-02"
