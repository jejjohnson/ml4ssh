#!/bin/bash

# go to appropriate directory
cd $HOME/code_projects/inr4ssh/
export PYTHONPATH=$HOME/code_projects/inr4ssh/:${PYTHONPATH}

# loading of modules
conda activate torch_py39

# run script (smoke-test: subset)
python experiments/dc20a/main.py \
    --stage="train" \
    --my_config=experiments/dc20a/configs/config_local.py \
    --my_config.datadir.staging.staging_dir="/Volumes/EMANS_HDD/data/dc20a_osse/test_2/ml_ready/" \
    --my_config.experiment="swot1nadir5" \
    --my_config.trainer.num_epochs=10 \
    --my_config.lr_scheduler.warmup_epochs=5 \
    --my_config.lr_scheduler.max_epochs=10 \
    --my_config.lr_scheduler.eta_min=1e-5 \
    --my_config.preprocess.subset_time.time_max="2012-12-02" \
    --my_config.evaluation.time_max="2012-11-01" \
    --my_config.log.mode="disabled" \
    --my_config.model.hidden_dim=256 \
    --my_config.preprocess.subset_spatial.lon_min=-62.0 \
    --my_config.preprocess.subset_spatial.lon_max=-58.0 \
    --my_config.preprocess.subset_spatial.lat_min=35.0 \
    --my_config.preprocess.subset_spatial.lat_max=40.0 \
    --my_config.evaluation.lon_min=-62.0 \
    --my_config.evaluation.lon_max=-58.0 \
    --my_config.evaluation.lat_min=35.0 \
    --my_config.evaluation.lat_max=40.0 \
    # --my_config.preprocess.subset_time.time_max="2012-11-01" \
    # --my_config.evaluation.time_max="2012-11-01" \
    # --my_config.data.train_data_dir="/Users/eman/.CMVolumes/cal1_workdir/data/dc_2021/raw/train" \
    # --my_config.data.ref_data_dir="/Users/eman/.CMVolumes/cal1_workdir/data/dc_2021/raw/ref" \
    # --my_config.data.test_data_dir="/Users/eman/.CMVolumes/cal1_workdir/data/dc_2021/raw/test" \
    # --my_config.preprocess.time_min="2017-01-01" \
    # --my_config.preprocess.time_max="2017-02-01" \
    # --my_config.eval_data.time_min="2017-01-01" \
    # --my_config.eval_data.time_max="2017-02-01" \
    # --my_config.trainer.accelerator="mps" \
    # --my_config.optimizer.num_epochs=2 \
    # --my_config.lr_scheduler.warmup_epochs=1 \
    # --my_config.lr_scheduler.max_epochs=2 \
    # --my_config.dataloader.num_workers=0 \
    # --my_config.dataloader.pin_memory=False \
    # --my_config.trainer.dev_run=False
