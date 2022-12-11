#!/bin/bash

#SBATCH --job-name=dc20a                     # name of job
#SBATCH --account=cli@cpu                   # for statistics
#SBATCH --nodes=1                            # we ALWAYS request one node
#SBATCH --ntasks-per-node=1                  # number of tasks per node
#SBATCH --cpus-per-task=32                   # number of cpus per task
#SBATCH --qos=qos_cpu-t4                     # GPU partition (max 100 hrs)
#SBATCH --time=90:00:00                      # maximum execution time requested (HH:MM:SS)
#SBATCH --output=/gpfsscratch/rech/cli/uvo53rl/logs/nerf4ssh_dc20a_%j.log      # name of output file
#SBATCH --error=/gpfsscratch/rech/cli/uvo53rl/errs/nerf4ssh_dc20a_%j.err       # name of error file
#SBATCH --export=ALL

# loading of modules
module purge

#module load cuda/11.2
module load cudnn/9.2-v7.5.1.10
module load git/2.31.1
module load github-cli/1.13.1
module load git-lfs/3.0.2
module load ffmpeg/4.2.2

# go to appropriate directory
cd $WORK/projects/inr4ssh
export PYTHONPATH=$WORK/projects/inr4ssh:${PYTHONPATH}

# loading of modules
source activate torch_py39

## run script
#python experiments/dc20a/main.py \
#    --stage="train" \
#    --my_config=experiments/dc20a/configs/config.py \
#    --my_config.experiment="swot1nadir5" \
#    --my_config.trainer.num_epochs=100 \
#    --my_config.lr_scheduler.warmup_epochs=10 \
#    --my_config.lr_scheduler.eta_min=1e-5 \
#    --my_config.preprocess.subset_time.time_max="2012-12-02" \
#    --my_config.evaluation.time_max="2012-11-01" \
#    --my_config.model.hidden_dim=256 \
#    --my_config.preprocess.subset_spatial.lon_min=-62.0 \
#    --my_config.preprocess.subset_spatial.lon_max=-58.0 \
#    --my_config.preprocess.subset_spatial.lat_min=35.0 \
#    --my_config.preprocess.subset_spatial.lat_max=40.0 \
#    --my_config.evaluation.lon_min=-62.0 \
#    --my_config.evaluation.lon_max=-58.0 \
#    --my_config.evaluation.lat_min=35.0 \
#    --my_config.evaluation.lat_max=40.0


# run script
python experiments/dc20a/main.py \
    --stage="train" \
    --my_config=experiments/dc20a/configs/config.py \
    --my_config.experiment="nadir1" \
    --my_config.trainer.num_epochs=1000 \
    --my_config.lr_scheduler.warmup_epochs=50 \
    --my_config.lr_scheduler.eta_min=1e-5
