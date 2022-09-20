#!/bin/bash

#SBATCH --job-name=dc21b                     # name of job
#SBATCH --account=cli@v100                   # for statistics
#SBATCH --nodes=1                            # we ALWAYS request one node
#SBATCH --ntasks-per-node=1                  # number of tasks per node
#SBATCH --cpus-per-task=10                   # number of cpus per task
#SBATCH -C v100-16g                          # V100 GPU + 16 GBs RAM
#SBATCH --qos=qos_gpu-t3                     # GPU partition (max 20� hrs)
#SBATCH --gres=gpu:1                         # number of GPUs (1/4 of GPUs)
#SBATCH --time=20:00:00                      # maximum execution time requested (HH:MM:SS)
#SBATCH --output=/gpfsscratch/rech/cli/uvo53rl/logs/inr4ssh_dc_2021b_%j.log      # name of output file
#SBATCH --error=/gpfsscratch/rech/cli/uvo53rl/errs/inr4ssh_dc_2021b_%j.err       # name of error file
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

# run script
srun python experiments/qg/qg_sim/main.py \
    --my_config=experiments/qg/qg_sim/config_jz.py \
    --my_config.optim.num_epochs=1000 \
    --my_config.optim.warmup=100 \
    --my_config.loss.qg=False
#    --my_config.loss.alpha=1e4 \
#    --my_config.log.mode="offline" \
#    --my_config.log.log_dir="/gpfsscratch/rech/cli/uvo53rl/" \
#    --my_config.trainer.accelerator="gpu" \
#    --my_config.dl.batchsize_train=4096 \
#    --my_config.dl.num_workers=10 \
#    --my_config.data.data_dir="/gpfswork/rech/cli/uvo53rl/projects/inr4ssh/data/qgsim_simple_128x128.nc"
