#!/bin/bash

#SBATCH --job-name=dc21a                     # name of job
#SBATCH --account=cli@v100                   # for statistics
#SBATCH --nodes=1                            # we ALWAYS request one node
#SBATCH --ntasks-per-node=1                  # number of tasks per node
#SBATCH --cpus-per-task=10                   # number of cpus per task
#SBATCH -C v100-16g                          # V100 GPU + 16 GBs RAM
#SBATCH --qos=qos_gpu-t3                     # GPU partition (max 20� hrs)
#SBATCH --gres=gpu:1                         # number of GPUs (1/4 of GPUs)
#SBATCH --time=20:00:00                      # maximum execution time requested (HH:MM:SS)
#SBATCH --output=/gpfsscratch/rech/cli/uvo53rl/logs/nerf4ssh_dc21a_overfit_%j.log      # name of output file
#SBATCH --error=/gpfsscratch/rech/cli/uvo53rl/errs/nerf4ssh_dc21a_overfit_%j.err       # name of error file
#SBATCH --export=ALL
#SBATCH --signal=SIGUSR1@90


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

srun experiments/dc21a/scripts_v2/jz/dc21a_overfit.sh

## run script (smoke test)
#srun python experiments/dc21a/main.py \
#    --experiment="dc21a" \
#    --my_config=experiments/dc21a/configs/config.py \
#    --my_config.log.mode="offline" \
#    --my_config.trainer.accelerator="gpu" \
#    --my_config.optimizer.num_epochs=1000 \
#    --my_config.lr_scheduler.warmup_epochs=50 \
#    --my_config.dataloader.num_workers=10 \
#    --my_config.dataloader.pin_memory=True \
#    --my_config.trainer.dev_run=False \
#    --my_config.preprocess.time_min="2017-01-01" \
#    --my_config.preprocess.time_max="2017-02-01" \
#    --my_config.eval_data.time_min="2017-01-01" \
#    --my_config.eval_data.time_max="2017-02-01"
