#!/bin/bash

#SBATCH --job-name=qgimg                     # name of job
#SBATCH --account=cli@v100                   # for statistics
#SBATCH --nodes=1                            # we ALWAYS request one node
#SBATCH --ntasks-per-node=1                  # number of tasks per node
#SBATCH --cpus-per-task=10                   # number of cpus per task
#SBATCH -C v100-16g                          # V100 GPU + 16 GBs RAM
#SBATCH --qos=qos_gpu-t4                     # GPU partition (max 20� hrs)
#SBATCH --gres=gpu:1                         # number of GPUs (1/4 of GPUs)
#SBATCH --time=90:00:00                      # maximum execution time requested (HH:MM:SS)
#SBATCH --output=/gpfsscratch/rech/cli/uvo53rl/logs/nerf4ssh_qgimg_128_%j.log      # name of output file
#SBATCH --error=/gpfsscratch/rech/cli/uvo53rl/errs/nerf4ssh_qgimg_128_%j.err       # name of error file
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

## run script (smoke test)
#srun python experiments/qg/main.py \
#    --experiment="image" \
#    --my_config=experiments/qg/config_image.py \
#    --my_config.log.mode="disabled" \
#    --my_config.optim.num_epochs=20 \
#    --my_config.optim.warmup=5 \
#    --my_config.optim_qg.num_epochs=20 \
#    --my_config.optim_qg.warmup=5 \
#    --my_config.trainer.grad_batches=10


# run script
srun python experiments/qg/main.py \
    --my_config=experiments/qg/configs/config.py
