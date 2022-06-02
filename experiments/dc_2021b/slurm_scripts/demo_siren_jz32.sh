#!/bin/bash

#SBATCH --job-name=dc21b                     # name of job
#SBATCH --account=cli@v100                   # for statistics
#SBATCH --nodes=1                            # we ALWAYS request one node
#SBATCH --ntasks-per-node=1                  # number of tasks per node
#SBATCH --cpus-per-task=10                   # number of cpus per task
#SBATCH -C v100-32g                          # V100 GPU + 16 GBs RAM
#SBATCH --qos=qos_gpu-t3                     # GPU partition (max 20â hrs)
#SBATCH --gres=gpu:1                         # number of GPUs (1/4 of GPUs)
#SBATCH --time=20:00:00                      # maximum execution time requested (HH:MM:SS)
#SBATCH --output=/gpfswork/rech/cli/uvo53rl/logs/slurm/logs/ml4ssh_dc_2021b_%j.log      # name of output file
#SBATCH --error=/gpfswork/rech/cli/uvo53rl/logs/slurm/errs/ml4ssh_dc_2021b_%j.err       # name of error file
#SBATCH --export=ALL

# loading of modules
module purge

module load cuda/11.2
module load cudnn/9.2-v7.5.1.10
module load git/2.31.1
module load github-cli/1.13.1
module load git-lfs/3.0.2
module load anaconda-py3/2021.05
module load ffmpeg/4.2.2

# go to appropriate directory
cd $WORK/projects/ml4ssh
export PYTHONPATH=$WORK/projects/ml4ssh:${PYTHONPATH}

# loading of modules
source activate jax_gpu_py39

# JAX-related environment variables
export XLA_PYTHON_CLIENT_PREALLOCATE=False
export JAX_PLATFORM_NAME=GPU

# code execution
srun python experiments/dc_2021b/demo_siren.py \
    --wandb-mode offline \
    --log-dir /gpfswork/rech/cli/uvo53rl/logs \
    --model siren \
    --activation sine \
    --n-epochs 1200 \
    --batch-size 4096 \
    --learning-rate 1e-3 \
    --train-data-dir /gpfsdswork/projects/rech/cli/uvo53rl/data/data_challenges/ssh_mapping_2021/train \
    --ref-data-dir /gpfsdswork/projects/rech/cli/uvo53rl/data/data_challenges/ssh_mapping_2021/ref \
    --test-data-dir /gpfsdswork/projects/rech/cli/uvo53rl/data/data_challenges/ssh_mapping_2021/test
    

