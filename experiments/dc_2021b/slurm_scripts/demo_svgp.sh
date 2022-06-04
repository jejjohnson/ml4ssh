#!/bin/bash

#SBATCH --job-name=dc21b                     # name of job
#SBATCH --account=cli@v100                   # for statistics
#SBATCH --nodes=1                            # we ALWAYS request one node
#SBATCH --ntasks-per-node=1                  # number of tasks per node
#SBATCH --cpus-per-task=10                   # number of cpus per task
#SBATCH -C v100-32g                          # V100 GPU + 16 GBs RAM
#SBATCH --qos=qos_gpu-t3                     # GPU partition (max 20� hrs)
#SBATCH --gres=gpu:1                         # number of GPUs (1/4 of GPUs)
#SBATCH --time=20:00:00                      # maximum execution time requested (HH:MM:SS)
#SBATCH --output=/gpfsscratch/rech/cli/uvo53rl/logs/ml4ssh_gp_dc_2021b_%j.log      # name of output file
#SBATCH --error=/gpfsscratch/rech/cli/uvo53rl/errs/ml4ssh_gp_dc_2021b_%j.err       # name of error file
#SBATCH --export=ALL

# loading of modules
module purge

module load cuda/11.2
module load cudnn/9.2-v7.5.1.10
module load git/2.31.1
module load github-cli/1.13.1
module load git-lfs/3.0.2
module load anaconda-py3/2021.05

# go to appropriate directory
cd $WORK/projects/ml4ssh
export PYTHONPATH=$WORK/projects/ml4ssh:${PYTHONPATH}

# loading of modules
source activate jaxtf_gpu_py39

# code execution
srun python experiments/dc_2021b/demo_svgp.py \
    --wandb-mode offline \
    --smoke-test \
    --log-dir /gpfsscratch/rech/cli/uvo53rl/logs \
    --model svgp \
    --kernel rbf \
    --likelihood gaussian \
    --n-inducing 2000 \
    --learning-rate 1e-3 \
    --learning-rate-ng 1e-1 \
    --train-data-dir /gpfsdswork/projects/rech/cli/uvo53rl/data/data_challenges/ssh_mapping_2021/train \
    --ref-data-dir /gpfsdswork/projects/rech/cli/uvo53rl/data/data_challenges/ssh_mapping_2021/ref \
    --test-data-dir /gpfsdswork/projects/rech/cli/uvo53rl/data/data_challenges/ssh_mapping_2021/test
    

