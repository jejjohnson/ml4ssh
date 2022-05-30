#!/bin/bash

#SBATCH --job-name=dc21b                     # name of job
#SBATCH --account=python                    # for statistics
#SBATCH --nodes=1                           # we ALWAYS request one node
#SBATCH --ntasks-per-node=1                 # number of tasks per node
#SBATCH --cpus-per-task=10                   # number of cpus per task
#SBATCH --time=12:00:00                      # maximum execution time requested (HH:MM:SS)
#SBATCH --output=/gpfswork/rech/cli/uvo53rl/logs/slurm/logs/ml4ssh_dc_2021b_%j.log      # name of output file
#SBATCH --error=/gpfswork/rech/cli/uvo53rl/logs/slurm/errs/ml4ssh_dc_2021b_%j.err       # name of error file


# loading of modules
module load git
module load cuda/10.2
module load cudnn/9.2-v7.5.1.10

# go to appropriate directory
cd $WORK/projects/ml4ssh
export PYTHONPATH=$WORK/projects/ml4ssh:${PYTHONPATH}

# loading of modules
source activate jax_cpu_py39

# code execution
srun python demo_siren.py \
    --wandb-mode offline \
    --smoke-test True \
    --model mlp \
    --train-data-dir /home/johnsonj/data/dc_2021/raw/train/ \
    --ref-data-dir /home/johnsonj/data/dc_2021/raw/ref/ \
    --test-data-dir /home/johnsonj/data/dc_2021/raw/test/

