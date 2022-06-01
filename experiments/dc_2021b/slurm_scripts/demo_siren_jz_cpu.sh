#!/bin/bash

#SBATCH --job-name=dc21b                     # name of job
#SBATCH --account=cli@cpu                    # for statistics
#SBATCH --nodes=1                           # we ALWAYS request one node
#SBATCH --ntasks-per-node=1                 # number of tasks per node
#SBATCH --cpus-per-task=32                   # number of cpus per task
#SBATCH --time=20:00:00                      # maximum execution time requested (HH:MM:SS)
#SBATCH --output=/gpfswork/rech/cli/uvo53rl/logs/slurm/logs/ml4ssh_dc_2021b_%j.log      # name of output file
#SBATCH --error=/gpfswork/rech/cli/uvo53rl/logs/slurm/errs/ml4ssh_dc_2021b_%j.err       # name of error file
#SBATCH --export=ALL

# loading of modules
module purge

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

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORM_NAME=CPU

# code execution
srun python experiments/dc_2021b/demo_siren.py \
    --wandb-mode offline \
    --log-dir /gpfswork/rech/cli/uvo53rl/logs \
    --model siren \
    --n-epochs 500 \
    --batch-size 4096 \
    --train-data-dir /gpfsdswork/projects/rech/cli/uvo53rl/data/data_challenges/ssh_mapping_2021/train \
    --ref-data-dir /gpfsdswork/projects/rech/cli/uvo53rl/data/data_challenges/ssh_mapping_2021/ref \
    --test-data-dir /gpfsdswork/projects/rech/cli/uvo53rl/data/data_challenges/ssh_mapping_2021/test
    

