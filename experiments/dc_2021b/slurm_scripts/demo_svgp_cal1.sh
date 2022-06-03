#!/bin/bash

#SBATCH --job-name=gpdcb                     # name of job
#SBATCH --export=ALL
#SBATCH --account=python                    # for statistics
#SBATCH --nodes=1                           # we ALWAYS request one node
#SBATCH --ntasks-per-node=1                 # number of tasks per node
#SBATCH --cpus-per-task=4  
#SBATCH --output=/mnt/meom/workdir/johnsonj/logs/slurm/logs/ml4ssh_gp_dc_2021b_%j.log      # name of output file
#SBATCH --error=/mnt/meom/workdir/johnsonj/logs/slurm/errs/ml4ssh_gp_dc_2021b_%j.err       # name of error file


# go to appropriate directory
cd $WORKDIR/projects/ml4ssh
export PYTHONPATH=$WORKDIR/projects/ml4ssh:${PYTHONPATH}

# # remove previous log files
# rm /mnt/meom/workdir/johnsonj/logs/slurm/logs/ml4ssh_dc_2021b_*
# rm /mnt/meom/workdir/johnsonj/logs/slurm/errs/ml4ssh_dc_2021b_*

# loading of modules
source activate jaxtf_cpu_py39

# code execution
srun python experiments/dc_2021b/demo_svgp.py \
    --wandb-mode disabled \
    --smoke-test \
    --log-dir /mnt/meom/workdir/johnsonj/logs \
    --model svgp \
    --kernel rbf \
    --likelihood gaussian \
    --n-inducing 2000 \
    --learning-rate 1e-3 \
    --learning-rate-ng 1e-1 \
    --train-data-dir /home/johnsonj/data/dc_2021/raw/train/ \
    --ref-data-dir /home/johnsonj/data/dc_2021/raw/ref/ \
    --test-data-dir /home/johnsonj/data/dc_2021/raw/test/
    
