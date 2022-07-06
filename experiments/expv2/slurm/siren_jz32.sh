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
python experiments/expv2/train.py \
    --num_epochs 2000 \
    --wandb_mode "offline" \
    --wandb_log_dir "/gpfsscratch/rech/cli/uvo53rl/" \
    --device "cuda" \
    --train_data_dir "/gpfsdswork/projects/rech/cli/uvo53rl/data/data_challenges/ssh_mapping_2021/train" \
    --ref_data_dir "/gpfsdswork/projects/rech/cli/uvo53rl/data/data_challenges/ssh_mapping_2021/ref" \
    --test_data_dir "/gpfsdswork/projects/rech/cli/uvo53rl/data/data_challenges/ssh_mapping_2021/test" \
    --num_workers 10 \
    --learning_rate 1e-4 \
    --factor 0.25 \
    --lr_scheduler.patience 10 \
    --callbacks.patience 20 \
    --abs_time_min 2016-01-01 \
    --abs_time_max 2019-01-01

#python train.py \
#    --num-epochs 100 \
#    --wandb-mode "offline" \
#    --device "cuda" \
#    --train-data-dir "/gpfsdswork/projects/rech/cli/uvo53rl/data/data_challenges/ssh_mapping_2021/train" \
#    --ref-data-dir "/gpfsdswork/projects/rech/cli/uvo53rl/data/data_challenges/ssh_mapping_2021/ref" \
#    --test-data-dir "/gpfsdswork/projects/rech/cli/uvo53rl/data/data_challenges/ssh_mapping_2021/test" \
#    --dl-num-workers 10 \
#    --learning-rate 1e-3 \
#    --wandb-log-dir "/gpfsscratch/rech/cli/uvo53rl/" \
#    --time-min "2017-01-01" \
#    --time-max "2017-02-01" \
#    --eval-time-min "2017-01-01" \
#    --eval-time-max "2017-02-01" \
#    --eval-dtime "12_h" \
#    --abs-time-min 2016-01-01 \
#    --abs-time-max 2019-01-01

## code execution (TEST)
#srun python experiments/expv2/train.py \
#    --wandb-mode offline \
#    --wandb-log-dir /gpfsscratch/rech/cli/uvo53rl/ \
#    --num-epochs 3000 \
#    --device cuda \
#    --dl-num-workers 10 \
#    --learning-rate 1e-4 \
#    --train-data-dir /gpfsdswork/projects/rech/cli/uvo53rl/data/data_challenges/ssh_mapping_2021/train \
#    --ref-data-dir /gpfsdswork/projects/rech/cli/uvo53rl/data/data_challenges/ssh_mapping_2021/ref \
#    --test-data-dir /gpfsdswork/projects/rech/cli/uvo53rl/data/data_challenges/ssh_mapping_2021/test \
#    --abs-time-min 2016-01-01 \
#    --abs-time-max 2019-01-01


# # code execution
# srun python experiments/dc_2021b/demo_siren_torch.py \
#     --wandb-mode offline \
#     --smoke-test \
#     --log-dir /gpfsscratch/rech/cli/uvo53rl/logs \
#     --model siren \
#     --activation sine \
#     --n-epochs 1200 \
#     --batch-size 4096 \
#     --hidden-dim 512 \
#     --gpus 1 \
#     --num-workers 10 \
#     --learning-rate 1e-4 \
#     --train-data-dir /gpfsdswork/projects/rech/cli/uvo53rl/data/data_challenges/ssh_mapping_2021/train \
#     --ref-data-dir /gpfsdswork/projects/rech/cli/uvo53rl/data/data_challenges/ssh_mapping_2021/ref \
#     --test-data-dir /gpfsdswork/projects/rech/cli/uvo53rl/data/data_challenges/ssh_mapping_2021/test

