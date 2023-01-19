#!/bin/bash
#SBATCH --job-name=dc20a                     # name of job
#SBATCH --account=yrf@v100                   # for statistics
#SBATCH --nodes=1                            # we ALWAYS request one node
#SBATCH --ntasks-per-node=1                  # number of tasks per node
#SBATCH --cpus-per-task=40                   # number of cpus per task
#SBATCH -C v100-16g                          # V100 GPU + 16 GBs RAM
#SBATCH --gres=gpu:4                         # number of GPUs (4/4 of GPUs)
#SBATCH --qos=qos_gpu-t3                     # GPU partition (max 20 hrs)
#SBATCH --time=20:00:00                      # maximum execution time requested (HH:MM:SS)
#SBATCH --output=/gpfsscratch/rech/cli/uvo53rl/logs/nerf4ssh_dc20a_%j.log      # name of output file
#SBATCH --error=/gpfsscratch/rech/cli/uvo53rl/errs/nerf4ssh_dc20a_%j.err       # name of error file
#SBATCH --export=ALL
#SBATCH --signal=SIGUSR1@90

# purge all modules
module purge

#module load cuda/11.2
#module load cudnn/9.2-v7.5.1.10
module load cudnn/10.1-v7.5.1.10
module load git/2.31.1
module load github-cli/1.13.1
module load git-lfs/3.0.2
module load ffmpeg/4.2.2

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# go to appropriate directory
cd /gpfswork/rech/cli/uvo53rl/projects/inr4ssh/
export PYTHONPATH=/gpfswork/rech/cli/uvo53rl/projects/inr4ssh:${PYTHONPATH}

# loading of modules
source activate torch_py39

srun python experiments/dc20a/main.py \
    --stage="train" \
    --my_config=experiments/dc20a/configs/config.py \
    --my_config.experiment="swot1nadir5" \
    --my_config.trainer.num_epochs=10000 \
    --my_config.lr_scheduler.warmup_epochs=500 \
    --my_config.lr_scheduler.max_epochs=100 \
    --my_config.lr_scheduler.eta_min=1e-6 \
    --my_config.log.mode="offline" \
    --my_config.model.hidden_dim=256 \
    --my_config.trainer.accelerator="gpu" \
    --my_config.trainer.strategy="dp" \
    --my_config.trainer.devices=4 \
    --my_config.trainer.grad_batches=1 \
    --my_config.dataloader.num_workers_train=40 \
    --my_config.dataloader.num_workers_valid=40 \
    --my_config.dataloader.num_workers_test=40 \
    --my_config.dataloader.num_workers_predict=40
#    --my_config.evaluation.lon_coarsen=5 \
#    --my_config.evaluation.lat_coarsen=5 \
