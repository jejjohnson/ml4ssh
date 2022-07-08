# go to appropriate directory
cd $WORK/projects/inr4ssh
export PYTHONPATH=$WORK/projects/inr4ssh:${PYTHONPATH}

# loading of modules
source activate torch_py39

python experiments/expv2/train.py \
    --num_epochs 1000 \
    --wandb_mode "offline" \
    --wandb_log_dir "/gpfsscratch/rech/cli/uvo53rl/" \
    --device "cuda" \
    --train_data_dir "/gpfsdswork/projects/rech/cli/uvo53rl/data/data_challenges/ssh_mapping_2021/train" \
    --ref_data_dir "/gpfsdswork/projects/rech/cli/uvo53rl/data/data_challenges/ssh_mapping_2021/ref" \
    --test_data_dir "/gpfsdswork/projects/rech/cli/uvo53rl/data/data_challenges/ssh_mapping_2021/test" \
    --num_workers 10 \
    --learning_rate 1e-4 \
    --preprocess.time_min "2017-01-01" \
    --preprocess.time_max "2017-02-01" \
    --factor 0.25 \
    --lr_scheduler.patience 10 \
    --callbacks.patience 20 \
    --eval.time_min "2017-01-01" \
    --eval.time_max "2017-02-01" \
    --eval.dtime_freq 12 \
    --eval.dtime_unit "h" \
    --abs_time_min 2016-01-01 \
    --abs_time_max 2019-01-01