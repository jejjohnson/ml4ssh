#!/bin/bash

# go to appropriate directory
cd /gpfswork/rech/cli/uvo53rl/projects/inr4ssh/
export PYTHONPATH=/gpfswork/rech/cli/uvo53rl/projects/inr4ssh/:${PYTHONPATH}

# loading of modules
conda activate torch_py39

## run script (smoke-test: subset)
#python experiments/dc20a/main.py \
#    --stage="download" \
#    --my_config=experiments/dc20a/configs/config.py \
#    --dldir=/gpfswork/rech/cli/uvo53rl/data/data_challenges/ssh_mapping_2020a/raw

# run script (smoke-test: subset)
python experiments/dc20a/main.py \
    --stage="preprocess" \
    --my_config=experiments/dc20a/configs/config.py

# run script (smoke-test: subset)
python experiments/dc20a/main.py \
    --stage="ml_ready" \
    --my_config=experiments/dc20a/configs/config.py
