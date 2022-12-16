#!/bin/bash

# go to appropriate directory
cd $HOME/code_projects/inr4ssh/
export PYTHONPATH=$HOME/code_projects/inr4ssh/:${PYTHONPATH}

# loading of modules
conda activate torch_py39

# run script (smoke-test: subset)
python experiments/dc21a/main.py \
    --stage="ml_ready" \
    --my_config=experiments/dc21a/configs/config.py
