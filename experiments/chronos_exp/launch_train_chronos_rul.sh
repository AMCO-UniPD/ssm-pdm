#!/bin/zsh

# Export the conda environment path to PATH
export PATH="/home/davide_frizzo/anaconda3/envs/hf/bin/:$PATH"

SCRIPT_PATH="train_chronos_rul.py"

python $SCRIPT_PATH \
  --exp_config_path "config/exp_config.yaml"
