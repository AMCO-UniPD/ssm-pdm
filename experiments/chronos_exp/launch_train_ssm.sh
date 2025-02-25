#!/bin/zsh

# Export the conda environment path to PATH
export PATH="/home/davide_frizzo/anaconda3/envs/hf/bin/:$PATH"

SCRIPT_PATH="train_ssm.py"

echo "###########################################"
echo "Executing $SCRIPT_PATH"
echo "###########################################"

python $SCRIPT_PATH \
  --exp_config_path "config/ssm_exp_config.yaml"
