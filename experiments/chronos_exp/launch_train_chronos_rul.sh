#!/bin/zsh

# Export the conda environment path to PATH
export PATH="/home/davide_frizzo/anaconda3/envs/hf/bin/:$PATH"

SCRIPT_PATH="train_chronos_rul.py"
TEST_PATH="tests.py"

echo "###########################################"
echo "Executing $SCRIPT_PATH"
echo "###########################################"

python $SCRIPT_PATH \
  --exp_config_path "config/exp_config.yaml"

# echo "###########################################"
# echo "Executing $TEST_PATH"
# echo "###########################################"
#
# python $TEST_PATH
