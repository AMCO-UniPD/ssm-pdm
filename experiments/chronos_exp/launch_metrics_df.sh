#!/bin/zsh

# Export the conda environment path to PATH
export PATH="/home/davide_frizzo/anaconda3/envs/hf/bin/:$PATH"

SCRIPT_PATH="quantile_reg_metrics.py"

echo "###########################################"
echo "Executing $SCRIPT_PATH"
echo "###########################################"

python $SCRIPT_PATH
