#!/bin/zsh

# Export the conda environment path to PATH
export PATH="/home/davide_frizzo/anaconda3/envs/hf/bin/:$PATH"

SCRIPT_PATH="chronos_forecast.py"

python3 $SCRIPT_PATH \
  --prediction_length 12 \
  --quantile_levels 0.1 0.5 0.9 \
  --device_num 0


