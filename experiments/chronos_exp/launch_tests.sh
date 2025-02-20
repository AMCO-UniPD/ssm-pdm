#!/bin/zsh

# Export the conda environment path to PATH
export PATH="/home/davide_frizzo/anaconda3/envs/hf/bin/:$PATH"

SCRIPT_PATH="tests.py"

python $SCRIPT_PATH
