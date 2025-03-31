#!/bin/zsh

# Export the conda environment path to PATH
export PATH="/home/davide_frizzo/anaconda3/envs/hf/bin/:$PATH"

SCRIPT_PATH="parallel_metrics.py"

MODELS=("S5" "RULTransformer" "RULInformer")
# MODELS=("S4")

for MODEL in "${MODELS[@]}"; do
  
  echo "###########################################"
  echo "Executing $SCRIPT_PATH for model $MODEL"
  echo "###########################################"

  python $SCRIPT_PATH \
    --model_name $MODEL \
    --file_pos 1
  done
