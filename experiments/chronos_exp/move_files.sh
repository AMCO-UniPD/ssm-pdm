#!/bin/bash

MODELS=('S4' 'S5' 'S4D' 'LSTM' 'RNN' 'GRU' 'RULTransformer' 'RULInformer')
# MODELS=('S4')
# FOLDER=('best_models','outputs','metrics','plots')
FOLDER=('outputs' 'plots')
DATASET='FD001'
APPROACH='padding'

# Store the initial working directory
INITIAL_DIR=$(pwd)

# Move in the path

for MODEL in "${MODELS[@]}"; do
  for DIR in "${FOLDER[@]}"; do
    path="$INITIAL_DIR/$DIR/$MODEL/$DATASET"
    cd $path
    if [[ -d "$path" ]]; then
      mkdir $APPROACH
      # Move all the files inside the path into the new folder
      files=$(find $path -type f)
      for file in $files; do
        echo "Moving file $file to $APPROACH"
        mv $file $APPROACH
      done
    else
      echo "The path $path does not exist"
    fi

  cd "$INITIAL_DIR"

  done
done
