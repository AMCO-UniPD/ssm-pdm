#!/bin/bash

MODELS=('LSTM')
FOLDER=('plots')
DATASET='FD001'
APPROACH='windowed'
N_RUNS=5
EXP_NAME='multi_run_14-03-2025_11-52-58_LSTM_FD001_windowed_quantile_reg'
FILE_NAME='13-03-2025_12-14-23_S4_FD001_quantile_reg_global_predictions_grid_pad_life_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14.pdf'

# Store the initial part of the path in acquario3
INITIAL_DIR='/home/davide_frizzo/chronos-pdm/experiments/chronos_exp'

acquario3_path="$INITIAL_DIR/$FOLDER/$MODELS/$DATASET/$APPROACH/quantile_reg/$EXP_NAME/"

# cd into local path in my local machine
cd ../../../ssm_pdm_paper/Img/rul_plots/LSTM/

# Iterate over the runs
for i in $(seq 1 $N_RUNS); do
  echo "############################################"
  echo "Copying the file for run $i"
  echo "############################################"
  run='run_'$i
  acquario3_path_run="$acquario3_path$run"
  latest_file=$(ssh acquario3 "find $acquario3_path_run -name '*.pdf' -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d' '") 
  scp acquario3:$latest_file .
  echo "############################################"
  echo "latest_file path: $latest_file"
  echo "############################################"
done

# echo "############################################"
# echo "Copying the file from acquario 3 in the current path"
# echo "############################################"

# Copy the file from acquario3 to my local machine
# scp acquario3:$acquario3_path .

# echo "############################################"
# echo "acquario3_path: $acquario3_path"
# echo "############################################"

# echo "############################################"
# echo "I am in path: $(pwd)"
# # echo "Files in this path: $(ls)"
# echo "############################################"
