#!/bin/bash

MODELS=('S4')
FOLDER=('plots')
DATASET='FD001'
APPROACH='windowed'
N_RUNS=5

# Useful experiment names
exp_name=multi_run_14-03-2025_08-20-13_S4_FD001_windowed_quantile_reg
# exp_name=multi_run_14-03-2025_09-43-44_S5_FD001_windowed_quantile_reg
# exp_name=multi_run_14-03-2025_10-56-41_S4D_FD001_windowed_quantile_reg
# exp_name=multi_run_14-03-2025_11-52-58_LSTM_FD001_windowed_quantile_reg
# exp_name=multi_run_14-03-2025_15-09-39_RULTransformer_FD001_windowed_quantile_reg
# exp_name=multi_run_14-03-2025_16-37-39_RULInformer_FD001_windowed_quantile_reg

# Store the initial part of the path in acquario3
INITIAL_DIR='/home/davide_frizzo/chronos-pdm/experiments/chronos_exp'

acquario3_path="$INITIAL_DIR/$FOLDER/$MODELS/$DATASET/$APPROACH/quantile_reg/$exp_name/"

# cd into local path in my local machine
cd ../../../ssm_pdm_paper/Img/rul_plots/$MODELS/

# Name to give to the RUL plots files when copied locally

rul_plots_name="${MODELS}_${DATASET}_quantile_reg_run_"
quantiles_names="quantiles_0_1_0_25_0_75"

# Iterate over the runs
for i in $(seq 1 $N_RUNS); do
  echo "############################################"
  echo "Copying the file for run $i"
  echo "############################################"
  run='run_'$i
  acquario3_path_run="$acquario3_path$run"

  latest_file=$(ssh acquario3 "find $acquario3_path_run -name '*.pdf' -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d' '") 
  scp acquario3:$latest_file .

  rul_plots_name_run="$rul_plots_name${i}_${quantiles_names}.pdf"
  mv $(basename $latest_file) $rul_plots_name_run
  # echo "############################################"
  # echo "latest_file path: $latest_file"
  # echo "############################################"
  # echo "############################################"
  # echo "basename: $(basename $latest_file)"
  echo "############################################"
  echo "Copied the file for run $i"
  echo "############################################"
done

# cd to_rename
# quantiles_names="quantiles_0_1_0_5_0_9"
#
# for i in $(seq 1 $N_RUNS); do
#   # Find the file whose name contains run_$i
#   run_file=$(find . -name "*run_$i*.pdf" -type f)
#   rul_plots_name_run="$rul_plots_name${i}_${quantiles_names}.pdf"
#   mv $(basename $run_file) $rul_plots_name_run
#   echo "############################################"
#   echo "Renamed ${run_file} to ${rul_plots_name_run}"
#   echo "############################################"
#
# done
