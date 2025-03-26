#!/bin/bash

MODELS=('RULInformer')
FOLDER=('plots')
DATASET='FD001'
APPROACH='windowed'
N_RUNS=1

# Experiment names FD001 dataset
# exp_name=multi_run_14-03-2025_08-20-13_S4_FD001_windowed_quantile_reg
# exp_name=multi_run_14-03-2025_09-43-44_S5_FD001_windowed_quantile_reg
# exp_name=multi_run_14-03-2025_10-56-41_S4D_FD001_windowed_quantile_reg
# exp_name=multi_run_14-03-2025_11-52-58_LSTM_FD001_windowed_quantile_reg
# exp_name=multi_run_14-03-2025_15-09-39_RULTransformer_FD001_windowed_quantile_reg
# exp_name=multi_run_14-03-2025_16-37-39_RULInformer_FD001_windowed_quantile_reg
#
# Experiment names FD001 tau multiplicative
# exp_name=multi_run_22-03-2025_15-41-44_S4_FD001_windowed_quantile_reg
# exp_name=multi_run_23-03-2025_09-14-06_S5_FD001_windowed_quantile_reg
# exp_name=multi_run_23-03-2025_16-52-06_S4D_FD001_windowed_quantile_reg
# exp_name=multi_run_24-03-2025_08-12-23_LSTM_FD001_windowed_quantile_reg
# exp_name=multi_run_24-03-2025_17-00-56_RULTransformer_FD001_windowed_quantile_reg
exp_name=multi_run_25-03-2025_08-31-33_RULInformer_FD001_windowed_quantile_reg


# Experiment names FD002 dataset
# exp_name=multi_run_18-03-2025_07-58-34_S4_FD002_windowed_quantile_reg

# Store the initial part of the path in acquario3
INITIAL_DIR='/home/davide_frizzo/chronos-pdm/experiments/chronos_exp'

# acquario3_path="$INITIAL_DIR/$FOLDER/$MODELS/$DATASET/$APPROACH/quantile_reg/$exp_name/"
# Path for Prediction Interval plots
acquario3_path="$INITIAL_DIR/$FOLDER/$MODELS/$DATASET/$APPROACH/quantile_reg/$exp_name/interval/"

# cd into local path in my local machine
# cd ../../../ssm_pdm_paper/Img/rul_plots/$MODELS/$DATASET/
cd ../../../ssm_pdm_paper/Img/interval_plots

# if folder interval does not exist create it and cd into it
# subfolder="interval"
#
# if [ ! -d $subfolder ]; then
#   echo "Creating the folder $subfolder"
#   mkdir $subfolder
#   cd $subfolder
# else
#   cd $subfolder
# fi

# Name to give to the RUL plots files when copied locally

# rul_plots_name="${MODELS}_${DATASET}_quantile_reg_run_"
# quantiles_names="quantiles_0_25_0_5_0_75"

file_pos=(1 2)
# file_pos=(1)

# Iterate over the runs
for pos in "${file_pos[@]}"; do
  # echo "############################################"
  # echo "Copying the file for pos $pos"
  # echo "############################################"
  # run='run_'$i
  # acquario3_path_run="$acquario3_path$run"

  latest_file=$(ssh acquario3 "find $acquario3_path -name '*.pdf' -type f -printf '%T@ %p\n' | sort -n | tail -$pos | head -1 | cut -f2- -d' '") 
  echo "############################################"
  echo "Copying file: $latest_file"
  echo "############################################"
  scp acquario3:$latest_file .
 
  acquario3_path_files=$(ssh acquario3 "cd $acquario3_path && ls")
  # scp -r acquario3:$acquario3_path .

  # rul_plots_name_run="$rul_plots_name${i}_${quantiles_names}.pdf"
  # mv $(basename $latest_file) $rul_plots_name_run

  # echo "############################################"
  # echo "basename: $(basename $latest_file)"
  # echo "############################################"
  # echo "############################################"
  # echo "Files in acquario3 path: $acquario3_path_files"
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
