#!/bin/bash

MODELS=('RULInformer')
FOLDER=('plots')
DATASET='FD002'
APPROACH='windowed'
# quantile_approach='quantile_reg'
# quantile_approach='tau_mult'
quantile_approach='tau_mult_no_tau_feat'
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
#
# exp_name=multi_run_22-03-2025_15-41-44_S4_FD001_windowed_quantile_reg
# exp_name=multi_run_23-03-2025_09-14-06_S5_FD001_windowed_quantile_reg
# exp_name=multi_run_23-03-2025_16-52-06_S4D_FD001_windowed_quantile_reg
# exp_name=multi_run_24-03-2025_08-12-23_LSTM_FD001_windowed_quantile_reg
# exp_name=multi_run_24-03-2025_17-00-56_RULTransformer_FD001_windowed_quantile_reg
# exp_name=multi_run_25-03-2025_08-31-33_RULInformer_FD001_windowed_quantile_reg

# Experiment names FD001 real tau multiplicative
# exp_name=multi_run_26-03-2025_17-48-45_S4_FD001_windowed_quantile_reg_tau_mult
# exp_name=multi_run_27-03-2025_14-23-46_S5_FD001_windowed_quantile_reg_tau_mult
# exp_name=multi_run_27-03-2025_17-30-47_S4D_FD001_windowed_quantile_reg_tau_mult

# Experiment names FD001 real tau multiplicative + no_tau_float
# exp_name=29-03-2025_10-07-53_S4_FD001_windowed_quantile_reg_tau_mult_no_tau_feat
# exp_name=multi_run_29-03-2025_12-31-34_S5_FD001_windowed_quantile_reg_tau_mult_no_tau_feat
# exp_name=multi_run_29-03-2025_14-29-45_RULTransformer_FD001_windowed_quantile_reg_tau_mult_no_tau_feat
# exp_name=multi_run_29-03-2025_16-39-17_S4D_FD001_windowed_quantile_reg_tau_mult_no_tau_feat
# exp_name=multi_run_29-03-2025_17-44-54_RULInformer_FD001_windowed_quantile_reg_tau_mult_no_tau_feat
# exp_name=multi_run_30-03-2025_07-32-15_LSTM_FD001_windowed_quantile_reg_tau_mult_no_tau_feat

# Experiment names - FD002 dataset
# exp_name=multi_run_18-03-2025_07-58-34_S4_FD002_windowed_quantile_reg
# exp_name=multi_run_18-03-2025_10-58-34_S5_FD002_windowed_quantile_reg
# exp_name=multi_run_18-03-2025_15-32-30_S4D_FD002_windowed_quantile_reg
# exp_name=multi_run_18-03-2025_18-13-05_RULInformer_FD002_windowed_quantile_reg
# exp_name=multi_run_19-03-2025_15-00-46_RULTransformer_FD002_windowed_quantile_reg
# exp_name=multi_run_19-03-2025_21-27-39_LSTM_FD002_windowed_quantile_reg

# Experiment names FD002 dataset real tau multiplicative
# exp_name=multi_run_28-03-2025_08-57-13_S4_FD002_windowed_quantile_reg_tau_mult
# exp_name=multi_run_28-03-2025_12-35-12_S5_FD002_windowed_quantile_reg_tau_mult
# exp_name=multi_run_29-03-2025_07-30-40_S4D_FD002_windowed_quantile_reg_tau_mult

# Experiment names FD002 dataset real tau multiplicative + no_tau_float
# exp_name=multi_run_30-03-2025_08-53-20_S4_FD002_windowed_quantile_reg_tau_mult_no_tau_feat
# exp_name=multi_run_30-03-2025_13-50-18_S5_FD002_windowed_quantile_reg_tau_mult_no_tau_feat
# exp_name=multi_run_31-03-2025_10-59-36_S4D_FD002_windowed_quantile_reg_tau_mult_no_tau_feat
# exp_name=multi_run_31-03-2025_15-10-51_RULTransformer_FD002_windowed_quantile_reg_tau_mult_no_tau_feat
# exp_name=multi_run_01-04-2025_08-30-46_LSTM_FD002_windowed_quantile_reg_tau_mult_no_tau_feat
exp_name=multi_run_01-04-2025_10-36-18_RULInformer_FD002_windowed_quantile_reg_tau_mult_no_tau_feat


# Store the initial part of the path in acquario3
INITIAL_DIR='/home/davide_frizzo/chronos-pdm/experiments/chronos_exp'

# acquario3_path="$INITIAL_DIR/$FOLDER/$MODELS/$DATASET/$APPROACH/quantile_reg/$exp_name/"
# Path for Prediction Interval plots
acquario3_path="$INITIAL_DIR/$FOLDER/$MODELS/$DATASET/$APPROACH/quantile_reg/$exp_name/interval/"

# blob plot path
# blob_plot_path="$INITIAL_DIR/$FOLDER/blob_plot/$DATASET/"

# cd into local path in my local machine
# cd ../../../ssm_pdm_paper/Img/rul_plots/$MODELS/$DATASET/
# path for prediction interval plots
cd ../../../ssm_pdm_paper/Img/interval_plots/$DATASET/
# path for blob plots
# cd ../../../ssm_pdm_paper/Img/blob_plots/$DATASET/

# if folder interval does not exist create it and cd into it
subfolder=$quantile_approach

if [ ! -d $subfolder ]; then
  echo "Creating the folder $subfolder"
  mkdir $subfolder
  cd $subfolder
else
  cd $subfolder
fi

# Name to give to the RUL plots files when copied locally

# rul_plots_name="${MODELS}_${DATASET}_quantile_reg_run_"
# quantiles_names="quantiles_0_25_0_5_0_75"

# file_pos=(1 2)
file_pos=(1)

# Iterate over the runs
for pos in "${file_pos[@]}"; do
  # run='run_'$i
  # acquario3_path_run="$acquario3_path$run"

  # latest_file=$(ssh acquario3 "find $acquario3_path -name '*.pdf' -type f -printf '%T@ %p\n' | sort -n | tail -$pos | head -1 | cut -f2- -d' '") 
  # echo "############################################"
  # echo "Copying file: $latest_file"
  # echo "############################################"
  # scp acquario3:$latest_file .

  # prediction interval plots
  latest_file=$(ssh acquario3 "find $acquario3_path -name '*.pdf' -type f -printf '%T@ %p\n' | sort -n | tail -$pos | head -1 | cut -f2- -d' '") 
  # blob plots
  # latest_file=$(ssh acquario3 "find $blob_plot_path -name '*.png' -type f -printf '%T@ %p\n' | sort -n | tail -$pos | head -1 | cut -f2- -d' '") 
  echo "############################################"
  echo "Copying file: $latest_file"
  echo "############################################"
  scp acquario3:$latest_file .

  # files=$(ssh acquario3 "cd $blob_plot_path && ls")
  # echo "############################################"
  # echo "Files in $blob_plot_path: $files"
  # echo "############################################"
 
  # acquario3_path_files=$(ssh acquario3 "cd $acquario3_path && ls")
  # echo "############################################"
  # echo "Files in acquario3 path: $acquario3_path_files"
  # echo "############################################"
  # scp -r acquario3:$acquario3_path .

  # rul_plots_name_run="$rul_plots_name${i}_${quantiles_names}.pdf"
  # mv $(basename $latest_file) $rul_plots_name_run

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
