#!/bin/zsh

# Export the conda environment path to PATH
export PATH="/home/davide_frizzo/anaconda3/envs/hf/bin/:$PATH"

SCRIPT_PATH="blob_plot.py"

echo "###########################################"
echo "Executing $SCRIPT_PATH"
echo "###########################################"

# NOTE: FD001

exp_names=(
  "multi_run_26-03-2025_17-48-45_S4_FD001_windowed_quantile_reg_tau_mult"
  "multi_run_27-03-2025_14-23-46_S5_FD001_windowed_quantile_reg_tau_mult"
  "multi_run_27-03-2025_17-30-47_S4D_FD001_windowed_quantile_reg_tau_mult"
  "multi_run_24-03-2025_08-12-23_LSTM_FD001_windowed_quantile_reg"
  "multi_run_24-03-2025_17-00-56_RULTransformer_FD001_windowed_quantile_reg"
  "multi_run_25-03-2025_08-31-33_RULInformer_FD001_windowed_tau_mult"
  )

# # NOTE: FD001
# exp_names=(
#   "multi_run_27-03-2025_17-30-47_S4D_FD001_windowed_quantile_reg_tau_mult"
#   )
#
# NOTE: FD002
# exp_names=(
#   "multi_run_28-03-2025_08-57-13_S4_FD002_windowed_quantile_reg_tau_mult"
#   "multi_run_28-03-2025_12-35-12_S5_FD002_windowed_quantile_reg_tau_mult"
#   "multi_run_29-03-2025_07-30-40_S4D_FD002_windowed_quantile_reg_tau_mult"
#   "multi_run_10-07-2025_16-34-55_LSTM_FD002_windowed_quantile_reg_tau_mult"
#   "multi_run_10-07-2025_17-13-14_RULTransformer_FD002_windowed_quantile_reg_tau_mult"
#   "multi_run_10-07-2025_17-28-16_RULInformer_FD002_windowed_quantile_reg_tau_mult
#   ")

python $SCRIPT_PATH \
  --exp_names "${exp_names[@]}" \
  --no_mult_adds

