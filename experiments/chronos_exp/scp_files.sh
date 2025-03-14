#!/bin/zsh

MODELS=('S4')
FOLDER=('plots')
DATASET='FD001'
APPROACH='windowed'
EXP_NAME='13-03-2025_10-48-53_S4_FD001_windowed_quantile_reg'
FILE_NAME='13-03-2025_12-14-23_S4_FD001_quantile_reg_global_predictions_grid_pad_life_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14.pdf'

# Store the initial working directory
INITIAL_DIR=$(pwd)

path="$INITIAL_DIR/$FOLDER/$MODELS/$DATASET/$APPROACH/quantile_reg/$EXP_NAME/$FILE_NAME"
cd $path

echo "I am in $path"




