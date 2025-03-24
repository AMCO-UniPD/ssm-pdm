#!/bin/bash

MODELS=('S4')
FOLDER=('plots')
DATASET='FD001'
run_id=1
test_quantile=0.25

# cd into the path where the files are stored
cd ../../../ssm_pdm_paper/Img/rul_plots/$MODELS/$DATASET/interval

# Name to give to the RUL plots files when copied locally
rul_plots_name="${MODELS}_${DATASET}_run_${run_id}_quantile_${test_quantile}_interval.pdf"
