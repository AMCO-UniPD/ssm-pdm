"""
Python script to save the predictions of
the quantile baseline models on the PHM dataset
for the cross validation case
"""

# general imports
import os
import sys
import setproctitle
import ipdb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.append(src_path)

from exp_config import setup_exp
from utils import (
    MergeData,
    generate_path,
    get_raw_phm_data,
    transform_phm_data,
    get_current_time,
    save_element,
)
from models import load_baseline_model

experiment_path = os.path.dirname((os.path.realpath(__file__)))
exp_config, model_config, device, exp_name = setup_exp()

skf = KFold(
    n_splits = exp_config.n_folds,
    shuffle = True,
    random_state = 42
)

train_phm_data, test_phm_data = get_raw_phm_data(config=exp_config)

_, _, transformed_test_data, _, _, _= transform_phm_data(
    config=exp_config,
    train_data=train_phm_data,
    test_data=test_phm_data
)

merged_phm_data = MergeData(data_list=[train_phm_data, test_phm_data])

for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(merged_phm_data)):

    print("-"*50)
    print(f"Processing fold {fold_idx+1}/{exp_config.n_folds}")
    print("-"*50)

    train_val_data, test_data = merged_phm_data[train_val_idx], merged_phm_data[test_idx]

    print("-"*50)
    print(f"Getting transformed data for {fold_idx+1}/{exp_config.n_folds}")
    print("-"*50)

    transformed_train_data, _, _, _, _, _  = transform_phm_data(
        config  = exp_config,
        train_data =  train_val_data,
        test_data = test_data
    )

    for quantile in exp_config.quantiles:

        print("-"*50)
        print(f"Computing outputs for baseline model with quantile {quantile} for fold {fold_idx+1}")
        print("-"*50)

        baseline_model_name = f"quantile_{quantile}"

        combined_outputs_path = generate_path(
            basepath=experiment_path,
            folders=[
                "baseline_outputs",
                baseline_model_name,
                f"fold_{fold_idx+1}"
            ],
        )

        model = load_baseline_model(model_name=baseline_model_name, tau=quantile)
        model.fit(transformed_train_data)
        baseline_outputs, baseline_true_vals = model.predict_lifes(transformed_test_data)

        if isinstance(baseline_true_vals[0], pd.DataFrame):
            baseline_true_vals = [np.squeeze(baseline_true_vals[i].values) for i in range(len(baseline_true_vals))]

        combined_outputs = {
            "y_pred": baseline_outputs,
            "y_true": baseline_true_vals,
        }

        filename = f"{get_current_time()}_baseline_outputs_quantile_{quantile}_fold_{fold_idx+1}"

        save_element(
            element=combined_outputs,
            dirpath=combined_outputs_path,
            filename=filename,
            filetype="pickle",
        )
