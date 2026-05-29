"""
Python script to save the predictions of
the quantile baseline models on the PHM dataset
"""

# general imports
import os
import sys
import setproctitle
import ipdb
import numpy as np
import pandas as pd

src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.append(src_path)

from exp_config import setup_exp
from utils import generate_path, transform_phm_data, get_current_time, save_element
from models import load_baseline_model

experiment_path = os.path.dirname((os.path.realpath(__file__)))
exp_config, model_config, device, exp_name = setup_exp()

transformed_train_data, _, transformed_test_data, _, _, _ = transform_phm_data(config=exp_config)

for quantile in exp_config.quantiles:

    print("-"*50)
    print(f"Computing outputs for baseline model with quantile {quantile}")
    print("-"*50)

    baseline_model_name = f"quantile_{quantile}"

    combined_outputs_path = generate_path(
        basepath=experiment_path,
        folders=[
            "baseline_outputs",
            baseline_model_name,
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

    filename = f"{get_current_time()}_baseline_outputs_quantile_{quantile}"

    save_element(
        element=combined_outputs,
        dirpath=combined_outputs_path,
        filename=filename,
        filetype="pickle",
    )

