"""
Python script to compute the average predictions
for the baseline models over the different folds
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
from utils import (
    generate_path,
    get_most_recent_file,
    get_current_time,
    open_element,
    save_element,
)

experiment_path = os.path.dirname((os.path.realpath(__file__)))
exp_config, model_config, device, exp_name = setup_exp()

for baseline_model_name in exp_config.baseline_model_names:

    baseline_outputs_list = []

    print("-"*50)
    print(f"Computing average baseline outputs for baseline model {baseline_model_name}")
    print("-"*50)

    for fold_idx in range(exp_config.n_folds):

        baseline_outputs_path = generate_path(
            basepath = experiment_path,
            folders = [
                "baseline_outputs",
                baseline_model_name,
            ]
        )
        fold_baseline_outputs_path = generate_path(
            basepath = baseline_outputs_path,
            folders = [
                f"fold_{fold_idx+1}",
            ]
        )
        baseline_outputs_filepath = get_most_recent_file(fold_baseline_outputs_path)
        outputs_dict = open_element(baseline_outputs_filepath,filetype="pickle")
        baseline_outputs_list.append(outputs_dict["y_pred"])

    mean_baseline_outputs = []

    for life in range(len(baseline_outputs_list[0])):

        pred_across_folds = [
            fold[life] for fold in baseline_outputs_list
        ]
        mean_pred = np.mean(pred_across_folds, axis=0)
        mean_baseline_outputs.append(mean_pred)

    filename=f"{get_current_time()}_mean_baseline_outputs_{baseline_model_name}"

    save_element(
        element=mean_baseline_outputs,
        dirpath=baseline_outputs_path,
        filename=filename,
        filetype="pickle",
    )
