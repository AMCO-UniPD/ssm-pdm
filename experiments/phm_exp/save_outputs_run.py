"""
Python script to save the prediction on the different quantiles of the same run together
into a single dictionary/dataframe
"""

# general imports
import os
import sys
import ipdb
import torch
import argparse
import setproctitle

src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.append(src_path)

from exp_config import setup_exp
from utils import (
    generate_path,
    open_element,
    save_element,
    get_current_time,
    get_most_recent_file
)
from models import best_model_perf

experiment_path = os.path.dirname((os.path.realpath(__file__)))

exp_config, model_config, device, exp_name = setup_exp()

best_model_path = generate_path(
    basepath=experiment_path,
    folders=[
        "best_models",
        exp_config.model_name,
        exp_config.failure_type,
        exp_config.approach,
        exp_name
    ],
)

outputs_path = generate_path(
    basepath=experiment_path,
    folders=[
        "combined_outputs",
        exp_config.model_name,
        exp_config.failure_type,
        exp_config.approach,
        exp_name
    ],
)

setproctitle.setproctitle(f"save_outputs_run_{exp_name}")

print("-" * 50)
print("Save the prediction on the different quantiles of the same run together")
print("-" * 50)

run_iterator = range(exp_config.n_folds) if exp_config.cv else range(exp_config.start_run_id, exp_config.n_runs)

for run in run_iterator:

    print("-" * 50)
    print(f"Saving outputs for run: {run+1}")
    print("-" * 50)

    run_folders = [f"run_{run+1}"] if not exp_config.cv else [f"fold_{run+1}"]
    run_outputs_path = generate_path(
        basepath=outputs_path, folders=run_folders
    )
    best_model_run_path = generate_path(
        basepath=best_model_path, folders=run_folders
    )

    run_outputs_dict = {}

    for i, quantile in enumerate(exp_config.quantiles):

        print("-" * 50)
        print(f"Saving outputs for quantile: {quantile}")
        print("-" * 50)

        outputs_dict_path = get_most_recent_file(os.path.join(run_outputs_path,f"quantile_{quantile}"),file_pos=exp_config.file_pos)
        outputs_dict = open_element(outputs_dict_path,filetype="pickle")

        y_pred, y_true = outputs_dict["y_pred"], outputs_dict["y_true"]

        if i == 0:
            run_outputs_dict["y_true"] = y_true

        run_outputs_dict[f"pred_quantile_{quantile}"] = y_pred

    filename = f"{get_current_time()}_{exp_config.model_name}_{exp_config.cmapss_models}_{exp_config.approach}_run_{run+1}_outputs" if exp_config.data_name == "CMAPSS" else f"{get_current_time()}_{exp_config.model_name}_{exp_config.approach}_run_{run+1}_outputs"

    save_element(
        element=run_outputs_dict,
        dirpath=run_outputs_path,
        filename=filename,
        filetype="pickle",
    )



