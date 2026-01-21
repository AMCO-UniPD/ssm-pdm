"""
Python script to save the outputs of each quantile level separately
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
from utils import generate_path
from models import best_model_perf

experiment_path = os.path.dirname((os.path.realpath(__file__)))

exp_config, model_config, device, exp_name = setup_exp()

best_model_path = generate_path(
    basepath=experiment_path,
    folders=[
        "best_models",
        exp_config.model_name,
        f"{exp_config.tool_type}_tools",
        exp_config.failure_type,
        exp_config.approach,
        exp_name
    ],
)

outputs_path = generate_path(
    basepath=experiment_path,
    folders=[
        "outputs",
        exp_config.model_name,
        f"{exp_config.tool_type}_tools",
        exp_config.failure_type,
        exp_config.approach,
        exp_name
    ],
)

setproctitle.setproctitle(f"save_outputs_quantile_{exp_name}")

print("#" * 50)
print("Saving outputs for each quantile level separately")
print("#" * 50)

for run in range(exp_config.n_runs):
    print("#" * 50)
    print(f"Saving outputs for run: {run+1}")
    print("#" * 50)

    if exp_config.save_summary_dict:
        exp_config.quantiles = [exp_config.quantile_run]

    for quantile in exp_config.quantiles:
        print("#" * 50)
        print(f"Saving outputs for quantile level: {quantile}")
        print("#" * 50)

        quantile_reg_folders = [
            f"run_{run+1}",
            f"quantile_{quantile}",
        ]
        quantile_outputs_path = generate_path(
            basepath=outputs_path, folders=quantile_reg_folders
        )
        quantile_best_model_path = generate_path(
            basepath=best_model_path, folders=quantile_reg_folders
        )

        best_model_perf(
            config=exp_config,
            model_config=model_config,
            device=device,
            best_model_path=quantile_best_model_path,
            outputs_path=quantile_outputs_path,
            tau=quantile,
        )
