"""
Python script to compute the metrics of a quantile regression model
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
    save_element,
    get_current_time,
)
from perf import lifes_metrics
from models import wandb_data

experiment_path = os.path.dirname((os.path.realpath(__file__)))

exp_config, model_config, device, exp_name = setup_exp()

outputs_path = generate_path(
    basepath=experiment_path,
    folders=[
        "outputs",
        exp_config.model_name,
        exp_config.failure_type,
        exp_config.approach,
        exp_name
    ],
)

metrics_path = generate_path(
    basepath=experiment_path,
    folders=[
        "metrics",
        exp_config.model_name,
        exp_config.failure_type,
        exp_config.approach,
        exp_name
    ],
)

setproctitle.setproctitle(f"compute_metrics_{exp_name}")

wandb_data(config=exp_config,model_config=model_config)

for run in range(exp_config.n_runs):

    print("#" * 50)
    print(f"Saving outputs for run: {run+1}")
    print("#" * 50)

    run_metrics_path = generate_path(
        basepath = metrics_path,
        folders = [f"run_{run+1}"]
    )

    run_outputs_path = generate_path(
        basepath = outputs_path,
        folders = [f"run_{run+1}"]
    )

    for quantile in exp_config.quantiles:

        print("#" * 50)
        print(f"Computing metrics for quantile level: {quantile}")
        print("#" * 50)

        quantile_reg_folders = [
            f"quantile_{quantile}",
        ]
        quantile_metrics_path = generate_path(
            basepath=run_metrics_path, folders=quantile_reg_folders
        )
        quantile_outputs_path = generate_path(
            basepath=run_outputs_path, folders=quantile_reg_folders
        )

        metrics_df = lifes_metrics(
            config=exp_config,
            outputs_path=quantile_outputs_path,
            metrics_path=quantile_metrics_path,
        )
