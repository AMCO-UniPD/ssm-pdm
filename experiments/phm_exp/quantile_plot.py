"""
Python script to produce the quantile plots
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
from config_vars import MAX_RUL
from plots import plot_prediction_interval, plot_an_scores
from models import wandb_data

experiment_path = os.path.dirname((os.path.realpath(__file__)))

exp_config, model_config, device, exp_name = setup_exp()

wandb_data(config=exp_config, model_config=model_config)

outputs_path = generate_path(
    basepath=experiment_path,
    folders=[
        "outputs",
        exp_config.model_name,
        exp_config.failure_type,
        exp_config.approach,
        exp_name,
        f"run_{exp_config.run_id}",
    ],
)

plot_path = generate_path(
    basepath=experiment_path,
    folders=[
        "plots",
        exp_config.model_name,
        exp_config.failure_type,
        exp_config.approach,
        exp_name,
        f"run_{exp_config.plot_run_id}"
    ],
)

if exp_config.ad:

    print("-"*50)
    print(f"Producing anomaly score plot for run {exp_config.run_id}")
    print("-"*50)

    plot_an_scores(
        config = exp_config,
        outputs_path = outputs_path,
        plot_path = plot_path,
    )

else:

    print("-"*50)
    print(f"Producing quantile plot for run {exp_config.run_id}")
    print("-"*50)

    _ = plot_prediction_interval(
        config = exp_config,
        outputs_path = outputs_path,
        plot_path = plot_path,
        n_last_samples = MAX_RUL*2
    )
