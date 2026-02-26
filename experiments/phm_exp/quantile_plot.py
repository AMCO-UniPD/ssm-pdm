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
from utils import generate_path, print_life_info
from config_vars import MAX_RUL, PHM_FEATURES
from plots import (
    plot_prediction_interval,
    plot_an_scores,
    plot_combined_signals,
)
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

    if exp_config.an_score_plots:

        print("-"*50)
        print(f"Producing anomaly score plot for run {exp_config.run_id}")
        print("-"*50)

        an_score_plot_path = generate_path(
            basepath = plot_path,
            folders = ["an_score_plots"]
        )

        plot_an_scores(
            config = exp_config,
            outputs_path = outputs_path,
            plot_path = an_score_plot_path,
        )

    if exp_config.combined_signal_plots:

        print("-"*50)
        print(f"Producing combined signals plots for run {exp_config.run_id}")
        print("-"*50)

        signal_plot_path = generate_path(
            basepath = plot_path,
            folders = ["combined_signal_plots"]
        )

        for i in range(len(PHM_FEATURES)):

            print("-"*50)
            print(f"Producing combined signal plot for column {PHM_FEATURES[i]}")
            print("-"*50)

            plot_combined_signals(
                config = exp_config,
                outputs_path = outputs_path,
                plot_path = signal_plot_path,
                col_idx = i,
            )

else:

    print("-"*50)
    print(f"Producing quantile plot for run {exp_config.run_id}")
    print("-"*50)

    plot_path = generate_path(
        basepath = plot_path,
        folders = ["prediction_interval_plots"]
    )

    _ = plot_prediction_interval(
        config = exp_config,
        outputs_path = outputs_path,
        plot_path = plot_path,
        n_last_samples = 0
    )
