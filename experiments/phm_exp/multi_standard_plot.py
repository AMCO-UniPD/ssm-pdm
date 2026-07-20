"""
Python script to produce the RUL plots for multiple models
"""

# general imports
import os
import sys

import ipdb

src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.append(src_path)

from exp_config import setup_exp
from models import wandb_data
from plots import multi_plot_predictions_grid
from utils import generate_path

experiment_path = os.path.dirname((os.path.realpath(__file__)))

exp_config, model_config, device, exp_name = setup_exp()

wandb_data(config=exp_config, model_config=model_config)

plot_dict = {}

for model_name, exp_name in zip(exp_config.model_names, exp_config.exp_names):

    outputs_path = generate_path(
        basepath=experiment_path,
        folders=[
            "combined_outputs",
            model_name,
            exp_config.failure_type,
            exp_config.plot_approach,
            exp_name,
            f"run_{exp_config.plot_run_id}" if not exp_config.cv else f"fold_{exp_config.plot_run_id}"
        ],
    )

    plot_dict[model_name]=outputs_path

plot_path = generate_path(
    basepath=experiment_path,
    folders=[
        "multi_plots",
        exp_config.failure_type,
        exp_config.plot_approach,
        f"run_{exp_config.plot_run_id}" if not exp_config.cv else f"fold_{exp_config.plot_run_id}"
    ],
)

print("-" * 50)
print(f"Producing multi plot for run {exp_config.run_id}")
print("-" * 50)

multi_plot_predictions_grid(
    config = exp_config,
    plot_dict = plot_dict,
    plot_path = plot_path,
)

print("-" * 50)
print(f"Producing multi plot for run {exp_config.run_id} on last {exp_config.n_last_samples} samples")
print("-" * 50)

multi_plot_predictions_grid(
    config = exp_config,
    plot_dict = plot_dict,
    plot_path = plot_path,
    n_last_samples = exp_config.n_last_samples
)
