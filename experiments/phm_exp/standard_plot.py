"""
Python script to produce the standard plots
"""

# general imports
import os
import sys

import ipdb

src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.append(src_path)

from config_vars import BASELINE_MODEL_NAMES
from exp_config import setup_exp
from models import wandb_data
from plots import plot_predictions_grid
from utils import generate_path

experiment_path = os.path.dirname((os.path.realpath(__file__)))

exp_config, model_config, device, exp_name = setup_exp()

is_baseline_model = exp_config.model_name in BASELINE_MODEL_NAMES

wandb_data(config=exp_config, model_config=model_config)

if is_baseline_model:

    outputs_path = generate_path(
        basepath=experiment_path,
        folders=[
            "baseline_outputs",
            exp_config.model_name,
        ],
    )
    
    plot_path = generate_path(
        basepath=experiment_path,
        folders=[
            "baseline_plots",
            exp_config.model_name,
        ],
    )

else:

    outputs_path = generate_path(
        basepath=experiment_path,
        folders=[
            "combined_outputs",
            exp_config.model_name,
            exp_config.failure_type,
            exp_config.approach,
            exp_name,
            f"run_{exp_config.plot_run_id}",
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
            f"run_{exp_config.plot_run_id}",
        ],
    )

print("-" * 50)
print(f"Producing standard plot for run {exp_config.run_id}")
print("-" * 50)

plot_path = generate_path(basepath=plot_path, folders=["standard_plots"])

_ = plot_predictions_grid(
    config=exp_config,
    outputs_path=outputs_path,
    plot_path=plot_path,
    start_idx=exp_config.start_idx,
    end_idx=exp_config.end_idx,
    n_last_samples=exp_config.n_last_samples
)
