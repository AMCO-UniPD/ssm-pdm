"""
Python script to produce the plots with the prediction intervals
"""

# general imports
import os
import sys
import ipdb
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import f

src_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "src",
)
sys.path.append(src_path)

from plots import plot_prediction_interval
from utils import (
    generate_path,
    load_yaml_to_dict,
    ExperimentConfig,
    get_most_recent_file,
    get_current_time,
    open_element,
)

experiment_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "chronos_exp"
)

config_path = os.path.join(experiment_path, "config", "ssm_exp_config.yaml")
config = load_yaml_to_dict(config_path)
config = ExperimentConfig(config)

# Set the plot path
plot_path = generate_path(
    basepath=experiment_path,
    folders=[
        "plots",
        config.model_name,
        config.cmapss_models,
        config.approach,
        "quantile_reg",
        config.exp_name,
        "interval",
    ],
)

# Get the outputs directory of the most recent experiment
outputs_dirpath = generate_path(
    basepath=experiment_path,
    folders=[
        "outputs",
        config.model_name,
        config.cmapss_models,
        config.approach,
        "quantile_reg",
        config.exp_name,
    ],
)

for run in range(config.n_runs):
    run_outputs_path = generate_path(
        basepath=outputs_dirpath,
        folders=[
            f"run_{run+1}",
            f"quantile_{config.quantile_run}",
            f"outputs_quantile_{config.quantile_run}",
        ],
    )

    plot_prediction_interval(
        config=config,
        outputs_path=run_outputs_path,
        plot_path=plot_path,
        run=run,
    )
