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
from plots import plot_prediction_interval

experiment_path = os.path.dirname((os.path.realpath(__file__)))

exp_config, model_config, device, exp_name = setup_exp()

outputs_path = generate_path(
    basepath=experiment_path,
    folders=[
        "outputs",
        exp_config.model_name,
        f"{exp_config.tool_type}_tools",
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
        f"{exp_config.tool_type}_tools",
        exp_config.failure_type,
        exp_config.approach,
        exp_name,
        f"run_{exp_config.plot_run_id}"
    ],
)

print("-"*50)
print(f"Producing quantile plot for run {exp_config.run_id}")
print("-"*50)

_ = plot_prediction_interval(
    config = exp_config,
    outputs_path = outputs_path,
    plot_path = plot_path,
)
