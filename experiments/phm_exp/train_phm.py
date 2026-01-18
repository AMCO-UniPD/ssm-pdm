"""
Training script for PHM dataset experiments
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

from exp_config import define_arguments, set_exp_name
from utils import (
    ExperimentConfig,
    generate_path,
    get_most_recent_file,
    load_yaml_to_dict,
    open_element,
    get_current_time,
    save_element,
    set_seed,
    load_phm_data
)

from models import wandb_run, best_model_perf

from ssm_models import ModelConfig

experiment_path = os.path.dirname((os.path.realpath(__file__)))

args = define_arguments()

exp_config = load_yaml_to_dict(args.exp_config_path)
exp_config = ExperimentConfig.from_dict(exp_config)
exp_config.add_params(args=args.__dict__)

model_config = load_yaml_to_dict(exp_config.model_config_path)
model_config = ModelConfig.from_dict(model_config)

device = torch.device(f"cuda:{exp_config.device_num}" if torch.cuda.is_available() else "cpu")
model_config.device = device

print("-" * 50)
print(f"Using device: {device}")
print("-" * 50)

exp_name = set_exp_name(exp_config)

print("-" * 50)
print(f"Experiment name set to {exp_name}")
print("-" * 50)

ipdb.set_trace()

best_model_path = generate_path(
    basepath=experiment_path,
    folders=[
        "best_models",
        exp_config.model_name,
        f"{config.tool_type}_tools",
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
        f"{config.tool_type}_tools",
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
        f"{config.tool_type}_tools",
        exp_config.failure_type,
        exp_config.approach,
        exp_name
    ],
)


plot_path = generate_path(
    basepath=experiment_path,
    folders=[
        "plots",
        exp_config.model_name,
        f"{config.tool_type}_tools",
        exp_config.failure_type,
        exp_config.approach,
        exp_name
    ],
)

if exp_config.test_script:
    print("#" * 50)
    print("Running best model performance test")
    print("#" * 50)

    setproctitle.setproctitle(f"{exp_config.model_name}-test-script")

for run in range(exp_config.start_run_id, exp_config.start_run_id + exp_config.n_runs):

    print("-"*50)
    print(f"Experiment run for run {run+1}")
    print(f"Setting the seed for this run to {run+1}")
    print("-"*50)

    set_seed(run)

    for quantile in exp_config.quantiles:

        print("-"*50)
        print(f"Experiment run for quantile level {quantile} and run {run+1}")
        print("-"*50)

        setproctitle.setproctitle(run_name)

        quantile_reg_folders = [
            f"run_{run+1}",
            f"quantile_{quantile}",
        ]

        quantile_best_model_path = generate_path(basepath=best_model_path, folders=quantile_reg_folders)
        quantile_outputs_path = generate_path(basepath=outputs_path, folders=quantile_reg_folders)

        model, model_info = wandb_run(
            run_name = exp_name,
            config = exp_config,
            model_config = model_config,
            device = model_config.device,
            best_model_path = quantile_best_model_path,
            outputs_path = quantile_outputs_path,
            tau = quantile
        )

