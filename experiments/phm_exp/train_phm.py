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

from exp_config import define_arguments, set_exp_name, setup_exp
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

metrics_path = generate_path(
    basepath=experiment_path,
    folders=[
        "metrics",
        exp_config.model_name,
        f"{exp_config.tool_type}_tools",
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
        f"{exp_config.tool_type}_tools",
        exp_config.failure_type,
        exp_config.approach,
        exp_name
    ],
)

#NOTE: Probably this test_script thing is useless, use the scripts I created

if exp_config.test_script:
    print("#" * 50)
    print("Running best model performance test")
    print("#" * 50)

    setproctitle.setproctitle(f"{exp_config.model_name}-test-script")

    best_model_perf(
        config = exp_config,
        model_config = model_config,
        device = device,
        best_model_path = best_model_path,
        outputs_path = outputs_path,
    )

for run in range(exp_config.start_run_id, exp_config.start_run_id + exp_config.n_runs):

    print("-"*50)
    print(f"Experiment run for run {run+1}")
    print(f"Setting the seed for this run to {run+1}")
    print("-"*50)

    set_seed(run+1)

    for quantile in exp_config.quantiles:

        print("-"*50)
        print(f"Experiment run for quantile level {quantile} and run {run+1}")
        print("-"*50)

        setproctitle.setproctitle(exp_name)

        quantile_reg_folders = [
            f"run_{run+1}",
            f"quantile_{quantile}",
        ]

        quantile_best_model_path = generate_path(basepath=best_model_path, folders=quantile_reg_folders)
        quantile_outputs_path = generate_path(basepath=outputs_path, folders=quantile_reg_folders)
        quantile_metrics_path = generate_path(basepath=metrics_path, folders=quantile_reg_folders)

        wandb_run(
            run_name = exp_name,
            config = exp_config,
            model_config = model_config,
            device = model_config.device,
            best_model_path = quantile_best_model_path,
            outputs_path = quantile_outputs_path,
            metrics_path = quantile_metrics_path,
            tau = quantile
        )

