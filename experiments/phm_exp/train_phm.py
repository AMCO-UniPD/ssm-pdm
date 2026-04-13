"""
Training script for PHM dataset experiments
"""

# general imports
import os
import sys
import setproctitle

src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.append(src_path)

from exp_config import setup_exp
from utils import (
    generate_path,
    set_seed,
)

from perf import lifes_metrics
from models import wandb_run, best_model_perf

experiment_path = os.path.dirname((os.path.realpath(__file__)))

exp_config, model_config, device, exp_name = setup_exp()

print("-"*50)
print("PHM Dataset Experiment")
print(f"Model: {exp_config.model_name}")
print(f"Failure type: {exp_config.failure_type}")
print(f"Model size: {model_config.d_model}")
print(f"Learning rate: {exp_config.lr}")
print(f"Dropout: {model_config.dropout}")
print(f"Activation: {model_config.act}")
print(f"Sequence length: {exp_config.sequence_length}")
print(f"Batch size: {exp_config.batch_size}")
print(f"Epochs: {exp_config.epochs}")
print(f"Approach: {exp_config.approach}")
print(f"Number of runs: {exp_config.n_runs}")
print(f"Evaluation quantiles: {exp_config.quantiles}")
print("-"*50)

best_model_path = generate_path(
    basepath=experiment_path,
    folders=[
        "best_models",
        exp_config.model_name,
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

plot_path = generate_path(
    basepath=experiment_path,
    folders=[
        "plots",
        exp_config.model_name,
        exp_config.failure_type,
        exp_config.approach,
        exp_name
    ],
)

if exp_config.test_script:
    print("-" * 50)
    print("Running best model performance test")
    print("-" * 50)

    setproctitle.setproctitle(f"{exp_config.model_name}-test-script")

    for run in range(exp_config.start_run_id, exp_config.n_runs):
        for quantile in exp_config.quantiles:

            quantile_reg_folders = [
                f"run_{run+1}",
                f"quantile_{quantile}"
            ]

            best_model_path_test_script = generate_path(
                basepath = best_model_path,
                folders = quantile_reg_folders
            )

            outputs_path_test_script = generate_path(
                basepath = outputs_path,
                folders = quantile_reg_folders
            )

            if exp_config.save_outputs:

                print("-"*50)
                print(f"Saving outputs for run {run+1} and quantile {quantile}")
                print("-"*50)

                best_model_perf(
                    config = exp_config,
                    model_config = model_config,
                    device = device,
                    best_model_path = best_model_path_test_script,
                    outputs_path = outputs_path_test_script,
                    tau = quantile
                )

            if exp_config.compute_metrics:

                print("-"*50)
                print(f"Computing metrics for run {run+1} and quantile {quantile}")
                print("-"*50)

                metrics_path_test_script = generate_path(
                    basepath = metrics_path,
                    folders = quantile_reg_folders
                )

                metrics_df = lifes_metrics(
                    config = exp_config,
                    outputs_path = outputs_path,
                    metrics_path = metrics_path,
                    tau = quantile
                )

    quit()

for run in range(exp_config.start_run_id, exp_config.n_runs):

    print("-"*50)
    print(f"Experiment run for run {run+1}")
    print(f"Setting the seed for this run to {run+1}")
    print("-"*50)

    set_seed(run+1)

    for quantile in exp_config.quantiles:

        print("-"*50)
        print(f"Experiment run for quantile level {quantile} and run {run+1}")
        print("-"*50)

        run_name = f"{exp_name}_run_{run+1}_quantile_{quantile}"
        setproctitle.setproctitle(run_name)

        quantile_reg_folders = [
            f"run_{run+1}",
            f"quantile_{quantile}",
        ]

        quantile_best_model_path = generate_path(basepath=best_model_path, folders=quantile_reg_folders)
        quantile_outputs_path = generate_path(basepath=outputs_path, folders=quantile_reg_folders)
        quantile_metrics_path = generate_path(basepath=metrics_path, folders=quantile_reg_folders)

        wandb_run(
            run_name = run_name,
            config = exp_config,
            model_config = model_config,
            device = model_config.device,
            best_model_path = quantile_best_model_path,
            outputs_path = quantile_outputs_path,
            metrics_path = quantile_metrics_path,
            tau = quantile
        )

