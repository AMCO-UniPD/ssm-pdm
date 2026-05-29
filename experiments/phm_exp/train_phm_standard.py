"""
Training script for PHM dataset for standard experiments (no quantile regression)
"""

# general imports
import os
import sys
import setproctitle
import ipdb
import wandb

src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.append(src_path)

from exp_config import setup_exp
from utils import (
    get_current_time,
    generate_path,
    set_seed,
)

from perf import lifes_metrics
from models import best_model_perf, exp_run
from wandb_funcs import init_wandb

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

combined_outputs_path = generate_path(
    basepath=experiment_path,
    folders=[
        "combined_outputs",
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

metrics_path = generate_path(
    basepath=experiment_path,
    folders=[
        "business_metrics",
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

    setproctitle.setproctitle(f"{exp_config.model_name}-standard-test-script")

    run_iterator = range(exp_config.n_folds) if exp_config.cv else range(exp_config.start_run_id, exp_config.n_runs)

    for run in run_iterator:

        best_model_path_test_script = generate_path(
            basepath = best_model_path,
            folders = [f"run_{run+1}"]
        )

        outputs_path_test_script = generate_path(
            basepath = outputs_path,
            folders = [f"run_{run+1}"]
        )

        combined_outputs_path_test_script = generate_path(
            basepath = combined_outputs_path,
            folders = [f"run_{run+1}"]
        )

        if exp_config.save_outputs or exp_config.save_combined_outputs:

            print("-"*50)
            print(f"Saving outputs for run {run+1}")
            print("-"*50)

            best_model_perf(
                config = exp_config,
                model_config = model_config,
                device = device,
                best_model_path = best_model_path_test_script,
                outputs_path = outputs_path_test_script,
                combined_outputs_path = combined_outputs_path_test_script,
            )

        if exp_config.compute_metrics:

            print("-"*50)
            print(f"Computing metrics for run {run+1}")
            print("-"*50)

            metrics_path_test_script = generate_path(
                basepath = metrics_path,
                folders = [f"run_{run+1}"]
            )

            metrics_df = lifes_metrics(
                config = exp_config,
                outputs_path = outputs_path,
                metrics_path = metrics_path,
            )

    quit()

run_iterator = range(exp_config.n_folds) if exp_config.cv else range(exp_config.start_run_id, exp_config.n_runs)

for run in run_iterator:

    print("-"*50)
    print(f"Experiment run for run {run+1}")
    print(f"Setting the seed for this run to {run+1}")
    print("-"*50)

    set_seed(run+1)

    print("-"*50)
    print(f"Experiment for run {run+1}")
    print("-"*50)

    runWB = init_wandb(
        config = exp_config,
        model_config = model_config
    )

    if runWB is not None:
        exp_name = runWB.name
        run_name = f"{runWB.name}_run_{run+1}"
        runWB.name = run_name
        setproctitle.setproctitle(run_name)
    else:
        exp_time = get_current_time()
        exp_name = f"{exp_time}_no_wandb_logging_exp"
        run_name = exp_name
        setproctitle.setproctitle(run_name)

    best_model_path_run = generate_path(basepath=best_model_path, folders=[f"run_{run+1}"])
    outputs_path_run = generate_path(basepath=outputs_path, folders=[f"run_{run+1}"])
    combined_outputs_path_run = generate_path(basepath=combined_outputs_path, folders=[f"run_{run+1}"])
    metrics_path_run = generate_path(basepath=metrics_path, folders=[f"run_{run+1}"])

    exp_run(
        config = exp_config,
        model_config = model_config,
        runWB=runWB,
        device = model_config.device,
        best_model_path = best_model_path_run,
        outputs_path = outputs_path_run,
        combined_outputs_path = combined_outputs_path_run,
        metrics_path = metrics_path_run,
    )

    wandb.finish()
