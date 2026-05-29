"""
Training script for PHM dataset experiments with quantile regression and
Cross Validation
"""

# general imports
import ipdb
import os
import sys
import setproctitle
import wandb

src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.append(src_path)

from exp_config import setup_exp
from utils import (
    load_cv_data,
    get_current_time,
    generate_path,
    set_seed,
)
from cv_training import train_k_fold

from perf import lifes_metrics
from models import exp_run, best_model_perf
from wandb_funcs import init_wandb

experiment_path = os.path.dirname((os.path.realpath(__file__)))

exp_config, model_config, device, exp_name = setup_exp()

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

    setproctitle.setproctitle(f"{exp_config.model_name}-test-script-cv")

metrics_list, exp_name = train_k_fold(
    exp_config = exp_config,
    model_config = model_config,
    device = device,
    best_model_path = best_model_path,
    outputs_path = outputs_path,
    combined_outputs_path = combined_outputs_path,
    metrics_path = metrics_path
)


