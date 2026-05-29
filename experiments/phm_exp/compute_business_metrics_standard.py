"""
Python script to compute business metrics for a non quantile regression model
"""

# general imports
import os
import sys
import ipdb
import setproctitle

src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.append(src_path)

from config_vars import BASELINE_MODEL_NAMES
from exp_config import setup_exp
from utils import generate_path, get_current_time, save_element
from perf import get_fitted_lifes, lifes_business_metrics
from models import wandb_data

experiment_path = os.path.dirname((os.path.realpath(__file__)))

exp_config, model_config, device, exp_name = setup_exp()

is_baseline_model = exp_config.model_name in BASELINE_MODEL_NAMES

if is_baseline_model:

    outputs_path = generate_path(
        basepath=experiment_path,
        folders=[
            "baseline_outputs",
            exp_config.model_name,
        ],
    )

    business_metrics_path = generate_path(
        basepath=experiment_path,
        folders=[
            "baseline_business_metrics",
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
            exp_name
        ],
    )

    business_metrics_path = generate_path(
        basepath=experiment_path,
        folders=[
            "business_metrics",
            exp_config.model_name,
            exp_config.failure_type,
            exp_config.approach,
            exp_name
        ],
    )

setproctitle.setproctitle(f"compute_business_metrics_standard_{exp_name}")

wandb_data(config=exp_config,model_config=model_config)

business_metrics_dfs = []

run_iterator = range(exp_config.n_folds) if exp_config.cv else range(exp_config.start_run_id, exp_config.n_runs)

for run_id in run_iterator:

    print("-"*50)
    print(f"Computing business metrics for run {run_id+1}")
    print("-"*50)

    fitted_lifes = get_fitted_lifes(
        config=exp_config,
        outputs_path=outputs_path,
        is_baseline=is_baseline_model,
    )

    business_metrics_df = lifes_business_metrics(
        fitted_lifes=fitted_lifes,
        config=exp_config,
    )
    business_metrics_dfs.append(business_metrics_df)

mean_metrics_df = (sum(business_metrics_dfs)/len(business_metrics_dfs)).round(2)

if exp_config.print_mean_metrics_df:

    print("-"*50)
    print("Mean business metrics table")
    print("-"*50)

    print("-"*50)
    print(mean_metrics_df.to_markdown())
    print("-"*50)

if exp_config.save_mean_metrics_df:

    filename = f"{get_current_time()}_mean_business_metrics_{exp_config.model_name}"

    save_element(
        element=mean_metrics_df,
        dirpath=business_metrics_path,
        filename=filename,
    )

    print("-"*50)
    print(f"Mean business metrics df saved at {os.path.join(business_metrics_path,filename)}")
    print("-"*50)

