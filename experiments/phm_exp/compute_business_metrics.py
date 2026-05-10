"""
Python script to compute business metrics for a quantile regression model
"""

# general imports
import os
import sys
import ipdb
import setproctitle

src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.append(src_path)

from exp_config import setup_exp
from utils import generate_path, get_current_time, save_element
from perf import get_fitted_lifes, lifes_business_metrics
from models import wandb_data

experiment_path = os.path.dirname((os.path.realpath(__file__)))

exp_config, model_config, device, exp_name = setup_exp()

is_baseline_model = "quantile" in exp_config.model_name

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

setproctitle.setproctitle(f"compute_business_metrics_{exp_name}")

wandb_data(config=exp_config,model_config=model_config)

for quantile in exp_config.quantiles:

    business_metrics_path_quantile = generate_path(
        basepath=business_metrics_path,
        folders=[f"quantile_{quantile}"]
    )
    business_metrics_dfs = []

    for run_id in range(exp_config.start_run_id, exp_config.n_runs):

        print("-"*50)
        print(f"Computing business metrics for quantile {quantile} and run {run_id+1}")
        print("-"*50)

        quantile_outputs_path = generate_path(
            basepath=outputs_path,
            folders=[
                f"run_{run_id+1}",
                f"quantile_{quantile}"
            ]
        )

        fitted_lifes = get_fitted_lifes(
            config=exp_config,
            outputs_path=outputs_path,
            is_baseline=is_baseline_model,
            tau=quantile
        )

        business_metrics_df = lifes_business_metrics(
            fitted_lifes=fitted_lifes,
            config=exp_config,
        )
        business_metrics_dfs.append(business_metrics_df)

    print("-"*50)
    print(f"Computing mean business metrics table for quantile {quantile}")
    print("-"*50)

    mean_metrics_df = (sum(business_metrics_dfs)/len(business_metrics_dfs)).round(2)

    if exp_config.print_mean_metrics_df:

        print("-"*50)
        print(f"Mean business metrics table for quantile {quantile}")
        print("-"*50)

        print("-"*50)
        print(mean_metrics_df.to_markdown())
        print("-"*50)

    if exp_config.save_mean_metrics_df:

        filename = f"{get_current_time()}_mean_business_metrics_{exp_config.model_name}_quantile_{quantile}"

        save_element(
            element=mean_metrics_df,
            dirpath=business_metrics_path_quantile,
            filename=filename,
        )

        print("-"*50)
        print(f"Mean business metrics df for quantile {quantile} saved at {os.path.join(business_metrics_path,filename)}")
        print("-"*50)
