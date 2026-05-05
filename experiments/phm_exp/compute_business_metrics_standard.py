"""
Python script to compute business metrics for a non quantile regression model
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
from utils import (
    generate_path,
    save_element,
    get_current_time,
)
from perf import lifes_business_metrics, print_summary_metrics
from models import wandb_data

experiment_path = os.path.dirname((os.path.realpath(__file__)))

exp_config, model_config, device, exp_name = setup_exp()

is_baseline_model = exp_config.model_name in ["mean", "median"]

if is_baseline_model:

    outputs_path = generate_path(
        basepath=experiment_path,
        folders=[
            "baseline_outputs",
            exp_config.model_name,
        ],
    )

    metrics_path = generate_path(
        basepath=experiment_path,
        folders=[
            "baseline_metrics",
            exp_config.model_name,
        ],
    )

else:

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

setproctitle.setproctitle(f"compute_business_metrics_{exp_name}")

wandb_data(config=exp_config,model_config=model_config)

if is_baseline_model:

    business_metrics_df = lifes_business_metrics(
        config=exp_config,
        outputs_path=outputs_path,
        business_metrics_path=business_metrics_path,
        compute_stats = True,
    )

else:

    business_metrics_dfs = []

    for run in range(exp_config.start_run_id, exp_config.n_runs):

        run_business_metrics_path = generate_path(
            basepath = business_metrics_path,
            folders = [f"run_{run+1}"]
        )

        run_outputs_path = generate_path(
            basepath = outputs_path,
            folders = [f"run_{run+1}"]
        )

        business_metrics_df = lifes_business_metrics(
            config=exp_config,
            outputs_path=run_outputs_path,
            business_metrics_path=run_business_metrics_path,
            compute_stats = True,
        )

        business_metrics_dfs.append(business_metrics_df)

    mean_business_metrics_df=(sum(business_metrics_dfs)/len(business_metrics_dfs)).round(2)
    mean_business_metrics_df.index.name = "Lifes"

    if exp_config.print_summary_metrics:
        print_summary_metrics(metrics_df=mean_business_metrics_df,model_name=exp_config.model_name)

    if exp_config.print_mean_business_metrics_df:

        print("-"*50)
        print("Mean metrics df:")
        print(mean_metrics_df.to_markdown())
        print("-"*50)

    if exp_config.save_mean_metrics_df:

        filename= f"{exp_config.model_name}_{exp_config.approach}_global_business__metrics_df"

        save_element(
            mean_business_metrics_df,
            dirpath=business_metrics_path,
            filename=filename,
            filetype="pickle",
        )
