"""
Python script that creates a summary metrics dataframe containing the average metrics
obtained on the different quantiles
"""

# general imports
import os
import sys
import ipdb
import setproctitle
import numpy as np
import pandas as pd
from scipy import stats

src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.append(src_path)

from exp_config import setup_exp
from utils import (
    generate_path,
    get_most_recent_file,
    open_element,
    save_element
)

from perf import print_summary_metrics

experiment_path = os.path.dirname((os.path.realpath(__file__)))

exp_config, model_config, device, exp_name = setup_exp()

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

setproctitle.setproctitle(f"quantile_metrics_{exp_name}")

metrics_dfs = []

for run in range(exp_config.n_runs):

    print("#" * 50)
    print(f"Saving quantile metrics for run: {run+1}")
    print("#" * 50)

    run_metrics_path = generate_path(
        basepath = metrics_path,
        folders = [f"run_{run+1}"]
    )

    metrics_df = pd.DataFrame()

    for quantile in exp_config.quantiles:

        print("#" * 50)
        print(f"Retrieving metrics for quantile level: {quantile}")
        print("#" * 50)

        quantile_metrics_path = generate_path(
            basepath = run_metrics_path,
            folders = [f"quantile_{quantile}"]
        )

        metrics_df_path = get_most_recent_file(quantile_metrics_path)
        quantile_df = open_element(metrics_df_path,filetype="pickle")

        metrics_df[f"quantile_{quantile}"] = quantile_df["Eval Loss"]

    metrics_dfs.append(metrics_df)

mean_metrics_df=(sum(metrics_dfs)/len(metrics_dfs)).round(2)

if exp_config.print_summary_metrics:
    print_summary_metrics(metrics_df=mean_metrics_df,model_name=exp_config.model_name)

if exp_config.print_mean_metrics_df:

    print("-"*50)
    print("Mean metrics df:")
    print(mean_metrics_df.to_markdown())
    print("-"*50)

if exp_config.save_mean_metrics_df:

    filename=f"{exp_config.model_name}_{exp_config.cmapss_models}_{exp_config.approach}_quantile_reg_global_metrics_df" if exp_config.data_name == "CMAPSS" else f"{exp_config.model_name}_{exp_config.tool_type}_{exp_config.approach}_quantile_reg_global_metrics_df"

    save_element(
        mean_metrics_df,
        dirpath=metrics_path,
        filename=filename,
        filetype="pickle",
    )
