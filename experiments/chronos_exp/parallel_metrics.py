"""
Python script to compute the metrics in parallel for a list of models
Just a new version of quantile_reg_metrics.py with some additional
command line arguments
"""

# general imports
import os
import sys
import ipdb
import argparse
import numpy as np
import pandas as pd

src_path = os.path.join(os.path.dirname(__file__),"..","..","src",
)
sys.path.append(src_path)

from utils import (
    generate_path,
    load_yaml_to_dict,
    ExperimentConfig,
    get_most_recent_file,
    get_most_recent_dir,
    open_element,
    save_element,
)

from perf import(
    lifes_metrics,
    print_summary_metrics,
    sub_lifes_metrics,
    df_with_index_to_obsidian_table
)

experiment_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"chronos_exp")

config_path=os.path.join(experiment_path,"config","ssm_exp_config.yaml")
config=load_yaml_to_dict(config_path)
config=ExperimentConfig(config)

parser=argparse.ArgumentParser(description="Parallel metrics computation")
parser.add_argument(
    "--model_name",
    type=str,
    help="List of model names to compute the metrics for"
)
parser.add_argument(
    "--file_pos_dir",
    type=int,
    help="Position of the experiment directory"
)
args=parser.parse_args()

# Get the outputs directory of the most recent experiment
outputs_dict_path = generate_path(basepath=experiment_path,
                                folders=[
                                    "outputs",
                                    args.model_name,
                                    config.cmapss_models,
                                    config.approach,
                                    "quantile_reg",
                                ])
outputs_dict_path = get_most_recent_dir(outputs_dict_path,file_pos=args.file_pos_dir)

metrics_dfs = []

for i in range(config.n_runs):

    metrics_df = pd.DataFrame()
    run_quantile_path = generate_path(basepath=outputs_dict_path,
                                      folders=[f"run_{i+1}"])

    for quantile in config.quantiles:
        quantile_path = generate_path(basepath=run_quantile_path,
                                      folders=[f"quantile_{quantile}"])
        quantile_df = lifes_metrics(
            config=config,
            outputs_path=quantile_path,
            compute_stats=False
        )
        metrics_df[f"quantile_{quantile}"] = quantile_df["Eval Loss"]

    metrics_dfs.append(metrics_df)

# Create a new pd.DataFrame with the same shape of all the metrics_df which contains the mean of all the pd.DataFrames inside metrics_dfs
mean_metrics_df=(sum(metrics_dfs)/len(metrics_dfs)).round(2)

if config.print_summary_metrics:
    print_summary_metrics(metrics_df=mean_metrics_df,model_name=args.model_name)

metrics_df_path = generate_path(basepath=experiment_path,
                                folders=[
                                    "metrics",
                                    args.model_name,
                                    config.cmapss_models,
                                    config.approach,
                                    "quantile_reg",
                                    # config.exp_name
                                ])
metrics_df_path = get_most_recent_dir(metrics_df_path,file_pos=args.file_pos_dir)

if config.save_mean_metrics_df:

    filename=f"{args.model_name}_{config.cmapss_models}_{config.approach}_quantile_reg_global_metrics_df"
    save_element(
        mean_metrics_df,
        dirpath=metrics_df_path,
        filename=filename,
        filetype="pickle",
    )


if config.obsidian_table:
    if config.sub_lifes_metrics:
        sub_metrics_df = sub_lifes_metrics(
            config=config,
            metrics_df=mean_metrics_df,
            compute_stats=True
        )
        obsidian_table = df_with_index_to_obsidian_table(sub_metrics_df)
    else:
        obsidian_table = df_with_index_to_obsidian_table(mean_metrics_df)
    print('#'* 50)
    print(obsidian_table)
    print('#'* 50)
