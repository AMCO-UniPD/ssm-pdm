"""
Python script containing some performance evaluation functions to use
inside `best_model_perf`
"""

import os
import sys
import ipdb
import torch
import pandas as pd
import numpy as np
from ceruleo.dataset.catalog.CMAPSS import CMAPSSDataset

chronos_path_src = os.path.join(os.path.dirname(__file__),"chronos-rul","src")
sys.path.append(chronos_path_src)

from utils import (
    ExperimentConfig,
    get_current_time,
    generate_path,
    get_feature_names,
    save_element,
    open_element,
    get_most_recent_file,
)

from loss import *

cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
experiment_path = os.path.join(cwd, "experiments", "chronos_exp")

def lifes_metrics(
        config: ExperimentConfig,
        outputs_path: str = experiment_path,
        metrics_path: str = experiment_path,
        compute_stats: bool = False,
) -> pd.DataFrame:
    """
    Calculate the metrics for each life and each sensor in the dataset and save them in a pd.DataFrame

    Args:
        config:dict ExperimentConfig object
        outputs_path:str Path to the outputs
        metrics_path:str Path to save the metrics
        compute_stats:bool Whether to compute the mean,median and std of the metrics

    Returns:
        pd.DataFrame Metrics DataFrame
    """

    feature_names = get_feature_names(config)
    metrics_df=pd.DataFrame()

    outputs_path = get_most_recent_file(outputs_path,file_pos=config.file_pos)
    outputs_dict = open_element(
        file_path=outputs_path,
        filetype="pickle"
    )
    print(f"Opened outputs_dict at path: {outputs_path}")
    y_pred,y_true=outputs_dict["y_pred"],outputs_dict["y_true"]

    _,eval_criterion=load_loss_functions(
        loss_name=config.loss,
        model_name=config.model_name,
        eval_loss_name=config.eval_loss,
        tau=config.tau
    )

    pd.options.display.float_format = "{:.2f}".format

    if config.model_name.startswith("chronos"):
        for i in range(y_pred.shape[0]):
            mask = torch.tensor(y_true[i,0,:]!=0).unsqueeze(0)
            for j,sensor in zip(range(y_pred.shape[1]),feature_names):
                pred=torch.tensor(y_pred[i,j,:]).unsqueeze(0)
                true=torch.tensor(y_true[i,j,:]).unsqueeze(0)
                eval_loss=eval_criterion(y_pred=pred,y_true=true,mask=mask).item()
                metrics_df.at[f"Life_{i+config.test_idx[0]+1}",sensor]=round(eval_loss,2)
    else:
        for i in range(len(y_pred)):
            mask = torch.tensor(y_true[i]!=0)
            pred=torch.tensor(y_pred[i])
            true=torch.tensor(y_true[i])
            eval_loss=eval_criterion(y_pred=pred,y_true=true,mask=mask).item()
            metrics_df.at[f"Life_{i+config.test_idx[0]+1}","Eval Loss"]=round(eval_loss,2)

    if compute_stats:
        # Add a row Life_mean with the mean of the metrics over all the columns
        metrics_df.loc["Life_mean"]=metrics_df.mean(axis=0).round(2)
        # Add a row Life_median with the median of the metrics over all the columns
        metrics_df.loc["Life_median"]=metrics_df.median(axis=0).round(2)
        # Add a row Life_std with the std of the metrics over all the columns
        metrics_df.loc["Life_std"]=metrics_df.std(axis=0).round(2)
        # Add a column Sensor_mean with the mean of the metrics over all the rows
        if config.model_name.startswith("chronos"):
            metrics_df["Sensor_mean"]=metrics_df.mean(axis=1).round(2)
            metrics_df["Sensor_median"]=metrics_df.median(axis=1).round(2)
            metrics_df["Sensor_std"]=metrics_df.std(axis=1).round(2)

    print('#'* 50)
    print(f"Mean eval loss over all the test lifes: {metrics_df.loc['Life_mean']}")
    print('#'* 50)
    print(f"Median eval loss over all the test lifes: {metrics_df.loc['Life_median']}")
    print('#'* 50)
    print(f"Std eval loss over all the test lifes: {metrics_df.loc['Life_std']}")
    print('#'* 50)

    if config.save_metrics_df:
        save_element(
            element=metrics_df,
            dirpath=metrics_path,
            filename=f"{get_current_time()}_lifes_metrics_{config.model_name}_{config.cmapss_models}.pickle"
        )

    pd.options.display.float_format = None

    return metrics_df

# Function to select a subset of the rows and a subset of the columns 
# of a metrics_df

def sub_lifes_metrics(
    config: ExperimentConfig,
    metrics_df: pd.DataFrame,
    compute_stats: bool = False,
):
    """
    Select a subset of the rows and a subset of the columns of a metrics_df

    Args:
        config:ExperimentConfig ExperimentConfig object
        metrics_df:pd.DataFrame Metrics DataFrame
        compute_stats:bool Whether to compute the mean,median and std of the metrics

    Returns:
        pd.DataFrame Subset of the metrics_df
    """

    if config.metrics_idx is None:
        config.metrics_idx=np.arange(config.nrows*config.ncols)
    metrics_idx = [f"Life_{i+config.test_idx[0]+1}" for i in config.metrics_idx]
    if config.model_name.startswith("chronos"):
        sub_metrics_df = metrics_df.loc[metrics_idx,config.metrics_cols]
    else:
        sub_metrics_df = metrics_df.loc[metrics_idx]

    if compute_stats:
        sub_metrics_df.loc["Life_mean"] = sub_metrics_df.mean(axis=0).round(2)
        sub_metrics_df.loc["Life_median"] = sub_metrics_df.median(axis=0).round(2)
        sub_metrics_df.loc["Life_std"] = sub_metrics_df.std(axis=0).round(2)
        if len(config.metrics_cols)>1 and config.model_name.startswith("chronos"):
            sub_metrics_df["Sensor_mean"] = sub_metrics_df.mean(axis=1).round(2)
            sub_metrics_df["Sensor_median"] = sub_metrics_df.median(axis=1).round(2)
            sub_metrics_df["Sensor_std"] = sub_metrics_df.std(axis=1).round(2)

    return sub_metrics_df


# Function to render the data contained in a pd.DataFrame into a markdown table

def df_to_obsidian_table(df):
    markdown = "| " + " | ".join(df.columns) + " |\n"
    markdown += "| " + " | ".join("---" for _ in df.columns) + " |\n"
    for row in df.itertuples(index=False):
        markdown += "| " + " | ".join(map(str, row)) + " |\n"
    return markdown


def df_with_index_to_obsidian_table(df):
    lines = []
    # Include index column name as the first header
    lines.append("| " + " | ".join([df.index.name or ""] + list(df.columns)) + " |")
    lines.append("| " + " | ".join(["---"] * (len(df.columns) + 1)) + " |")
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + list(map(str, row))) + " |")
    return "\n".join(lines)


