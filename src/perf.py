"""
Python script containing some performance evaluation functions to use
inside `best_model_perf`
"""

import os
import sys
import ipdb
import torch
import pandas as pd
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
):
    """
    Calculate the metrics for each life and each sensor in the dataset and save them in a pd.DataFrame

    Args:
        config:dict ExperimentConfig object
        outputs_path:str Path to the outputs
        metrics_path:str Path to save the metrics
    """

    feature_names = get_feature_names(config)
    metrics_df=pd.DataFrame()

    outputs_path = get_most_recent_file(outputs_path,file_pos=config.file_pos)
    outputs_dict = open_element(
        file_path=outputs_path,
        filetype="pickle"
    )
    y_pred,y_true=outputs_dict["y_pred"],outputs_dict["y_true"]

    _,eval_criterion=load_loss_functions(
        loss_name=config.loss,
        eval_loss_name=config.eval_loss,
        tau=config.tau
    )

    pd.options.display.float_format = "{:.2f}".format

    for i in range(y_pred.shape[0]):
        mask = torch.tensor(y_true[i,0,:]!=0).unsqueeze(0)
        for j,sensor in zip(range(y_pred.shape[1]),feature_names):
            pred=torch.tensor(y_pred[i,j,:]).unsqueeze(0)
            true=torch.tensor(y_true[i,j,:]).unsqueeze(0)
            eval_loss=eval_criterion(y_pred=pred,y_true=true,mask=mask).item()
            metrics_df.at[f"Life_{i}",sensor]=round(eval_loss,2)
    
    # Add a row Life_mean with the mean of the metrics over all the columns
    metrics_df.loc["Life_mean"]=metrics_df.mean(axis=0).round(2)
    # Add a column Sensor_mean with the mean of the metrics over all the rows
    metrics_df["Sensor_mean"]=metrics_df.mean(axis=1).round(2)

    save_element(
        element=metrics_df,
        dirpath=metrics_path,
        filename=f"{get_current_time()}_lifes_metrics_{config.model_name}_{config.cmapss_models}.pkl"
    )

    pd.options.display.float_format = None

    return metrics_df

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


