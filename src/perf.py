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

    for i in range(y_pred.shape[0]):
        mask = y_pred[i,0,:]!=0
        for j,sensor in zip(range(y_pred.shape[1]),feature_names):
            metrics_df.at[f"Life_{i}",sensor]=eval_criterion(y_pred=y_pred[i,j,:],y_true=y_true[i,j,:],mask=mask)

    save_element(
        element=metrics_df,
        dirpath=metrics_path,
        filename=f"{get_current_time()}_lifes_metrics_{config.model_name}_{config.cmapss_models}.pkl"
    )

    return metrics_df


