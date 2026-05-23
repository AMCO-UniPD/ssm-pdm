"""
Python module to contain all the function needed for k fold cross
validation training
"""

import ipdb
import os
import sys
import time
import wandb
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import KFold
import setproctitle

from exp_config import ExperimentConfig, ModelConfig
from utils import load_cv_data

cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
experiment_path = os.path.join(cwd,"experiments")

def wandb_cv_data(
    config: ExperimentConfig,
    model_config: ModelConfig,
    loaders_dict: dict,
    best_model_path: str = os.getcwd(),
) -> Union[
    Tuple[
        DataLoader,
        DataLoader,
        DataLoader,
        nn.Module,
        optim.Optimizer,
        optim.lr_scheduler._LRScheduler,
        nn.Module,
        nn.Module,
        ExperimentConfig,
    ],
    ExperimentConfig,
]:
    """
    Equivalent of wandb_data but for the cv case. We skip the dataloaders
    creation part because that is done in load_cv_data

    Args:
        exp_config (ExperimentConfig): ExperimentConfig object
        model_config (ModelConfig): ModelConfig object
        loaders_dict (dict): dictionary containing the dataloaders
        best_model_path (str): path containing the best model. Needed in
        case we want to use the resume training option
    """

    pass

def train_k_fold(
    exp_config: ExperimentConfig,
    model_config: ModelConfig
) -> Tuple[dict,str]:
    """
    Function to implement the k fold cross validation training

    Args:
        exp_config (ExperimentConfig): experiment configuration
        model_config (ModelConfig): model configuration

    Returns:
        metrics_dict (dict): the function performs the training and the evaluation and logs the results to
        wandb and returns the metrics_dict dictionary which contains the main metrics from all the folds
        exp_name (str): name of the experiment, needed to save the metrics_dict into a pickle file
    """

    exp_name = f"k_fold_training_{exp_config.exp_name}"

    skf = KFold(
        n_splits = exp_config.n_folds,
        shuffle = True,
        random_state = 42
    )

    train_phm_data, test_phm_data = get_raw_phm_data(config=exp_config)
    merged_phm_data = MergeData(data_list=[train_phm_data, test_phm_data])

    metrics_dict = {
        "test_loss": [],
        "val_loss": [],
        "eval_test_loss": [],
        "eval_val_loss": []
    }

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(merged_phm_data)):

        print("-"*50)
        print(f"Processing fold {fold_idx+1}/{exp_config.n_folds}")
        print(f"train_val_idx: {train_val_idx}")
        print(f"test_idx: {test_idx}")
        print("-"*50)

        train_val_data, test_data = merged_phm_data[train_val_idx], merged_phm_data[test_idx]

        loaders_dict = load_cv_data(
            config = exp_config,
            train_val_data = train_val_data,
            test_data = test_data,
            eval = False
        )


    return metrics_dict, exp_name

