"""
Python module to contain all the function needed for k fold cross
validation training
"""

import ipdb
import os
import sys
import time
import wandb
from argparse import Namespace
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import StratifiedKFold, train_test_split
import setproctitle

from exp_config import ExperimentConfig, ModelConfig

cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
experiment_path = os.path.join(cwd,"experiments")

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

    #NOTE: Define stratified folds

    skf = StratifiedKFold(
        n_splits = exp_config.n_folds,
        shuffle = True,
        random_state = exp_config.seed
    )

    #NOTE: The PHMDataset2018 should be compatible with all sklearn.model_selection functions
    # → KFold is one of them so if I use it on a PHMDataset2018 instance I should be able to obtain
    # a cv fold splitting based on the cycles!

    #WARN: The problem is that I need to merge together the `train_phm_data` and `test_phm_data`
    # objects into a single one. I cannot simply pass `train_phm_tools+test_phm_tools` as the tools
    # argument because there are some common names and I distinguish them with the train=True
    # and train=False arguments

    #TODO: Idea → create a function that takes in input the two datasets to merge
    # and creates a new object (like TransData) that inserts the two lists of dataframes
    # into a single one

    metrics_dict = {
        "test_loss": [],
        "val_loss": [],
        "eval_test_loss": [],
        "eval_val_loss": []
    }

