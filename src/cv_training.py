"""
Python module to contain all the function needed for k fold cross
validation training
"""

import os
import sys
import time
from typing import List, Tuple, Union

import ipdb
import numpy as np
import pandas as pd
import setproctitle
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from exp_config import ExperimentConfig, ModelConfig
from loss import load_loss_functions
from models import (best_model_perf, compute_window_info, load_best_model,
                    load_ssm_model)
from perf import lifes_metrics
from trainer import get_trainer
from utils import (MergeData, generate_path, get_current_time, get_mono_mask,
                   get_phm_feature_names, load_cv_data, get_raw_phm_data, load_phm_data)
from wandb_funcs import init_wandb

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

    train_loader, val_loader, test_loader, test_idx = (
        loaders_dict["train_loader"],
        loaders_dict["val_loader"],
        loaders_dict["test_loader"],
        loaders_dict["test_idx"],
    )
    config.test_idx = test_idx

    if "windowed" in config.approach:
        compute_window_info(loaders_dict=loaders_dict, config=config)

    if config.get_test_idx:
        return config

    feature_names = (get_phm_feature_names(config))

    mono_mask = get_mono_mask(config=config, feature_names=feature_names)
    setattr(config, "mono_mask", mono_mask)

    model, optimizer, scheduler = load_ssm_model(
        exp_config=config,
        model_config=model_config,
        model_name=config.model_name,
        output_size=config.sequence_length,
        mono_mask=mono_mask,
    )

    if config.resume_training:

        print("-"*50)
        print("Loading best model to resume training")
        print("-"*50)

        model, _ = load_best_model(
            config=config,
            model_config=model_config,
            best_model_path=best_model_path
        )

    model = model.to(model_config.device)

    criterion, eval_criterion = load_loss_functions(
        loss_name=config.loss,
        eval_loss_name=config.eval_loss,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        model,
        optimizer,
        scheduler,
        criterion,
        eval_criterion,
        config,
    )

def train_k_fold(
    exp_config: ExperimentConfig,
    model_config: ModelConfig,
    device: str = "cpu",
    best_model_path: str = experiment_path,
    outputs_path: str = experiment_path,
    combined_outputs_path: str = experiment_path,
    metrics_path: str = experiment_path,
) -> Tuple[List[pd.DataFrame],str]:
    """
    Function to implement the k fold cross validation training

    Args:
        exp_config (ExperimentConfig): experiment configuration
        model_config (ModelConfig): model configuration
        device (str): CUDA device to use
        best_model_path (str): path where to save the best model
        outputs_path (str): path where to save the outputs

    Returns:
        metrics_list (List[pd.DataFrame]): the function performs the training and the evaluation and logs the results to
        wandb and returns the metrics_list dictionary which contains the main metrics from all the folds
        exp_name (str): name of the experiment, needed to save the metrics_dict into a pickle file
    """

    skf = KFold(
        n_splits = exp_config.n_folds,
        shuffle = True,
        random_state = 42
    )

    train_phm_data, test_phm_data = get_raw_phm_data(config=exp_config)
    merged_phm_data = MergeData(data_list=[train_phm_data, test_phm_data])

    metrics_list = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(merged_phm_data)):

        print("-"*50)
        print(f"Processing fold {fold_idx+1}/{exp_config.n_folds}")
        print("-"*50)

        train_val_data, test_data = merged_phm_data[train_val_idx], merged_phm_data[test_idx]

        print("-"*50)
        print(f"Creating dataloaders for fold {fold_idx+1}/{exp_config.n_folds}")
        print("-"*50)

        loaders_dict = load_cv_data(
            config = exp_config,
            train_val_data = train_val_data,
            test_data = test_data,
            eval = False
        )

        print("-"*50)
        print(f"Creating evaluation dataloaders for fold {fold_idx+1}/{exp_config.n_folds}")
        print("-"*50)

        eval_loaders_dict = load_phm_data(
            config = exp_config,
            eval = True
        )

        (
            train_loader,
            val_loader,
            test_loader,
            model,
            optimizer,
            scheduler,
            criterion,
            eval_criterion,
            config
        ) = wandb_cv_data(
            config = exp_config,
            model_config = model_config,
            loaders_dict = loaders_dict,
            best_model_path = best_model_path
        )

        for quantile in exp_config.quantiles:

            print("-"*50)
            print(f"Experiment run for quantile level {quantile} and fold {fold_idx+1}")
            print("-"*50)

            runWB = init_wandb(
                config = exp_config,
                model_config = model_config
            )

            if runWB is not None:
                exp_name = runWB.name
                run_name = f"{runWB.name}_fold_{fold_idx+1}_quantile_{quantile}"
                runWB.name = run_name
                setproctitle.setproctitle(run_name)
            else:
                exp_time = get_current_time()
                exp_name = f"{exp_time}_no_wandb_logging_exp"
                run_name = exp_name
                setproctitle.setproctitle(run_name)

            quantile_reg_folders = [
                f"fold_{fold_idx+1}",
                f"quantile_{quantile}",
            ]

            quantile_best_model_path = generate_path(basepath=best_model_path, folders=quantile_reg_folders)
            quantile_outputs_path = generate_path(basepath=outputs_path, folders=quantile_reg_folders)
            quantile_combined_outputs_path = generate_path(basepath=combined_outputs_path, folders=quantile_reg_folders)
            quantile_metrics_path = generate_path(basepath=metrics_path, folders=quantile_reg_folders)

            trainer = get_trainer(
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                eval_criterion=eval_criterion,
                scheduler=scheduler,
                device=device,
                best_model_path=quantile_best_model_path,
                config=config,
                tau=quantile,
            )

            trainer.run(runWB=runWB)

            if config.return_outputs or config.save_outputs or config.save_combined_outputs:

                print("#" * 50)
                print("Saving outputs")
                print("#" * 50)

                best_model_perf(
                    loaders_dict=eval_loaders_dict,
                    config=config,
                    model_config=model_config,
                    device=device,
                    best_model_path=quantile_best_model_path,
                    outputs_path=quantile_outputs_path,
                    combined_outputs_path=quantile_combined_outputs_path,
                    tau=quantile,
                )

            if config.compute_metrics:

                print("#" * 50)
                print("Computing metrics for each life in the test set")
                print("#" * 50)

                metrics_df = lifes_metrics(
                    config=config,
                    outputs_path=quantile_outputs_path,
                    metrics_path=quantile_metrics_path,
                    compute_stats=True,
                    tau=quantile
                )

                metrics_list.append(metrics_df)

            #NOTE: End run for the current quantile

            wandb.finish()

        #NOTE: End run for the current fold

        wandb.finish()


    return metrics_list, exp_name

