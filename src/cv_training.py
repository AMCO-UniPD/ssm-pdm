"""
Python module to contain all the function needed for k fold cross
validation training
"""

import os
import sys
import time
from typing import List, Optional, Tuple, Union

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
                   get_phm_feature_names, load_cv_data, get_raw_phm_data, load_phm_data,
                   set_seed)
from wandb_funcs import init_wandb
from phm_monitor import emit, table_payload

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
        window_weight_ratio=config.window_weight_ratio,
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
    seed: Optional[int] = None,
    evaluation_quantiles: Optional[List[float]] = None,
) -> Tuple[List[pd.DataFrame],str]:
    """
    Function to implement the k fold cross validation training

    Args:
        exp_config (ExperimentConfig): experiment configuration
        model_config (ModelConfig): model configuration
        device (str): CUDA device to use
        best_model_path (str): path where to save the best model
        outputs_path (str): path where to save the outputs
        evaluation_quantiles: additional inference levels for the same SQR checkpoint

    Returns:
        metrics_list (List[pd.DataFrame]): the function performs the training and the evaluation and logs the results to
        wandb and returns the metrics_list dictionary which contains the main metrics from all the folds
        exp_name (str): name of the experiment, needed to save the metrics_dict into a pickle file
    """

    if evaluation_quantiles is not None:
        if not exp_config.quantile_reg or len(exp_config.quantiles) != 1:
            raise ValueError("Additional quantile evaluation requires one SQR training run")
        if not exp_config.save_outputs:
            raise ValueError("Additional quantile evaluation requires saved outputs")
        if any(not 0 < tau < 1 for tau in evaluation_quantiles):
            raise ValueError("Evaluation quantiles must lie strictly between zero and one")

    if seed is not None:
        set_seed(seed)

    skf = KFold(
        n_splits = exp_config.n_folds,
        shuffle = True,
        random_state = 42
    )

    #NOTE: Here I do not have to merge the training and test data:
    # the test data should be held out and used just to evalute the model
    # over different folds in best_model_perf

    train_phm_data, test_phm_data = get_raw_phm_data(config=exp_config)
    merged_phm_data = MergeData(data_list=[train_phm_data])

    #NOTE: Since the evaluation dataloaders are the same across all folds we can
    # create them just once

    print("-"*50)
    print(f"Creating evaluation dataloaders")
    print("-"*50)

    eval_loaders_dict = load_phm_data(
        config = exp_config,
        eval = True
    )

    metrics_list = []
    fold_tables = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(merged_phm_data)):

        if seed is not None:
            set_seed(seed + fold_idx)

        if fold_idx < exp_config.start_fold_id:

            print("-"*50)
            print(f"Skipping fold {fold_idx+1} because the training was already done")
            print("-"*50)
            continue

        elif fold_idx >= exp_config.stop_fold_id:

            print("-"*50)
            print(f"Skipping fold {fold_idx+1} because the training was already done")
            print("-"*50)
            continue

        else:

            print("-"*50)
            print(f"Processing fold {fold_idx+1}/{exp_config.n_folds}")
            print("-"*50)

        emit("fold_started", fold=fold_idx + 1, n_folds=exp_config.n_folds,
             stage="training", status="running")
        fold_table = pd.DataFrame()
        validation_losses = []

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

            # Each quantile run starts from a fresh model and optimizer.
            if quantile != exp_config.quantiles[0]:
                (_, _, _, model, optimizer, scheduler, criterion, eval_criterion,
                 config) = wandb_cv_data(
                    config=exp_config, model_config=model_config,
                    loaders_dict=loaders_dict, best_model_path=best_model_path)
            emit("quantile_started", tau=quantile)

            print("-"*50)
            print(f"Experiment run for quantile level {quantile} and fold {fold_idx+1}")
            print("-"*50)

            runWB = init_wandb(
                config = exp_config,
                model_config = model_config
            )

            if runWB is not None:
                emit("wandb_run", url=runWB.url,
                     project_url=runWB.url.split("/runs/", 1)[0])
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
            validation_losses.append(trainer.best_val_loss)

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

            # Evaluate the same selected checkpoint at other quantiles, without
            # entering the training loop again or changing checkpoint selection.
            for eval_tau in dict.fromkeys(evaluation_quantiles or []):
                if eval_tau == quantile:
                    continue
                eval_folders = [f"fold_{fold_idx+1}", f"quantile_{eval_tau}"]
                best_model_perf(
                    loaders_dict=eval_loaders_dict,
                    config=config,
                    model_config=model_config,
                    device=device,
                    best_model_path=quantile_best_model_path,
                    outputs_path=generate_path(basepath=outputs_path, folders=eval_folders),
                    combined_outputs_path=generate_path(
                        basepath=combined_outputs_path, folders=eval_folders
                    ),
                    tau=eval_tau,
                )

            if config.compute_metrics:

                # These predictions are for the fixed held-out PHM lives.
                config.test_idx = eval_loaders_dict["test_idx"]

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
                fold_table[f"quantile_{quantile}"] = metrics_df["Eval Loss"]

            #NOTE: End run for the current quantile

            wandb.finish()

        #NOTE: End run for the current fold

        wandb.finish()

        if not fold_table.empty:
            fold_tables.append(fold_table)
            print(f"Fold {fold_idx + 1} quantile metrics:\n{fold_table.to_markdown()}")
            emit("fold_completed", fold=fold_idx + 1,
                 validation_loss=float(np.mean(validation_losses)),
                 test_idx=[int(index) for index in eval_loaders_dict["test_idx"]],
                 table=table_payload(fold_table))

    if fold_tables:
        mean_table = sum(fold_tables) / len(fold_tables)
        print(f"Average quantile metrics:\n{mean_table.to_markdown()}")
        emit("summary", table=table_payload(mean_table), completed_folds=len(fold_tables))

    return metrics_list, exp_name
