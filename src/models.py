"""
Python script containing utility functions for the models of the `chronos-pdm` project
"""

# general imports
import os
import sys
import ipdb
import traceback
import gc
import time
import wandb
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import Tuple, Union

# torch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

# from apex.optimizers import FusedAdam
from torch.optim import AdamW, lr_scheduler

chronos_path_src = os.path.join(os.path.dirname(__file__), "chronos-rul", "src")
chronos_path_scripts = os.path.join(os.path.dirname(__file__), "chronos-rul", "scripts")
imports_path = os.path.join(os.path.dirname(__file__), "AD_MG", "src")
sys.path.append(chronos_path_src)
sys.path.append(chronos_path_scripts)
sys.path.append(imports_path)

# import from other modules
from utils import (
    get_current_time,
    get_phm_feature_names,
    sample_quantile,
    save_element,
    ExperimentConfig,
    generate_path,
    load_reg_data,
    load_phm_data,
    get_most_recent_file,
    open_element,
    get_feature_names,
    combine_values,
)

from config_vars import MAX_RUL
from ssm_models import ModelConfig, load_ssm_model
from loss import load_loss_functions
from perf import lifes_metrics, sub_lifes_metrics, df_with_index_to_obsidian_table
from plots import plot_predictions_grid

cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
experiment_path = os.path.join(cwd, "experiments", "chronos_exp")

def get_activation(act: str) -> nn.Module:
    """
    Get the activation function

    Args:
        act (str): The name of the activation function for the regression head

    Returns:
        activation (nn.Module): The activation function
    """

    if act == "relu":
        activation = nn.ReLU()
    elif act == "tanh":
        activation = nn.Tanh()
    elif act == "gelu":
        activation = nn.GELU()
    elif act == "glu":
        activation = nn.GLU()
    else:
        raise ValueError(f"Activation function {act} not recognized")

    return activation


class RegressionHead(nn.Module):
    def __init__(
        self,
        sequence_length: int = 500,
        hidden_size: int = 512,
        num_fc_layers: int = 1,
        activation: str = "relu",
        dropout_rate: float = 0.1,
        use_fc_layers: bool = False,
    ):
        super(RegressionHead, self).__init__()

        self.fc = nn.Linear(hidden_size, sequence_length)
        self.use_fc_layers = use_fc_layers
        if self.use_fc_layers:
            self.fc_layers = nn.ModuleList()
            for _ in range(num_fc_layers):
                self.fc_layers.append(nn.Linear(hidden_size, hidden_size))
            self.activation = get_activation(act=activation)
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fc_layers:
            for layer in self.fc_layers:
                x = layer(x)
                x = self.activation(x)
                x = self.dropout(x)

            x = self.fc(x)
            return x

        x = self.fc(x)  # (n_sensors,hidden_size) -> (n_sensors,sequence_length)
        # x = self.dropout(x)
        return x

def train_loop(
    dataloader: DataLoader,
    model: nn.Module,
    config: ExperimentConfig,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Train loop for one epoch

    Args:
        dataloader (DataLoader): The DataLoader object
        model (torch.nn.Module): The model object
        config (ExperimentConfig): The configuration object
        optimizer (torch.optim.Optimizer): The optimizer object
        criterion (torch.nn.Module): The loss function
        device (str): The device to use

    Returns:
        loss (float): The loss value
    """

    model.train()
    train_loss = 0.0
    num_batches = len(dataloader)
    pbar = tqdm(enumerate(dataloader))

    for batch_idx, (life, rul, mask) in pbar:

        life = (
            life.to(device)
            if "padding" in config.approach
            else life.to(device).squeeze(-1)
        )
        rul = rul.to(device).squeeze(-1)
        mask = (
            mask.to(device)
            if config.approach == "padding"
            else mask.to(device).squeeze(-1)
        )

        if config.quantile_reg:
            tau = sample_quantile(
                quantile_dist=config.quantile_dist,
                bounds=config.bounds,
                print_quantile=True,
            )
        else:
            tau = 0.5
            print(f"No quantile regression so tau={tau}")

        output = model(life, tau=tau)

        loss = (
            criterion(output, rul, mask)
            if not config.quantile_reg
            else criterion(output, rul, mask, tau)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()
        gc.collect()

        train_loss += loss.item()

        pbar.set_description(
            f"Batch Idx: {batch_idx}/{len(dataloader)} | Train Loss: {train_loss / (batch_idx + 1):.4f}"
        )

    return train_loss / num_batches

def mixed_train_loop(
    dataloader: DataLoader,
    model: nn.Module,
    config: ExperimentConfig,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str = "cpu",
) -> float:
    """
    Clone of train_loop adapted to mixed precision training.

    Args:
        dataloader (DataLoader): The DataLoader object
        model (torch.nn.Module): The model object
        config (ExperimentConfig): The configuration object
        optimizer (torch.optim.Optimizer): The optimizer object
        criterion (torch.nn.Module): The loss function
        device (str): The device to use

    Returns:
        train_loss (float): training loss
    """

    model.train()
    train_loss = 0.0
    num_batches = len(dataloader)
    pbar = tqdm(enumerate(dataloader))
    scaler = torch.GradScaler("cuda")

    for batch_idx, (life, rul, mask) in pbar:

        life = (
            life.to(device)
            if config.approach == "padding"
            else life.to(device).squeeze(-1)
        )
        rul = rul.to(device).squeeze(-1)
        mask = (
            mask.to(device)
            if config.approach == "padding"
            else mask.to(device).squeeze(-1)
        )

        if config.quantile_reg:
            tau = sample_quantile(
                quantile_dist=config.quantile_dist,
                bounds=config.bounds,
                print_quantile=True,
            )
        else:
            tau = 0.5
            print(f"No quantile regression so tau={tau}")

        optimizer.zero_grad()

        with torch.autocast(device_type=device, dtype=torch.float16):

            output = model(life, tau=tau)

            loss = (
                criterion(output, rul, mask)
                if not config.quantile_reg
                else criterion(output, rul, mask, tau)
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.empty_cache()
        gc.collect()

        train_loss += loss.item()

        pbar.set_description(
            f"Batch Idx: {batch_idx}/{len(dataloader)} | Train Loss: {train_loss / (batch_idx + 1):.4f}"
        )

    return train_loss / num_batches


# Evaluation loop (i.e. validation and test) for one epoch


def eval_loop(
    dataloader: DataLoader,
    model: nn.Module,
    config: ExperimentConfig,
    criterion: nn.Module,
    eval_criterion: nn.Module,
    mode: str = "Test",
    device: str = "cpu",
    use_tqdm: bool = True,
    tau: float = 0.5,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluation loop for one epoch

    Args:
        dataloader (DataLoader): The DataLoader object
        model (torch.nn.Module): The model object
        config (ExperimentConfig): The configuration object
        tokenizer (MeanScaleUniformBinsSensor): The tokenizer object
        criterion (torch.nn.Module): The loss function
        eval_criterion (torch.nn.Module): The evaluation loss function
        mode (str): The mode of evaluation
        device (str): The device to use
        use_tqdm (bool): Whether to use tqdm or not
        tau (float): The quantile level on which the model will be evaluated if the quantile regression approach is used

    Returns:
        eval_loss (float): The evaluation loss value
        eval_rmse_loss (float): The evaluation RMSE loss value
        y_pred (np.ndarray): predictions over the different windows
        y_true (np.ndarray): true values overt the different windows
        mean_an_scores (np.ndarray): anomaly scores over the different windows for the AD approach computing the mean of the column differences
        max_an_scores (np.ndarray): anomaly scores over the different windows for the AD approach computing the max of the column differences
    """

    model.eval()
    eval_loss, eval_rmse_loss = 0.0, 0.0
    num_batches = len(dataloader)
    pbar = tqdm(dataloader) if use_tqdm else dataloader
    y_pred, y_true, mean_an_scores, max_an_scores = [], [], [], []

    with torch.no_grad():
        for life, rul, mask in pbar:
            life = (
                life.to(device)
                if config.approach == "padding"
                else life.to(device).squeeze(-1)
            )
            rul = rul.to(device).squeeze(-1)
            mask = (
                mask.to(device)
                if config.approach == "padding"
                else mask.to(device).squeeze(-1)
            )

            output = model(life, tau=tau)

            batch_out = output.to("cpu").detach().numpy()
            batch_target = rul.to("cpu").detach().numpy()

            #NOTE: Since in both the padding and windowed approach we have
            # mini batches of size > 1 we have to use extend and not append
            y_pred.extend(batch_out)
            y_true.extend(batch_target)

            loss = (
                criterion(output, rul, mask)
                if not config.quantile_reg
                else criterion(output, rul, mask, tau)
            )
            rmse_loss = eval_criterion(output, rul, mask)
            eval_loss += loss.item()
            eval_rmse_loss += rmse_loss.item()

            #NOTE: In the AD case I have to compute the difference sample per sample
            # between the input life and the reconstructed one so that I have an anomaly
            # score for each sample (same dimension of the batch_out and batch_pred
            # that I insert in y_pred and y_true)
            # array to have the anomaly scores over all windows

            if config.ad:
                mean_an_score = np.mean((batch_out-batch_target)**2,axis=2)
                max_an_score = np.max((batch_out-batch_target)**2,axis=2)
                mean_an_scores.extend(mean_an_score)
                max_an_scores.extend(max_an_score)

        eval_loss /= num_batches
        eval_rmse_loss /= num_batches
        print(f"Avg {mode} Loss: {eval_loss:.4f} | Avg {mode} eval Loss: {eval_rmse_loss:.4f}")

    return eval_loss, eval_rmse_loss, np.array(y_pred), np.array(y_true), np.array(mean_an_scores), np.array(max_an_scores)

def mixed_eval_loop(
    dataloader: DataLoader,
    model: nn.Module,
    config: ExperimentConfig,
    criterion: nn.Module,
    eval_criterion: nn.Module,
    mode: str = "Test",
    device: str = "cpu",
    use_tqdm: bool = True,
    tau: float = 0.5,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Clone of eval_loop adapted to mixed precision training

    Args:
        dataloader (DataLoader): The DataLoader object
        model (torch.nn.Module): The model object
        config (ExperimentConfig): The configuration object
        tokenizer (MeanScaleUniformBinsSensor): The tokenizer object
        criterion (torch.nn.Module): The loss function
        eval_criterion (torch.nn.Module): The evaluation loss function
        mode (str): The mode of evaluation
        device (str): The device to use
        use_tqdm (bool): Whether to use tqdm or not
        tau (float): The quantile level on which the model will be evaluated if the quantile regression approach is used

    Returns:
        eval_loss (float): The evaluation loss value
        eval_rmse_loss (float): The evaluation RMSE loss value
        y_pred (np.ndarray): predictions over the different windows
        y_true (np.ndarray): true values overt the different windows
    """

    model.eval()
    eval_loss, eval_rmse_loss = 0.0, 0.0
    num_batches = len(dataloader)
    pbar = tqdm(dataloader) if use_tqdm else dataloader
    y_pred, y_true= [], []

    with torch.no_grad():
        for life, rul, mask in pbar:
            life = (
                life.to(device)
                if config.approach == "padding"
                else life.to(device).squeeze(-1)
            )
            rul = rul.to(device).squeeze(-1)
            mask = (
                mask.to(device)
                if config.approach == "padding"
                else mask.to(device).squeeze(-1)
            )

            with torch.autocast(device_type=device, dtype=torch.float16):

                output = model(life, tau=tau)
                loss = (
                    criterion(output, rul, mask)
                    if not config.quantile_reg
                    else criterion(output, rul, mask, tau)
                )
                rmse_loss = eval_criterion(output, rul, mask)

            torch.cuda.empty_cache()
            gc.collect()

            eval_loss += loss.item()
            eval_rmse_loss += rmse_loss.item()

            #NOTE: Since in full_life we have batch_size=1
            # we have to use append
            batch_out = output.to("cpu").detach().numpy()
            batch_target = rul.to("cpu").detach().numpy()
            y_pred.append(batch_out)
            y_true.append(batch_target)

        eval_loss /= num_batches
        eval_rmse_loss /= num_batches
        print(f"Avg {mode} Loss: {eval_loss:.4f} | Avg {mode} eval Loss: {eval_rmse_loss:.4f}")

    return eval_loss, eval_rmse_loss, np.array(y_pred), np.array(y_true)

# Save the best model

def save_best_model(
    best_model_state_dict: dict,
    best_model_path: str = experiment_path,
) -> None:
    """
    This function saves the best model into best_model_path

    Args:
        best_model_state_dict (dict): The state dictionary of the best model

    Returns:
        The function saves the best model and does not return anything
    """

    save_element(
        element=best_model_state_dict,
        dirpath=best_model_path,
        filename=f"{get_current_time()}_best_model",
        filetype="pickle",
    )
    print("#" * 50)
    print(f"Best model saved at: {best_model_path}")
    print("#" * 50)

def wandb_data(
    config: ExperimentConfig,
    model_config: ModelConfig
) -> Union[
        Tuple[
        DataLoader,
        DataLoader,
        DataLoader,
        nn.Module,
        optim.Optimizer,
        optim.lr_scheduler._LRScheduler,
        nn.Module,
        nn.Module
        ],
        ExperimentConfig
]:
    """
    Function to prepare the data and all the ingredients needed for model training
    and evaluation.

    Args:
        exp_config (ExperimentConfig): ExperimentConfig object
        model_config (ModelConfig): ModelConfig object

    Returns:
        If get_test_idx is false the method returns the following:
            train_loader (DataLoader): train loader
            val_loader (DataLoader): val loader
            test_loader (DataLoader): test loader
            model (nn.Module): the RUL prediction model
            optimizer (optim.Optimizer): optimizer
            lr_scheduler (optim.lr_scheduler): lr scheduler
            criterion (nn.Module): training loss
            eval_criterion (nn.Module): evaluation loss
        otherwise it returns:
            config (ExperimentConfig): experiment configuration object updated with the test_idx
    """

    loaders_dict = load_phm_data(config) if config.data_name == "PHM" else load_reg_data(config)

    train_loader, val_loader, test_loader, test_idx = (
        loaders_dict["train_loader"],
        loaders_dict["val_loader"],
        loaders_dict["test_loader"],
        loaders_dict["test_idx"],
    )
    config.test_idx = test_idx

    if config.get_test_idx:
        return config

    feature_names = get_feature_names(config) if config.data_name == "CMAPSS" else get_phm_feature_names(config)
    d_input = len(feature_names) if ((not config.quantile_reg) or (not model_config.tau_feat)) else len(feature_names) + 1
    model, optimizer, scheduler = load_ssm_model(
        model_config=model_config,
        exp_config=config,
        d_input=d_input
    )
    model = model.to(model_config.device)
    ipdb.set_trace()

    criterion, eval_criterion = load_loss_functions(
        loss_name=config.loss,
        eval_loss_name=config.eval_loss,
    )

    return train_loader, val_loader, test_loader, model, optimizer, scheduler, criterion, eval_criterion

def exp_run(
    config: ExperimentConfig,
    model_config: ModelConfig,
    device: str = "cpu",
    best_model_path: str = experiment_path,
    outputs_path: str = experiment_path,
    metrics_path: str = experiment_path,
    tau: float = 0.5,
) -> None:
    """
    This function implements all the stuff that compose a wandb run: from
    the data pre processing to the model evaluations

    Args:
        config (ExperimentConfig): experiment configuration object
        model_config (ModelConfig): model configuration object
        device (torch.device): CUDA device where to perform the experiment
        best_model_path (str): basepath where to save the best model
        outputs_path (str): basepath where to save the outputs dictionary
        metrics_path (str): basepath where to save the metrics
        tau (float): quantile level for the evaluation

    Returns:
        This function performs the experiment, logs the results on wandb, saves the outputs and
        metrics but does not return anything
    """

    (
        train_loader,
        val_loader,
        test_loader,
        model,
        optimizer,
        scheduler,
        criterion,
        eval_criterion,
    ) = wandb_data(
        config = config,
        model_config = model_config
    )

    wandb_train_test(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        eval_criterion=eval_criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        best_model_path=best_model_path,
        tau=tau,
    )

    if config.return_outputs or config.save_outputs:

        print("#" * 50)
        print("Saving outputs")
        print("#" * 50)

        best_model_perf(
            config=config,
            model_config=model_config,
            device=device,
            best_model_path=best_model_path,
            outputs_path=outputs_path,
            tau=tau,
        )

    if config.compute_metrics:

        print("#" * 50)
        print("Computing metrics for each life and for each sensor in the test set")
        print("#" * 50)

        metrics_df = lifes_metrics(
            config=config,
            outputs_path=outputs_path,
            metrics_path=metrics_path,
            tau = tau
        )
        print("#" * 50)
        print(f"metrics_df shape: {metrics_df.shape}")

    if config.obsidian_table:

        print("#" * 50)
        print("Producing the obsidian table")
        print("#" * 50)

        metrics_path = get_most_recent_file(metrics_path, file_pos=config.file_pos)
        metrics_df = open_element(metrics_path)

        if config.sub_lifes_metrics:
            sub_metrics_df = sub_lifes_metrics(config=config, metrics_df=metrics_df)
            # obsidian_table = df_with_index_to_obsidian_table(sub_metrics_df)
            obsidian_table = sub_metrics_df.to_markdown()
        else:
            # obsidian_table = df_with_index_to_obsidian_table(metrics_df)
            obsidian_table = metrics_df.to_markdown()

        print(obsidian_table)

# Function to train and test the model on a wandb run

def wandb_train_test(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    eval_criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    config: ExperimentConfig,
    device: str = "cpu",
    best_model_path: str = experiment_path,
    tau: float = 0.5,
) -> None:
    """
    Train and test the model on a wandb run and log the metrics

    Args:
        model (nn.Module): The model object
        train_loader (DataLoader): The DataLoader object for training
        val_loader (DataLoader): The DataLoader object for validation
        test_loader (DataLoader): The DataLoader object for testing
        criterion (nn.Module): The loss function
        optimizer (optim.Optimizer): The optimizer object
        scheduler (optim.lr_scheduler._LRScheduler): The scheduler object
        exp_config (ExperimentConfig): The configuration object
        device (str): The device to use
        best_model_path (str): The path to save the best model
        tau (float): The quantile level on which the model will be evaluated if the quantile regression approach is used

    Returns:
        None: the model does not return anything
    """

    if config.use_wandb:
        wandb.watch(model, criterion, log="all", log_freq=10)
        wandb.define_metric("epoch")
        wandb.define_metric("loss/*", step_metric="epoch")
        wandb.define_metric("times/*", step_metric="epoch")
        wandb.define_metric("eval_loss/*", step_metric="epoch")

    error = False
    train_times, val_times, test_times = [], [], []
    min_val_loss = np.inf
    best_model_state_dict = model.state_dict()
    pbar = tqdm(range(config.epochs))

    try:
        for epoch in pbar:
            if epoch == 0:
                pbar.set_description("Epoch: %d" % (epoch))
                val_loss, test_loss = 0.0, 0.0
            else:
                pbar.set_description(
                    f"Epoch: {epoch:d} | Val loss: {val_loss:1.3f} | Eval Val loss: {eval_val_loss:1.3f}"
                )
                pbar.set_description(
                    f"Epoch: {epoch:d} | Test loss: {test_loss:1.3f} | Eval Test loss: {eval_test_loss:1.3f}"
                )

            train_time = time.time()

            if config.approach == "full_life":

                print("-"*50)
                print(f"We are in approach {config.approach} so we use mixed train_loop")
                print("-"*50)

                train_loss = mixed_train_loop(
                    dataloader=train_loader,
                    model=model,
                    config=config,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                )

            else:

                print("-"*50)
                print(f"We are in approach {config.approach} so we use train_loop")
                print("-"*50)

                train_loss = train_loop(
                    dataloader=train_loader,
                    model=model,
                    config=config,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                )

            train_time = time.time() - train_time

            val_time = time.time()

            if config.approach == "full_life":

                print("-"*50)
                print(f"We are in approach {config.approach} so we use mixed_val_loop")
                print("-"*50)

                val_loss, eval_val_loss, y_pred, y_true = mixed_eval_loop(
                    dataloader=val_loader,
                    model=model,
                    config=config,
                    criterion=criterion,
                    eval_criterion=eval_criterion,
                    mode="Val",
                    device=device,
                    tau=tau,
                )

            else:

                print("-"*50)
                print(f"We are in approach {config.approach} so we use val_loop")
                print("-"*50)

                val_loss, eval_val_loss, y_pred, y_true, _, _ = eval_loop(
                    dataloader=val_loader,
                    model=model,
                    config=config,
                    criterion=criterion,
                    eval_criterion=eval_criterion,
                    mode="Val",
                    device=device,
                    tau=tau,
                )

            val_time = time.time() - val_time

            test_time = time.time()

            if config.approach == "full_life":

                print("-"*50)
                print(f"We are in approach {config.approach} so we use mixed_eval_loop")
                print("-"*50)

                test_loss, eval_test_loss, y_pred, y_true = mixed_eval_loop(
                    dataloader=test_loader,
                    model=model,
                    config=config,
                    criterion=criterion,
                    eval_criterion=eval_criterion,
                    mode="Test",
                    device=device,
                    tau=tau,
                )

            else:

                print("-"*50)
                print(f"We are in approach {config.approach} so we use eval_loop")
                print("-"*50)

                test_loss, eval_test_loss, y_pred, y_true, _, _ = eval_loop(
                    dataloader=test_loader,
                    model=model,
                    config=config,
                    criterion=criterion,
                    eval_criterion=eval_criterion,
                    mode="Test",
                    device=device,
                    tau=tau,
                )

            test_time = time.time() - test_time

            if scheduler is not None:
                scheduler.step()
                print(f"Epoch {epoch} learning rate: {scheduler.get_last_lr()}")

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                print(f"Epoch {epoch} | New best model found with val loss: {min_val_loss}")
                print("#" * 50)
                best_model_state_dict = model.state_dict().copy()

            train_times.append(train_time)
            val_times.append(val_time)
            test_times.append(test_time)

            model_info = {
                "epoch": epoch,
                "times/train_time": train_time,
                "times/val_time": val_time,
                "times/test_time": test_time,
                "loss/train_loss": train_loss,
                "loss/val_loss": val_loss,
                "eval_loss/eval_val_loss": eval_val_loss,
                "loss/test_loss": test_loss,
                "eval_loss/eval_test_loss": eval_test_loss,
                }

            if config.use_wandb:
                wandb.log(model_info)
            else:
                model_info_df = pd.DataFrame(model_info,index=["values"])
                print("-"*50)
                print("Information on the model training and evaluation:")
                print(model_info_df.T)
                print("-"*50)

    except KeyboardInterrupt:
        print("-"*50)
        print("Manual Early Stopping triggered. Saving the best model up to now")
        print("-"*50)

        if config.save_best_model:
            save_best_model(
                best_model_state_dict=best_model_state_dict,
                best_model_path=best_model_path,
            )

    except torch.cuda.OutOfMemoryError:
        print("-"*50)
        print("CUDA Out of Memory Error, stopping execution")
        print("-"*50)
        traceback.print_exc()  # Print the full traceback of the error
        error = True
        quit()

    except Exception as e:
        print("-"*50)
        print("An error occured during the training process:")
        print("-"*50)
        print(e)
        traceback.print_exc()  # Print the full traceback of the error
        error = True

    if not error:
        print("No errors occured during the training process, saving the best model")
        save_best_model(
            best_model_state_dict=best_model_state_dict,
            best_model_path=best_model_path
        )

def load_best_model(
    config: ExperimentConfig,
    model_config: ModelConfig,
    best_model_path: str
) -> nn.Module:
    """
    This function loads the best model given the best model path

    Args:
        config (ExperimentConfig): experiment configuration object
        best_model_path (str): path to the best model

    Returns:
        model (nn.Module): best model
    """

    best_model_filepath = get_most_recent_file(dirpath=best_model_path, file_pos=config.file_pos)

    best_model_state_dict = open_element(best_model_filepath, filetype="pickle")

    feature_names = get_feature_names(config) if config.data_name == "CMAPSS" else get_phm_feature_names(config)

    if config.save_summary_dict:
        model, summary_dict = load_ssm_model(
            model_config=model_config,
            exp_config=config,
            d_input=len(feature_names)
            if ((not config.quantile_reg) or (not model_config.tau_feat))
            else len(feature_names) + 1,
        )
        print("#" * 50)
        print(f"Summary dict keys: {summary_dict.keys()}")
        print("#" * 50)
    else:
        model, _, _ = load_ssm_model(
            model_config=model_config,
            exp_config=config,
            d_input=len(feature_names)
            if ((not config.quantile_reg) or (not model_config.tau_feat))
            else len(feature_names) + 1,
        )

    model.load_state_dict(best_model_state_dict)
    model = model.to(model_config.device)

    return model

# Function to get the best model performance


def best_model_perf(
    config: ExperimentConfig,
    model_config: ModelConfig,
    device: torch.device = torch.device("cpu"),
    best_model_path: str = experiment_path,
    outputs_path: str = experiment_path,
    tau: float = 0.5,
) -> Union[None, nn.Module, dict]:
    """
    This function loads the best model according to the validation set and
    computes the performance on the test set. In particular it saves
    (and potentially returns) the
    prediction and true values obtained on the different test lifes

    Args:
        config (ExperimentConfig): The configuration dictionary
        model_config (ModelConfig): The model configuration object
        device (str): The device to use
        best_model_path (str): The path to save the best model
        outputs_path (str): The path to save the outputs
        plot_path (str): The path to save the plots
        metrics_path (str): The path to save the test metrics

    Returns:
        Union[None,nn.Module,dict]: The function saves the plots and the metrics and does not return anything
            If model_summary is set to True the function returns the model object, but it will not save the outputs
            If return_outputs is set to True the function returns the outputs dictionary, but it will not save the outputs

    """

    loaders_dict = load_phm_data(config,eval=True) if config.data_name == "PHM" else load_reg_data(config)
    test_lifes = loaders_dict["test_lifes"]
    test_loaders = loaders_dict["test_loaders"]
    test_idx = loaders_dict["test_idx"] if config.data_name == "PHM" else None

    model = load_best_model(
        config = config,
        model_config = model_config,
        best_model_path = best_model_path
    )

    if config.model_summary:
        return model

    criterion, eval_criterion = load_loss_functions(
        loss_name=config.loss,
        eval_loss_name=config.eval_loss,
    )

    print("#" * 50)
    print("Evaluating the best model on the test set")
    print("#" * 50)

    preds, true_vals, mean_anomaly_scores, max_anomaly_scores = [], [], [], []

    for i, test_loader in enumerate(test_loaders):

        print("#" * 50)
        print(f"Testing on life {i+1+config.test_idx[0]}") if config.data_name == "CMAPSS" else print(f"Testing on life {test_idx[i]}")
        print("#" * 50)

        _, _, y_pred, y_true, mean_an_scores, max_an_scores = eval_loop(
            dataloader=test_loader,
            model=model,
            config=config,
            criterion=criterion,
            eval_criterion=eval_criterion,
            mode="Test",
            device=device,
            use_tqdm=False,
            tau=tau,
        )

        if config.normalize_rul:

            y_pred = y_pred*MAX_RUL
            y_true = y_true*MAX_RUL

        if config.approach == "padding":

            preds.append(y_pred)
            true_vals.append(y_true)

        else:

            combined_preds, combined_true_vals, combined_mean_an_scores = combine_values(
                predictions = y_pred,
                true_values = y_true,
                an_scores = mean_an_scores,
                original_shape = test_lifes[test_idx[i]].shape[0],
                sequence_length = config.sequence_length,
                stride = config.stride
            )

            _, _, combined_max_an_scores = combine_values(
                predictions = y_pred,
                true_values = y_true,
                an_scores = max_an_scores,
                original_shape = test_lifes[test_idx[i]].shape[0],
                sequence_length = config.sequence_length,
                stride = config.stride
            )

            preds.append(combined_preds)
            true_vals.append(combined_true_vals)
            mean_anomaly_scores.append(combined_mean_an_scores)
            max_anomaly_scores.append(combined_max_an_scores)

    outputs_dict = {
        "y_pred": preds,
        "y_true": true_vals,
        "mean_an_scores": mean_anomaly_scores,
        "max_an_scores": max_anomaly_scores,
    }

    filename=f"{get_current_time()}_outputs_mean_max_an_scores_{config.model_name}_{config.cmapss_models}" if config.data_name == "CMAPSS" else f"{get_current_time()}_outputs_{config.model_name}"

    if config.save_outputs:
        save_element(
            element=outputs_dict,
            dirpath=outputs_path,
            filename=filename,
            filetype="pickle",
        )

    if config.return_outputs:
        return outputs_dict

# Function that implements a wandb run

def wandb_run(
    run_name: str,
    config: ExperimentConfig,
    model_config: ModelConfig,
    device: str = "cpu",
    best_model_path: str = experiment_path,
    outputs_path: str = experiment_path,
    metrics_path: str = experiment_path,
    tau: float = 0.5,
) -> None:
    """
    Function that implements a wandb run. It's essentially a wrapper of
    the exp_run function with two different ways of calling it depending on
    weather we want to log the results on wandb or not

    Args:
        run_name (str): The name of the run
        config (ExperimentConfig): The experiment configuration object
        model_config (ModelConfig): The model configuration object
        device (str): The device to use
        best_model_path (str): The path to save the best model
        outputs_path (str): The path to save the outputs
        metrics_path (str): The path to save the metrics
        tau (float): The quantile level on which the model will be evaluated if the quantile regression approach is used

    Returns:
        None: Performs a wandb run and does not return anything
    """

    if config.use_wandb:

        with wandb.init(project=config.project_name, name=run_name):
            exp_run(
                config = config,
                model_config = model_config,
                device = device,
                best_model_path = best_model_path,
                outputs_path = outputs_path,
                metrics_path = metrics_path,
                tau = tau
            )
    else:

        exp_run(
            config = config,
            model_config = model_config,
            device = device,
            best_model_path = best_model_path,
            outputs_path = outputs_path,
            metrics_path = metrics_path,
            tau = tau
        )

