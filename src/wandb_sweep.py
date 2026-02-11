"""
Python module containing all the functions needed to perform a wandb sweep
"""
import os
import ipdb
import setproctitle
import time
from tqdm.auto import tqdm
import numpy as np
import wandb
from wandb.sdk.wandb_config import Config as WandbConfig
from exp_config import (
    set_sweep_name,
    setup_exp,
    ExperimentConfig,
    define_arguments,
)
from ssm_models import ModelConfig
from models import (
    wandb_data,
    train_loop,
    eval_loop,
)
from utils import load_yaml_to_dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

experiment_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"experiments","phm_exp")
sweep_config_path = os.path.join(experiment_path,"config","sweep_config.yaml")

def define_sweep_config(
    sweep_config_path: str,
    script_path: str
) -> dict:
    """
    This function uses the hyperparameters set on the yaml sweep file
    to define the dictionary containing the sweep configuration

    Args:
        sweep_config_path (str): path to the sweep yaml config file
        script_path (str): path to the script launching the sweep

    Returns:
        sweep_config (dict): sweep configuration dictionary
    """

    args = define_arguments()
    sweep_config_dict = load_yaml_to_dict(sweep_config_path)
    config = ExperimentConfig.from_dict(sweep_config_dict)
    config.add_params(args=args.__dict__)

    if config.sweep_name == "sweep":
        sweep_name = set_sweep_name(config = config)
    else:
        sweep_name = config.sweep_name

    sweep_config = {
        "program": script_path,
        "name": sweep_name,
        "method": config.sweep_method,
        "metric": {"goal": "minimize", "name": "score"},
        "parameters": {}
    }

    for param_name in config.sweep_param_names:
        param_vals = f"{param_name}_vals"
        if hasattr(config,param_vals):
            if param_name in ["lr", "dropout"]:
                sweep_config["parameters"][param_name] = {"min": getattr(config,param_vals)[0], "max": getattr(config,param_vals)[1]}
            else:
                sweep_config["parameters"][param_name] = {"values": getattr(config,param_vals)}
        else:
            print("-"*50)
            print(f"Error: {param_vals} is not a valid config parameter")
            print("-"*50)
            break

    return sweep_config

def set_sweep_exp_name(
    exp_config: ExperimentConfig,
    wandb_config: WandbConfig,
) -> str:
    """
    This function creates the string that will be used to set the name of the single
    experiments produced by a wandb sweep by concatenating together the hyperparameter
    values for the current run

    Args:
        exp_config (ExperimentConfig): experiment config object
        wandb_config (WandConfig): wandb config object

    Returns:
        sweep_run_name (str): name of the current run in the wandb sweep
    """

    sweep_run_name = exp_config.sweep_name

    for key in wandb_config.keys():

        sweep_run_name = f"{sweep_run_name}_{key}_{wandb_config[key]}"

    print("-"*50)
    print(f"Run name set to {sweep_run_name}")
    print("-"*50)

    return sweep_run_name

def wandb_train_test_sweep(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    eval_criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    config: ExperimentConfig,
    device: torch.device = torch.device("cpu"),
    tau: float = 0.5,
) -> float:
    """
    Modified version of wandb_train_test to adapt to the wandb sweep experiment.
    The modifications are the following:
    - only the training and validation loop are executed (we only need to track the validation loss)
    - no model is saved
    - the min_val_loss is returned

    Args:
        model (nn.Module): The model object
        train_loader (DataLoader): The DataLoader object for training
        val_loader (DataLoader): The DataLoader object for validation
        criterion (nn.Module): The loss function
        optimizer (optim.Optimizer): The optimizer object
        scheduler (optim.lr_scheduler._LRScheduler): The scheduler object
        exp_config (ExperimentConfig): The configuration object
        device (str): The device to use
        tau (float): The quantile level on which the model will be evaluated if the quantile regression approach is used

    Returns:
        min_val_loss: minimum validation loss over the epochs.
        This is the metric that will be tracked in the sweep
    """

    wandb.watch(model, criterion, log="all", log_freq=10)
    train_times, val_times = [], []
    min_val_loss = np.inf
    pbar = tqdm(range(config.epochs))

    for epoch in pbar:
        if epoch == 0:
            pbar.set_description("Epoch: %d" % (epoch))
            val_loss = 0.0
        else:
            pbar.set_description(
                f"Epoch: {epoch:d} | Val loss: {val_loss:1.3f} | Eval Val loss: {eval_val_loss:1.3f}"
            )

        train_time = time.time()
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
        val_loss, eval_val_loss, _, _ = eval_loop(
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

        if scheduler is not None:
            scheduler.step()
            print(f"Epoch {epoch} learning rate: {scheduler.get_last_lr()}")

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            print(f"Epoch {epoch} | New best model found with val loss: {min_val_loss}")
            print("#" * 50)

        train_times.append(train_time)
        val_times.append(val_time)

        model_info = {
                "train_time": train_time,
                "val_time": val_time,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "eval_val_loss": eval_val_loss,
            }

        wandb.log(model_info)

    return min_val_loss


def exp_run_sweep(
    sweep_config: dict,
    wandb_config: WandbConfig,
    exp_config: ExperimentConfig,
    model_config: ModelConfig,
    device: torch.device = torch.device("cpu")
) -> float:
    """
    This is the function that returns the score being tracked
    by the wandb sweep experiment. It's a modification of exp_run that
    loads the data, creates the model, optimizer and scheduler, runs the training
    and evaluation part and then returns in output the minimum validation loss as
    the score

    Args:
        sweep_config (dict): dictionary containing the sweep config
        wandb_config (WandbConfig): wandb config object
        exp_config (Config): experiment config object
        model_config (ModelConfig): model config object
        device (torch.device): CUDA device where to run the sweep experiments

    Returns:
        score (float): score of the run (i.e. minimum validation loss obtained over the different epochs)
    """

    for param_name in sweep_config["parameters"].keys():
        if param_name in exp_config.__dict__.keys():
            setattr(exp_config,param_name,getattr(wandb_config,param_name))
        elif param_name in model_config.__dict__.keys():
            setattr(model_config,param_name,getattr(wandb_config,param_name))
        else:
            raise ValueError(f"Error: {param_name} is neither in the experiment or the model config")

    (
        train_loader,
        val_loader,
        _,
        model,
        optimizer,
        scheduler,
        criterion,
        eval_criterion,
    ) = wandb_data(
        config = exp_config,
        model_config = model_config
    )

    min_val_loss = wandb_train_test_sweep(
        model = model,
        train_loader = train_loader,
        val_loader = val_loader,
        criterion = criterion,
        eval_criterion = eval_criterion,
        optimizer = optimizer,
        scheduler = scheduler,
        config = exp_config,
        device = device,
    )

    return min_val_loss

def wandb_run_sweep():
    """
    Function to pass to wandb.sweep to run a wandb sweep.
    Following the wandb sweep docs this function must not have any input arguments or return anything.
    The function does the following:
        - call the exp_run_sweep function that returns the metric to track (the validation loss) and that takes as input a wandb config object
        → this will be a modified version of the exp_run function
        - log the returned score to wandb
    """

    exp_config, model_config, device, _ = setup_exp()
    sweep_config = define_sweep_config(sweep_config_path)

    with wandb.init(project=exp_config.project_name) as sweep_run:

        sweep_run.name = set_sweep_exp_name(
            exp_config = exp_config,
            wandb_config = sweep_run.config
        )

        setproctitle.setproctitle(sweep_run.name)

        score = exp_run_sweep(
            sweep_config = sweep_config,
            wandb_config = sweep_run.config,
            exp_config = exp_config,
            model_config = model_config,
            device = device
        )
        sweep_run.log({"score": score})


