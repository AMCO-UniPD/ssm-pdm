"""
Python script containing utility functions for the models of the `ssm-pdm` project
"""

import os
from typing import Tuple, Union

import ipdb
import numpy as np

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# wandb imports
from wandb.sdk.wandb_run import Run as WandbRun

# from apex.optimizers import FusedAdam
from torch.utils.data import DataLoader

from config_vars import MAX_RUL, BASELINE_MODEL_NAMES
from evaluator import get_life_evaluator
from exp_config import ModelConfig
from loss import load_loss_functions
from perf import lifes_metrics

# general imports
from trainer import get_trainer
from model_classes import (
    RULModel,
    QuantileRULModel,
    MonotonicRULModel,
    MonoQuantileRULModel
)
from utils import (
    ExperimentConfig,
    combine_values,
    get_current_time,
    get_feature_names,
    get_mono_mask,
    get_most_recent_file,
    get_phm_feature_names,
    load_phm_data,
    load_reg_data,
    open_element,
    save_element,
)
from ceruleo.models.baseline import BaselineModel

cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
experiment_path = os.path.join(cwd, "experiments", "phm_exp")

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

        # (n_sensors,hidden_size) -> (n_sensors,sequence_length)
        x = self.fc(x)
        # x = self.dropout(x)
        return x

def setup_optimizer(model, lr, weight_decay, epochs):
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim")
           for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s)
        for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(
            p, "_optim", None) == hp]
        optimizer.add_param_group({"params": params, **hp})

    # Create a lr scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        # print(' | '.join([
        #     f"Optimizer group {i}",
        #     f"{len(g['params'])} tensors",
        # ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler

def load_best_model(
    config: ExperimentConfig, model_config: ModelConfig, best_model_path: str
) -> Tuple[nn.Module, np.ndarray]:
    """
    This function loads the best model given the best model path

    Args:
        config (ExperimentConfig): experiment configuration object
        best_model_path (str): path to the best model

    Returns:
        model (nn.Module): best model
        mono_mask (np.ndarray): monotonic mask
    """

    best_model_filepath = get_most_recent_file(
        dirpath=best_model_path, file_pos=config.file_pos
    )

    best_model_state_dict = open_element(best_model_filepath, filetype="pickle")

    feature_names = (
        get_feature_names(config)
        if config.data_name == "CMAPSS"
        else get_phm_feature_names(config)
    )
    mono_mask = get_mono_mask(config=config, feature_names=feature_names)

    if config.save_summary_dict:
        model, summary_dict = load_ssm_model(
            model_config=model_config,
            exp_config=config,
            d_input=(
                len(feature_names)
                if ((not config.quantile_reg) or (not model_config.tau_feat))
                else len(feature_names) + 1
            ),
        )
        print("#" * 50)
        print(f"Summary dict keys: {summary_dict.keys()}")
        print("#" * 50)
    else:
        model, _, _ = load_ssm_model(
            exp_config=config,
            model_config=model_config,
            model_name=config.model_name,
            output_size=config.sequence_length,
            mono_mask=mono_mask,
        )

    model.load_state_dict(best_model_state_dict)
    model = model.to(model_config.device)

    return model, mono_mask

# Function to create the model

def load_ssm_model(
    exp_config: ExperimentConfig,
    mono_mask: np.ndarray = np.zeros(shape=(10, 1)),
    tau: float = 0.5,
    n_const_wins: int = 1000,
    n_decreasing_wins: int = 1000,
    **kwargs
) -> Tuple[nn.Module, optim.Optimizer, optim.lr_scheduler]:
    """
    Function to create the model based on the configuration.
    The function also sets up the optimizer and the scheduler.

    Args:
        exp_config (ExperimentConfig): experiment configuration object
        model_config (ModelConfig): model configuration object
        mono_mask (np.ndarray): boolean mask to identify monotonic features
        tau (float): The quantile level on which the model will be evaluated if the quantile regression approach is used

    Returns:
        model: nn.Module object
        optimizer: torch.optim object
        scheduler: torch.optim.lr_scheduler object
    """

    if exp_config.quantile_reg and exp_config.monotonic:
        model = MonoQuantileRULModel(tau=tau, mono_mask=mono_mask, **kwargs)
    elif not exp_config.quantile_reg and exp_config.monotonic:
        model = MonotonicRULModel(mono_mask=mono_mask, **kwargs)
    elif exp_config.quantile_reg and not exp_config.monotonic:
        model = QuantileRULModel(tau=tau, **kwargs)
    else:
        model = RULModel(**kwargs)

    optimizer, scheduler = setup_optimizer(
        model,
        lr=exp_config.lr,
        weight_decay=exp_config.weight_decay,
        epochs=exp_config.epochs,
    )

    return model, optimizer, scheduler

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


def compute_window_info(
    loaders_dict: dict,
    config: ExperimentConfig,
) -> None:
    """
    Compute informations on the windows used in the windowed
    approaches, needed to compute the loss weights

    Args:
        loaders_dict (dict): dictionary with dataloaders and windows info
        config (ExperimentConfig): experiment configuration object to update
        with windows informations

    Returns:
        None: the config object is modified in place
    """

    train_wins = {
        "constant": loaders_dict["n_train_constant_wins"],
        "decreasing": loaders_dict["n_train_decreasing_wins"],
    }
    val_wins = {
        "constant": loaders_dict["n_val_constant_wins"],
        "decreasing": loaders_dict["n_val_decreasing_wins"],
    }
    test_wins = {
        "constant": loaders_dict["n_test_constant_wins"],
        "decreasing": loaders_dict["n_test_decreasing_wins"],
    }

    setattr(config, "train_wins", train_wins)
    setattr(config, "val_wins", val_wins)
    setattr(config, "test_wins", test_wins)


def wandb_data(
    config: ExperimentConfig,
    model_config: ModelConfig,
    best_model_path: str = os.getcwd()
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
    Function to prepare the data and all the ingredients needed for model training
    and evaluation.

    Args:
        exp_config (ExperimentConfig): ExperimentConfig object
        model_config (ModelConfig): ModelConfig object
        best_model_path (str): path containing the best model. Needed in
        case we want to use the resume training option

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
            config (ExperimentConfig): experiment configuration object
            updated with mono_mask and constant
            and decreasing window information
        otherwise it returns:
            config (ExperimentConfig): experiment configuration object updated with the test_idx
    """

    loaders_dict = (
        load_phm_data(config) if config.data_name == "PHM" else load_reg_data(config)
    )

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

    feature_names = (
        get_feature_names(config)
        if config.data_name == "CMAPSS"
        else get_phm_feature_names(config)
    )
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


def exp_run(
    config: ExperimentConfig,
    model_config: ModelConfig,
    runWB: Union[WandbRun, None],
    device: str = "cpu",
    best_model_path: str = experiment_path,
    outputs_path: str = experiment_path,
    combined_outputs_path: str = experiment_path,
    metrics_path: str = experiment_path,
    tau: float = 0.5,
    mono_mask: np.ndarray = np.zeros(shape=(10, 1)),
) -> None:
    """
    This function implements all the stuff that compose a wandb run: from
    the data pre processing to the model evaluations

    Args:
        config (ExperimentConfig): experiment configuration object
        model_config (ModelConfig): model configuration object
        runWB (Union[WandbRun, None]): WandbRun instance
        device (torch.device): CUDA device where to perform the experiment
        best_model_path (str): basepath where to save the best model
        outputs_path (str): basepath where to save the outputs dictionary
        combined_outputs_path (str): basepath where to save the outputs dictionary
        metrics_path (str): basepath where to save the metrics
        tau (float): quantile level for the evaluation
        mono_mask (np.ndarray): boolean mask to identify monotonic features

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
        config,
    ) = wandb_data(
        config=config,
        model_config=model_config,
        best_model_path=best_model_path
    )

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
        best_model_path=best_model_path,
        config=config,
        tau=tau,
    )

    trainer.run(runWB=runWB)

    if config.return_outputs or config.save_outputs or config.save_combined_outputs:

        print("#" * 50)
        print("Saving outputs")
        print("#" * 50)

        best_model_perf(
            config=config,
            model_config=model_config,
            device=device,
            best_model_path=best_model_path,
            outputs_path=outputs_path,
            combined_outputs_path=combined_outputs_path,
            tau=tau,
        )

    if config.compute_metrics:

        print("#" * 50)
        print("Computing metrics for each life in the test set")
        print("#" * 50)

        metrics_df = lifes_metrics(
            config=config, outputs_path=outputs_path, metrics_path=metrics_path, tau=tau
        )
        print("#" * 50)
        print(f"metrics_df shape: {metrics_df.shape}")

def load_baseline_model(
    model_name: str = "mean",
    tau: float = 0.5
) -> BaselineModel:
    """
    Load a baseline RUL model representing a Preventive Maintenance
    approach

    Args:
        model_name (str): statistics used to produce the prediction
        tau (float): quantile level if using the quantile baseline models

    Returns:
        baseline_model (BaselineModel): BaselineModel instance representing
        the baseline model
    """

    mode = "quantile" if "quantile" in model_name else model_name

    baseline_model = BaselineModel(mode=mode, tau=tau)
    return baseline_model


# Function to get the best model performance


def best_model_perf(
    config: ExperimentConfig,
    model_config: ModelConfig,
    device: torch.device = torch.device("cpu"),
    best_model_path: str = experiment_path,
    outputs_path: str = experiment_path,
    combined_outputs_path: str = experiment_path,
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
        metrics_path (str): The path to save the test metrics
        outputs_path (str): The path to save the outputs
        combined_outputs_path (str): The path to save the combined outputs
        tau (float): The quantile level on which the model will be evaluated if the quantile regression approach is used

    Returns:
        Union[None,nn.Module,dict]: The function saves the plots and the metrics and does not return anything
            If model_summary is set to True the function returns the model object, but it will not save the outputs
            If return_outputs is set to True the function returns the outputs dictionary, but it will not save the outputs
    """

    loaders_dict = (
        load_phm_data(config, eval=True)
        if config.data_name == "PHM"
        else load_reg_data(config)
    )
    test_lifes = loaders_dict["test_lifes"]
    test_loaders = loaders_dict["test_loaders"]
    test_idx = loaders_dict["test_idx"] if config.data_name == "PHM" else None

    model, mono_mask = load_best_model(
        config=config, model_config=model_config, best_model_path=best_model_path
    )

    if config.model_summary:
        return model

    _, life_criterion = load_loss_functions(
        loss_name=config.loss,
        eval_loss_name=config.life_eval_loss,
        tau=tau,
    )

    print("#" * 50)
    print("Evaluating the best model on the test set")
    print("#" * 50)

    preds, true_vals = [], []

    if config.save_combined_outputs:
        combined_preds_list, combined_true_vals_list = [], []

    for i, test_loader in enumerate(test_loaders):

        print("#" * 50)
        (
            print(f"Testing on life {i+1+config.test_idx[0]}")
            if config.data_name == "CMAPSS"
            else print(f"Testing on life {test_idx[i]}")
        )
        print("#" * 50)

        evaluator = get_life_evaluator(
            model=model,
            config=config,
            life_criterion=life_criterion,
            device=device,
            tau=tau,
        )

        _, y_pred, y_true = evaluator.life_eval_loop(
            loader=test_loader, mode="Test", use_tqdm=False
        )

        if hasattr(config,"rul_scaler"):
            y_pred = config.rul_scaler.inverse_transform(y_pred)
            y_true = config.rul_scaler.inverse_transform(y_true)

        if config.normalize_rul:

            y_pred = y_pred * MAX_RUL
            y_true = y_true * MAX_RUL

        #NOTE: If we are in the padding approach
        # here we have a single sequence predicting the RUL → (1,sequence_length)
        # In case we are in the windowed approach we have a set
        # of predictions on the different windows → (n_windows,sequence_length).
        # In both cases we have np.arrays so we simply need to append them to the
        # preds and true_vals lists

        preds.append(y_pred)
        true_vals.append(y_true)

        if config.save_combined_outputs:

            combined_preds, combined_true_vals = combine_values(
                predictions=y_pred,
                true_values=y_true,
                original_shape=test_lifes[test_idx[i]].shape[0],
                sequence_length=config.sequence_length,
                stride=config.stride
            )

            combined_preds_list.append(combined_preds)
            combined_true_vals_list.append(combined_true_vals)

    if config.save_outputs:

        outputs_dict = {
            "y_pred": preds,
            "y_true": true_vals,
        }

        filename = (
            f"{get_current_time()}_outputs_{config.model_name}_{config.cmapss_models}"
            if config.data_name == "CMAPSS"
            else f"{get_current_time()}_outputs_{config.model_name}"
        )
        save_element(
            element=outputs_dict,
            dirpath=outputs_path,
            filename=filename,
            filetype="pickle",
        )

    if config.save_combined_outputs:

        combined_outputs_dict = {
            "y_pred": combined_preds_list,
            "y_true": combined_true_vals_list,
        }

        filename = (
            f"{get_current_time()}_combined_outputs_{config.model_name}_{config.cmapss_models}"
            if config.data_name == "CMAPSS"
            else f"{get_current_time()}_combined_outputs_{config.model_name}"
        )

        save_element(
            element=combined_outputs_dict,
            dirpath=combined_outputs_path,
            filename=filename,
            filetype="pickle",
        )

    if config.return_outputs:
        return outputs_dict
