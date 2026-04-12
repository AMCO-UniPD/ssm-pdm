"""
Python module containing all the evaluator classes
"""

import copy
import gc
import os
from typing import Tuple, Union

import ipdb
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils import (
    ExperimentConfig,
    split_input
)

cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
experiment_path = os.path.join(cwd, "experiments", "phm_exp")


class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        criterion: nn.Module,
        eval_criterion: nn.Module,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Evaluator base class
        """

        self.model = model
        self.config = config
        self.criterion = criterion
        self.eval_criterion = eval_criterion
        self.device = device

    def eval_loop(self, loader: DataLoader, mode: str = "Val", use_tqdm: bool = True):
        """
        Method implementing an evaluation loop on a single epoch

        Args:
            loader (DataLoader): dataloader to use
            mode (str): evaluation mode (i.e. validation or test)
            use_tqdm (bool): weather to use tqdm to show the loading bar
        """
        raise NotImplementedError

# NOTE: Standard Evaluator → no quantile_reg and no monotonic


class StandardEvaluator(Evaluator):
    """
    Standard Evaluator class without any fancy approach
    """

    def eval_loop(
        self, loader: torch.Tensor, mode: str = "Val", use_tqdm: bool = True
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Implementation of eval_loop for StandardEvaluator
        """

        self.model.eval()
        eval_loss, eval_rmse_loss = 0.0, 0.0
        num_batches = len(loader)
        pbar = tqdm(loader) if use_tqdm else loader
        y_pred, y_true = [], []

        with torch.no_grad():
            for life, rul, mask in pbar:
                life = (
                    life.to(self.device)
                    if "padding" in self.config.approach
                    else life.to(self.device).squeeze(-1)
                )
                rul = rul.to(self.device).squeeze(-1)
                mask = (
                    mask.to(self.device)
                    if "padding" in self.config.approach
                    else mask.to(self.device).squeeze(-1)
                )

                output = self.model(life)

                batch_out = output.to("cpu").detach().numpy()
                batch_target = rul.to("cpu").detach().numpy()

                # NOTE: Since in both the padding and windowed approach we have
                # mini batches of size > 1 we have to use extend and not append
                y_pred.extend(batch_out)
                y_true.extend(batch_target)

                loss = self.criterion(output, rul, mask)
                rmse_loss = self.eval_criterion(output, rul, mask)
                eval_loss += loss.item()
                eval_rmse_loss += rmse_loss.item()

            eval_loss /= num_batches
            eval_rmse_loss /= num_batches
            print(
                f"Avg {mode} Loss: {eval_loss:.4f} | Avg {mode} eval Loss: {eval_rmse_loss:.4f}"
            )

        return eval_loss, eval_rmse_loss, np.array(y_pred), np.array(y_true)

# NOTE: QuantileEvaluator → quantile_reg and no monotonic


class QuantileEvaluator(Evaluator):
    """
    QuantileEvaluator subclasses used to train the models using Quantile
    Regression

    Args:
        tau (float): The quantile level on which the model will be evaluated if the quantile regression approach is used
    """

    def __init__(self, tau: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tau = tau

    def eval_loop(
        self, loader: torch.Tensor, mode: str = "Val", use_tqdm: bool = True
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Implementation of eval_loop for StandardEvaluator
        """

        self.model.eval()
        eval_loss, eval_rmse_loss = 0.0, 0.0
        num_batches = len(loader)
        pbar = tqdm(loader) if use_tqdm else loader
        y_pred, y_true = [], []

        with torch.no_grad():
            for life, rul, mask in pbar:
                life = (
                    life.to(self.device)
                    if "padding" in self.config.approach
                    else life.to(self.device).squeeze(-1)
                )
                rul = rul.to(self.device).squeeze(-1)
                mask = (
                    mask.to(self.device)
                    if "padding" in self.config.approach
                    else mask.to(self.device).squeeze(-1)
                )

                self.model.tau = self.tau
                output = self.model(life)

                batch_out = output.to("cpu").detach().numpy()
                batch_target = rul.to("cpu").detach().numpy()
                y_pred.extend(batch_out)
                y_true.extend(batch_target)

                loss = self.criterion(output, rul, mask, self.tau)
                rmse_loss = self.eval_criterion(output, rul, mask, self.tau)
                eval_loss += loss.item()
                eval_rmse_loss += rmse_loss.item()

            eval_loss /= num_batches
            eval_rmse_loss /= num_batches
            print(
                f"Avg {mode} Loss: {eval_loss:.4f} | Avg {mode} eval Loss: {eval_rmse_loss:.4f}"
            )

        return eval_loss, eval_rmse_loss, np.array(y_pred), np.array(y_true)


def get_evaluator(
    mono_mask: np.ndarray = np.zeros(shape=(10, 1)),
    tau: float = 0.5,
    **kwargs
) -> Evaluator:
    """
    Get the correct Evaluator subclass for the current experiment

    Args:
        config (ExperimentConfig): experiment configuration object
        mono_mask (np.ndarray): boolean mask to identify monotonic features
        tau (float): The quantile level on which the model will be evaluated if the quantile regression approach is used

    Returns:
        trainer (Evaluator): trainer object
    """

    config = kwargs["config"]

    if config.quantile_reg:
        trainer = QuantileEvaluator(tau=tau, **kwargs)
    else:
        trainer = StandardEvaluator(**kwargs)

    return trainer
