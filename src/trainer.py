"""
Python module containing all the trainer classes
"""

import copy
import gc
import os
import time
import traceback
from typing import Tuple, Union

import ipdb
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from evaluator import(
    Evaluator,
    QuantileEvaluator,
    MonotonicEvaluator,
    MonoQuantileEvaluator
)

from utils import(
    ExperimentConfig,
    get_current_time,
    sample_quantile,
    save_element,
    split_input
)

cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
experiment_path = os.path.join(cwd, "experiments", "phm_exp")


class Trainer:
    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        model: nn.Module,
        config: ExperimentConfig,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        eval_criterion: nn.Module,
        scheduler: optim.lr_scheduler._LRScheduler,
        device: torch.device = torch.device("cpu"),
        best_model_path: str = experiment_path,
    ):
        """
        Trainer base class
        """

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.criterion = criterion
        self.eval_criterion = eval_criterion
        self.scheduler = scheduler
        self.device = device
        self.best_model_path = best_model_path

        self.evaluator = Evaluator(
            model = self.model,
            config = self.config,
            criterion = self.criterion,
            eval_criterion = self.eval_criterion,
            device = self.device
        )

    def train_loop(self):
        """
        Method implementing a training loop on a single epoch
        """
        raise NotImplementedError

    def save_best_model(
        self,
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

    def run(self):
        """
        Method implemeting an entire training and evaluation run for multiple epochs.
        """

        if self.config.use_wandb:
            wandb.watch(self.model, self.criterion, log="all", log_freq=10)
            wandb.define_metric("epoch")
            wandb.define_metric("loss/*", step_metric="epoch")
            wandb.define_metric("times/*", step_metric="epoch")
            wandb.define_metric("eval_loss/*", step_metric="epoch")

        error = False
        train_times, val_times, test_times = [], [], []
        min_val_loss = np.inf
        best_model_state_dict = self.model.state_dict()
        pbar = tqdm(range(self.config.epochs))
        val_loss, test_loss = 0.0, 0.0
        eval_val_loss, eval_test_loss = 0.0, 0.0

        try:
            for epoch in pbar:
                if epoch == 0:
                    pbar.set_description("Epoch: %d" % (epoch))
                else:
                    pbar.set_description(
                        f"Epoch: {epoch:d} | Val loss: {val_loss:1.3f} | Eval Val loss: {eval_val_loss:1.3f}"
                    )
                    pbar.set_description(
                        f"Epoch: {epoch:d} | Test loss: {test_loss:1.3f} | Eval Test loss: {eval_test_loss:1.3f}"
                    )

                train_time = time.time()
                train_loss = self.train_loop()
                train_time = time.time() - train_time

                val_time = time.time()
                val_loss, eval_val_loss, _, _ = self.evaluator.eval_loop(
                    loader=self.val_loader, mode="Val"
                )
                val_time = time.time() - val_time

                test_time = time.time()
                test_loss, eval_test_loss, _, _ = self.evaluator.eval_loop(
                    loader=self.test_loader, mode="Test"
                )
                test_time = time.time() - test_time

                if self.scheduler is not None:
                    self.scheduler.step()
                    print(
                        f"Epoch {epoch} learning rate: {self.scheduler.get_last_lr()}"
                    )

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    print(
                        f"Epoch {epoch} | New best model found with val loss: {min_val_loss}"
                    )
                    print("#" * 50)
                    best_model_state_dict = copy.deepcopy(self.model.state_dict())

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

                if self.config.use_wandb:
                    wandb.log(model_info)
                else:
                    model_info_df = pd.DataFrame(model_info, index=["values"])
                    print("-" * 50)
                    print("Information on the model training and evaluation:")
                    print(model_info_df.T)
                    print("-" * 50)

        except KeyboardInterrupt:
            print("-" * 50)
            print("Manual Early Stopping triggered. Saving the best model up to now")
            print("-" * 50)

            self.save_best_model(
                best_model_state_dict=best_model_state_dict,
                best_model_path=self.best_model_path,
            )

        except torch.cuda.OutOfMemoryError:
            print("-" * 50)
            print("CUDA Out of Memory Error, stopping execution")
            print("-" * 50)
            traceback.print_exc()  # Print the full traceback of the error
            error = True
            quit()

        except Exception as e:
            print("-" * 50)
            print("An error occured during the training process:")
            print("-" * 50)
            print(e)
            traceback.print_exc()  # Print the full traceback of the error
            error = True

        if not error:
            print(
                "No errors occured during the training process, saving the best model"
            )
            self.save_best_model(
                best_model_state_dict=best_model_state_dict,
                best_model_path=self.best_model_path,
            )


# NOTE: Standard Trainer → no quantile_reg and no monotonic

class StandardTrainer(Trainer):
    """
    Standard Trainer class without any fancy approach
    """

    def train_loop(self) -> float:
        """
        train_loop implementation for StandardTrainer
        """

        self.model.train()
        train_loss = 0.0
        num_batches = len(self.train_loader)
        pbar = tqdm(enumerate(self.train_loader))

        for batch_idx, (life, rul, mask) in pbar:

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
            loss = self.criterion(output, rul, mask)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            torch.cuda.empty_cache()
            gc.collect()

            train_loss += loss.item()

            pbar.set_description(
                f"Batch Idx: {batch_idx}/{len(self.train_loader)} | Train Loss: {train_loss / (batch_idx + 1):.4f}"
            )

        return train_loss / num_batches

# NOTE: QuantileTrainer → quantile_reg and no monotonic


class QuantileTrainer(Trainer):
    """
    QuantileTrainer subclasses used to train the models using Quantile
    Regression

    Args:
        tau (float): The quantile level on which the model will be evaluated if the quantile regression approach is used
    """

    def __init__(self, tau: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tau = tau
        self.evaluator = QuantileEvaluator(tau=self.tau,**kwargs)

    def train_loop(self) -> float:
        """
        train_loop implementation for QuantileTrainer
        """

        self.model.train()
        train_loss = 0.0
        num_batches = len(self.train_loader)
        pbar = tqdm(enumerate(self.train_loader))

        for batch_idx, (life, rul, mask) in pbar:

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

            tau = sample_quantile(
                quantile_dist=self.config.quantile_dist,
                bounds=self.config.bounds,
                print_quantile=True,
            )

            output = self.model(life, tau=tau)
            loss = self.criterion(output, rul, mask, tau)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            torch.cuda.empty_cache()
            gc.collect()

            train_loss += loss.item()

            pbar.set_description(
                f"Batch Idx: {batch_idx}/{len(self.train_loader)} | Train Loss: {train_loss / (batch_idx + 1):.4f}"
            )

        return train_loss / num_batches

    def eval_loop(
        self, loader: torch.Tensor, mode: str = "Val", use_tqdm: bool = True
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Implementation of eval_loop for QuantileTrainer
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

                output = self.model(life, tau=self.tau)

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


# NOTE: MonotonicTrainer → no quantile_reg and monotonic

class MonotonicTrainer(Trainer):
    """
    MonotonicTrainer subclasses used to train the models using Partially Monotonic NN

    Args:
        mono_mask (np.ndarray): boolean mask to identify monotonic features
    """

    def __init__(self, mono_mask: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mono_mask = mono_mask
        self.evaluator = MonotonicEvaluator(mono_mask=self.mono_mask,**kwargs)

    def train_loop(self) -> float:
        """
        train_loop implementation for MonotonicTrainer

        Args:
            mono_mask (np.ndarray): boolean mask to identify monotonic features
        """

        self.model.train()
        train_loss = 0.0
        num_batches = len(self.train_loader)
        pbar = tqdm(enumerate(self.train_loader))

        for batch_idx, (life, rul, mask) in pbar:

            life, life_mono = split_input(mask_mono=self.mono_mask, inputs=life, device=self.device)

            life = (
                life.to(self.device)
                if "padding" in self.config.approach
                else life.to(self.device).squeeze(-1)
            )
            life_mono = (
                life_mono.to(self.device)
                if "padding" in self.config.approach
                else life_mono.to(self.device).squeeze(-1)
            )
            rul = rul.to(self.device).squeeze(-1)
            mask = (
                mask.to(self.device)
                if "padding" in self.config.approach
                else mask.to(self.device).squeeze(-1)
            )

            output = self.model(life,life_mono)
            loss = self.criterion(output, rul, mask)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            torch.cuda.empty_cache()
            gc.collect()

            train_loss += loss.item()

            pbar.set_description(
                f"Batch Idx: {batch_idx}/{len(self.train_loader)} | Train Loss: {train_loss / (batch_idx + 1):.4f}"
            )

        return train_loss / num_batches

    def eval_loop(
        self, loader: torch.Tensor, mode: str = "Val", use_tqdm: bool = True
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Implementation of eval_loop for MonotonicTrainer
        """

        self.model.eval()
        eval_loss, eval_rmse_loss = 0.0, 0.0
        num_batches = len(loader)
        pbar = tqdm(loader) if use_tqdm else loader
        y_pred, y_true = [], []

        with torch.no_grad():
            for life, rul, mask in pbar:
                life, life_mono = split_input(mask_mono=self.mono_mask, inputs=life, device=self.device)

                life = (
                    life.to(self.device)
                    if "padding" in self.config.approach
                    else life.to(self.device).squeeze(-1)
                )
                life_mono = (
                    life_mono.to(self.device)
                    if "padding" in self.config.approach
                    else life_mono.to(self.device).squeeze(-1)
                )
                rul = rul.to(self.device).squeeze(-1)
                mask = (
                    mask.to(self.device)
                    if "padding" in self.config.approach
                    else mask.to(self.device).squeeze(-1)
                )

                output = self.model(life,life_mono)

                batch_out = output.to("cpu").detach().numpy()
                batch_target = rul.to("cpu").detach().numpy()
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

# NOTE: MonoQuantileTrainer → quantile_reg and monotonic

class MonoQuantileTrainer(Trainer):
    """
    MonoQuantileTrainer subclasses used to train the models using Quantile
    Regression and Partially Monotonic NN

    Args:
        tau (float): The quantile level on which the model will be evaluated if the quantile regression approach is used
        mono_mask (np.ndarray): boolean mask to identify monotonic features
    """

    def __init__(self, mono_mask: np.ndarray, tau: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau
        self.mono_mask = mono_mask
        self.evaluator = MonoQuantileEvaluator(tau=self.tau, mono_mask=self.mono_mask,**kwargs)

    def train_loop(self) -> float:
        """
        train_loop implementation for MonoQuantileTrainer

        Args:
            mono_mask (np.ndarray): boolean mask to identify monotonic features
        """

        self.model.train()
        train_loss = 0.0
        num_batches = len(self.train_loader)
        pbar = tqdm(enumerate(self.train_loader))

        for batch_idx, (life, rul, mask) in pbar:

            life, life_mono = split_input(mask_mono=self.mono_mask, inputs=life, device=self.device)

            life = (
                life.to(self.device)
                if "padding" in self.config.approach
                else life.to(self.device).squeeze(-1)
            )
            life_mono = (
                life_mono.to(self.device)
                if "padding" in self.config.approach
                else life_mono.to(self.device).squeeze(-1)
            )
            rul = rul.to(self.device).squeeze(-1)
            mask = (
                mask.to(self.device)
                if "padding" in self.config.approach
                else mask.to(self.device).squeeze(-1)
            )

            tau = sample_quantile(
                quantile_dist=self.config.quantile_dist,
                bounds=self.config.bounds,
                print_quantile=True,
            )

            output = self.model(life, tau=tau)
            loss = self.criterion(output, rul, mask, tau)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            torch.cuda.empty_cache()
            gc.collect()

            train_loss += loss.item()

            pbar.set_description(
                f"Batch Idx: {batch_idx}/{len(self.train_loader)} | Train Loss: {train_loss / (batch_idx + 1):.4f}"
            )

        return train_loss / num_batches

    def eval_loop(
        self, loader: torch.Tensor, mode: str = "Val", use_tqdm: bool = True
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Implementation of eval_loop for MonoQuantileTrainer
        """

        self.model.eval()
        eval_loss, eval_rmse_loss = 0.0, 0.0
        num_batches = len(loader)
        pbar = tqdm(loader) if use_tqdm else loader
        y_pred, y_true = [], []

        with torch.no_grad():
            for life, rul, mask in pbar:
                life, life_mono = split_input(mask_mono=self.mono_mask, inputs=life, device=self.device)

                life = (
                    life.to(self.device)
                    if "padding" in self.config.approach
                    else life.to(self.device).squeeze(-1)
                )
                life_mono = (
                    life_mono.to(self.device)
                    if "padding" in self.config.approach
                    else life_mono.to(self.device).squeeze(-1)
                )
                rul = rul.to(self.device).squeeze(-1)
                mask = (
                    mask.to(self.device)
                    if "padding" in self.config.approach
                    else mask.to(self.device).squeeze(-1)
                )

                output = self.model(life, tau=self.tau)

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

def get_trainer(
    mono_mask: np.ndarray = np.zeros(shape=(10,1)),
    tau: float = 0.5,
    **kwargs
) -> Trainer:
    """
    Get the correct Trainer subclass for the current experiment

    Args:
        config (ExperimentConfig): experiment configuration object
        mono_mask (np.ndarray): boolean mask to identify monotonic features
        tau (float): The quantile level on which the model will be evaluated if the quantile regression approach is used

    Returns:
        trainer (Trainer): trainer object
    """

    config = kwargs["config"]

    if config.quantile_reg and config.monotonic:
        trainer = MonoQuantileTrainer(tau=tau, mono_mask=mono_mask, **kwargs)
    elif not config.quantile_reg and config.monotonic:
        trainer = MonotonicTrainer(mono_mask=mono_mask, **kwargs)
    elif config.quantile_reg and not config.monotonic:
        trainer = QuantileTrainer(tau=tau, **kwargs)
    else:
        trainer = StandardTrainer(**kwargs)

    return trainer
