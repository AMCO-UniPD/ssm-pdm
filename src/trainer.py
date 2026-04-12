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
        Method implemeting an entire training and evaluation run for multiple epochs
        logging the results to wandb
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


# NOTE: Standard Trainer → no quantile_reg

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

            ipdb.set_trace()

            output = self.model(life)
            loss = self.criterion(output, rul, mask)

            ipdb.set_trace()

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
        self.evaluator = QuantileEvaluator(tau=self.tau, **kwargs)

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

            #NOTE: In the new version of the models tau is a model attribute
            # so we have to set it to the sampled quantile level

            self.model.tau = tau
            output = self.model(life)
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

    if config.quantile_reg:
        trainer = QuantileTrainer(tau=tau, **kwargs)
    else:
        trainer = StandardTrainer(**kwargs)

    return trainer
