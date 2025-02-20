"""
Python script containing utility functions for the models of the `chronos-pdm` project
"""

# general imports
import os
import sys
from gluonts import model
import ipdb
import traceback
import time
import wandb
import numpy as np
from tqdm.auto import tqdm
from typing import Tuple

# torch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
# from apex.optimizers import FusedAdam
from torch.optim import AdamW, lr_scheduler

chronos_path_src = os.path.join(os.path.dirname(__file__),"chronos-rul","src")
chronos_path_scripts = os.path.join(os.path.dirname(__file__),"chronos-rul","scripts")
imports_path = os.path.join(os.path.dirname(__file__),"AD_MG","src")
sys.path.append(chronos_path_src)
sys.path.append(chronos_path_scripts)
sys.path.append(imports_path)

# import from other modules
from utils import(
    save_element,
    ExperimentConfig,
    generate_path,
    load_reg_data,
    get_most_recent_file,
    open_element,
)

from loss import load_loss_functions

from perf import lifes_metrics, df_with_index_to_obsidian_table

from plots import plot_predictions_grid

# chronos imports
from training.train import load_model
from chronos import ChronosModel, ChronosConfig, MeanScaleUniformBins
from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from transformers import AutoConfig, T5Config
from utils import ExperimentConfig, MeanScaleUniformBinsSensor, get_current_time

cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
experiment_path = os.path.join(cwd, "experiments", "chronos_exp")

class RegressionHead(nn.Module):
    def __init__(self, sequence_length:int, hidden_size:int):
        super(RegressionHead, self).__init__()

        self.fc = nn.Linear(hidden_size, sequence_length)
 
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.fc(x) # (n_sensors,hidden_size) -> (n_sensors,sequence_length)
        return x

def load_model_tokenizer(
        train_loader:DataLoader,
        model_config:ChronosConfig,
        exp_config:ExperimentConfig) -> Tuple[nn.Module, MeanScaleUniformBins, optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    """
    Load the model and the tokenizer for the `chronos` model

    Args:
        train_loader (DataLoader): The DataLoader object for training, needed for the definition of the lr scheduler
        model_config (ChronosConfig): The configuration dictionary for the model
        exp_config (ExperimentConfig): The configuration dictionary for the experiment

    Returns:
        model (torch.nn.Module): The model object
        tokenizer (MeanScaleUniformBins): The tokenizer object
        optimizer (torch.optim.Optimizer): The optimizer object
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler object
    """

    # Create the ChronosConfig object
    chronos_config=ChronosConfig(
        tokenizer_class=model_config["tokenizer_class"],
        tokenizer_kwargs=model_config["tokenizer_kwargs"],
        n_tokens=model_config["n_tokens"],
        n_special_tokens=model_config["n_special_tokens"],
        pad_token_id=model_config["pad_token_id"],
        eos_token_id=model_config["eos_token_id"],
        use_eos_token=model_config["use_eos_token"],
        model_type=model_config["model_type"],
        context_length=model_config["context_length"],
        prediction_length=model_config["prediction_length"],
        num_samples=model_config["num_samples"],
        temperature=model_config["temperature"],
        top_k=model_config["top_k"],
        top_p=model_config["top_p"],
    )

    tokenizer=MeanScaleUniformBinsSensor(
        low_limit=chronos_config.tokenizer_kwargs["low_limit"],
        high_limit=chronos_config.tokenizer_kwargs["high_limit"],
        config=chronos_config
    )

    if model_config["random_init"]:
        random_conf = AutoConfig.from_pretrained(exp_config.model_id)
        if isinstance(random_conf,T5Config):
            random_conf.initializer_factor = 0.05
        random_conf.tie_word_embeddings = True
        model = AutoModelForSequenceClassification.from_config(random_conf)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(exp_config.model_id)
    
    # The classification head can also be defined with a nn.Module
    new_classification_head = RegressionHead(sequence_length=exp_config.sequence_length, hidden_size=model.config.hidden_size)

    # Substitute classification_head with the new one
    model.classification_head = new_classification_head

    optimizer = AdamW(model.parameters(), lr=exp_config.lr)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=exp_config.epochs*len(train_loader))

    return model, tokenizer, optimizer, scheduler

# Train loop for one epoch

def train_loop(
        dataloader: DataLoader,
        model: nn.Module,
        tokenizer: MeanScaleUniformBinsSensor,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device = torch.device("cpu"),
) -> float:
    """
    Train loop for one epoch

    Args:
        dataloader (DataLoader): The DataLoader object
        model (torch.nn.Module): The model object
        optimizer (torch.optim.Optimizer): The optimizer object
        criterion (torch.nn.Module): The loss function
        device (str): The device to use

    Returns:
        loss (float): The loss value
    """

    model.train()
    train_loss = 0.0
    num_batches = len(dataloader)
    pbar=tqdm(enumerate(dataloader))

    for batch_idx, (life, rul, mask) in pbar:
        life = life.to(device)
        rul = rul.to(device).squeeze(-1)
        mask = mask.to(device)

        input_ids, attention_mask, _ = tokenizer.context_input_transform(context=life, mask=mask)
        output = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = criterion(output.logits, rul, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        pbar.set_description(f"Batch Idx: {batch_idx}/{len(dataloader)} | Train Loss: {train_loss / (batch_idx + 1):.4f}")

    return train_loss / num_batches

# Evaluation loop (i.e. validation and test) for one epoch

def eval_loop(
        dataloader: DataLoader,
        model: nn.Module,
        tokenizer: MeanScaleUniformBinsSensor,
        criterion: nn.Module,
        eval_criterion: nn.Module,
        mode: str = "Test",
        device: torch.device = torch.device("cpu"),
        use_tqdm: bool = True,
) -> Tuple[float,float,np.ndarray,np.ndarray]:
    """
    Evaluation loop for one epoch

    Args:
        dataloader (DataLoader): The DataLoader object
        model (torch.nn.Module): The model object
        tokenizer (MeanScaleUniformBinsSensor): The tokenizer object
        criterion (torch.nn.Module): The loss function
        eval_criterion (torch.nn.Module): The evaluation loss function
        mode (str): The mode of evaluation
        device (str): The device to use
        use_tqdm (bool): Whether to use tqdm or not

    Returns:
        loss (float): The loss value
    """

    model.eval()
    eval_loss, eval_rmse_loss = 0.0, 0.0
    num_batches = len(dataloader)
    pbar = tqdm(dataloader) if use_tqdm else dataloader
    y_pred,y_true = [],[]

    with torch.no_grad():
        for life, rul, mask in pbar:
            life = life.to(device)
            rul = rul.to(device).squeeze(-1)
            mask = mask.to(device)

            input_ids, attention_mask, _ = tokenizer.context_input_transform(context=life, mask=mask)
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_out = output.logits.to("cpu").detach().numpy()
            batch_target = rul.to("cpu").detach().numpy()
            y_pred.append(batch_out)
            y_true.append(batch_target)

            loss = criterion(output.logits, rul, mask)
            rmse_loss = eval_criterion(output.logits, rul, mask)
            eval_loss += loss.item()
            eval_rmse_loss += rmse_loss.item()

        eval_loss/=num_batches
        eval_rmse_loss/=num_batches
        print(f"Avg {mode} Loss: {eval_loss:.4f} | \
                Avg {mode} eval Loss: {eval_rmse_loss:.4f}")

    return eval_loss, eval_rmse_loss, np.array(y_pred), np.array(y_true)

# Save the best model

def save_best_model(
    best_model_state_dict: dict,
    config: ExperimentConfig,
    best_model_path: str = experiment_path,
) -> None:
    """
    This function saves the best model into best_model_path

    Args:
        best_model_state_dict (dict): The state dictionary of the best model
        config (ExperimentConfig): The configuration object

    Returns:
        The function saves the best model and does not return anything
    """

    best_model_path = generate_path(basepath=best_model_path,
                                    folders=["best_models",
                                             config.model_name,
                                             config.cmapss_models])


    save_element(
        element=best_model_state_dict,
        dirpath=best_model_path,
        filename=f"{get_current_time()}_best_model",
        filetype="pickle",
    )
    print("#" * 50)
    print(f"Best model saved at: {best_model_path}")
    print("#" * 50)

# Function to train and test the model on a wandb run

def wandb_train_test(
        model:nn.Module,
        tokenizer:MeanScaleUniformBinsSensor,
        train_loader:DataLoader,
        val_loader:DataLoader,
        test_loader:DataLoader,
        criterion:nn.Module,
        eval_criterion:nn.Module,
        optimizer:optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        config: ExperimentConfig,
        device:torch.device=torch.device("cpu"),
        best_model_path:str = experiment_path,
) -> dict:
    """
    Train and test the model on a wandb run and log the metrics

    Args:
        model (nn.Module): The model object
        tokenizer (MeanScaleUniformBinsSensor): The tokenizer object
        train_loader (DataLoader): The DataLoader object for training
        val_loader (DataLoader): The DataLoader object for validation
        test_loader (DataLoader): The DataLoader object for testing
        criterion (nn.Module): The loss function
        optimizer (optim.Optimizer): The optimizer object
        scheduler (optim.lr_scheduler._LRScheduler): The scheduler object
        exp_config (ExperimentConfig): The configuration object
        device (str): The device to use
        best_model_path (str): The path to save the best model

    Returns:
        model_info (dict): The dictionary containing the model information
    """

    wandb.watch(model, criterion, log="all", log_freq=10)
    preds, true_vals, train_times, val_times, test_times = [], [], [], [], []
    min_val_loss = np.inf
    best_model_state_dict = model.state_dict()
    pbar = tqdm(range(config.epochs))

    try:
        for epoch in pbar:
            if epoch == 0:
                pbar.set_description("Epoch: %d" % (epoch))
                val_loss,test_loss=0.0,0.0
            else:
                pbar.set_description(f"Epoch: {epoch:d} | Val loss: {val_loss:1.3f} | Eval Val loss: {eval_val_loss:1.3f}")
                pbar.set_description(f"Epoch: {epoch:d} | Test loss: {test_loss:1.3f} | Eval Test loss: {eval_test_loss:1.3f}")

            train_time = time.time()
            train_loss = train_loop(
                dataloader=train_loader,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )
            train_time = time.time() - train_time

            val_time = time.time()
            val_loss,eval_val_loss,y_pred,y_true = eval_loop(
                dataloader=val_loader,
                model=model,
                tokenizer=tokenizer,
                criterion=criterion,
                eval_criterion=eval_criterion,
                mode="Val",
                device=device,
            )
            val_time = time.time() - val_time

            test_time = time.time()
            test_loss,eval_test_loss, y_pred, y_true = eval_loop(
                dataloader=test_loader,
                model=model,
                tokenizer=tokenizer,
                criterion=criterion,
                eval_criterion=eval_criterion,
                mode="Test",
                device=device,
            )
            test_time = time.time() - test_time

            if scheduler is not None:
                scheduler.step()
                print(f"Epoch {epoch} learning rate: {scheduler.get_last_lr()}")

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                print(
                    f"Epoch {epoch} | New best model found with val loss: {min_val_loss}"
                )
                print("#" * 50)
                best_model_state_dict = model.state_dict().copy()

            # Save the predictions and true values only for the last epoch
            if epoch == config.epochs - 1:
                preds.append(y_pred)
                true_vals.append(y_true)

            train_times.append(train_time)
            val_times.append(val_time)
            test_times.append(test_time)

            model_info = {
                "preds": preds,
                "true_vals": true_vals,
                "train_times": train_times,
                "val_times": val_times,
                "test_times": test_times,
            }

            wandb.log(
                {
                    "train_time": train_time,
                    "val_time": val_time,
                    "test_time": test_time,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "eval_val_loss": eval_val_loss,
                    "test_loss": test_loss,
                    "eval_test_loss": eval_test_loss,
                }
            )

    except KeyboardInterrupt:
        print("Manual Early Stopping triggered. Saving the best model up to now")

        if config.save_best_model:
            save_best_model(
                best_model_state_dict=best_model_state_dict,
                config=config,
                best_mdoel_path=best_model_path,
            )

    except torch.cuda.OutOfMemoryError:
        print("CUDA Out of Memory Error")
        traceback.print_exc()  # Print the full traceback of the error
        quit()

    except Exception as e:
        print("An error occured during the training process:")
        print(e)
        traceback.print_exc()  # Print the full traceback of the error

    print("No errors occured during the training process, saving the best model")
    save_best_model(
        best_model_state_dict=best_model_state_dict,
        config=config,
    )

    return model_info

# Function to get the best model performance

def best_model_perf(
    config: ExperimentConfig,
    model_config: ChronosConfig,
    device: torch.device = torch.device("cpu"),
    best_model_path: str = experiment_path,
    outputs_path: str = experiment_path,
    ) -> None:
    """
    This function loads the best model according to the validation set and
    computes the performance on the test set
    
    Args:
        config (ExperimentConfig): The configuration dictionary
        model_config (ChronosConfig): The model configuration object
        device (str): The device to use
        best_model_path (str): The path to save the best model
        outputs_path (str): The path to save the outputs
        plot_path (str): The path to save the plots
        metrics_path (str): The path to save the test metrics

    Returns:
        The function saves the plots and the metrics and does not return anything
    """

    # Load the best model

    if config.zero_shot:

        print("#"*50)
        print("Zero-shot inference")
        print("#"*50)

        train_loader,_,test_loader=load_reg_data(config)

        model,tokenizer,_,_ = load_model_tokenizer(train_loader=train_loader,
                                                   model_config=model_config,
                                                   exp_config=config)
        model=model.to(device)

    else:

        best_model_filepath = get_most_recent_file(dirpath=best_model_path,
                                                   file_pos=config.file_pos)

        best_model_state_dict = open_element(best_model_filepath,
                                             filetype="pickle")
        
        train_loader,_,test_loader=load_reg_data(config)

        model,tokenizer,_,_ = load_model_tokenizer(train_loader=train_loader,
                                                   model_config=model_config,
                                                   exp_config=config)

        model.load_state_dict(best_model_state_dict)
        model=model.to(device)

    criterion,eval_criterion=load_loss_functions(
        loss_name=config.loss,
        eval_loss_name=config.eval_loss,
        tau=config.tau
    )

    # Evaluate the model on the test set
    print("#" * 50)
    print("Evaluating the best model on the test set")
    print("#" * 50)
    _,_,y_pred,y_true = eval_loop(
        dataloader=test_loader,
        model=model,
        tokenizer=tokenizer,
        criterion=criterion,
        eval_criterion=eval_criterion,
        mode="Test",
        device=device,
        use_tqdm=False,
    )
    print("#" * 50)
    print(f"y_pred shape: {y_pred.shape} | y_true shape: {y_true.shape}")

    # Save the predictions and true values
    outputs_dict = {
        "y_pred": y_pred,
        "y_true": y_true,
    }

    save_element(
        element=outputs_dict,
        dirpath=outputs_path,
        filename=f"{get_current_time()}_outputs_{config.model_name}_{config.cmapss_models}",
        filetype="pickle",
    )


# Function that implements a wandb run

def wandb_run(
    run_name:str,
    config:ExperimentConfig,
    model_config:ChronosConfig,
    device:torch.device=torch.device("cpu"),
    best_model_path:str=experiment_path,
    outputs_path:str=experiment_path,
    metrics_path:str=experiment_path,
    plot_path:str=experiment_path,
) -> Tuple[nn.Module, dict]:
    """
    Function that implements a wandb run

    Args:
        run_name (str): The name of the run
        config (ExperimentConfig): The experiment configuration object
        model_config (ChronosConfig): The model configuration object
        device (str): The device to use
        best_model_path (str): The path to save the best model
        outputs_path (str): The path to save the outputs
        metrics_path (str): The path to save the metrics
        plot_path (str): The path to save the plots

    Returns:
        model (nn.Module): The model object
        model_info (dict): The dictionary containing the model information
    """

    with wandb.init(project=config.project_name, name=run_name):

        train_loader,val_loader,test_loader=load_reg_data(config)

        model,tokenizer,optimizer,scheduler=load_model_tokenizer(
            train_loader=train_loader,
            model_config=model_config,
            exp_config=config,
        )
        model=model.to(device)

        criterion,eval_criterion=load_loss_functions(
            loss_name=config.loss,
            eval_loss_name=config.eval_loss,
            tau=config.tau
        )
        model_info = wandb_train_test(
            model=model,
            tokenizer=tokenizer,
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
        )

    if config.save_outputs:

        print("#"*50)
        print("Saving outputs")
        print("#"*50)

        best_model_perf(
            config=config,
            model_config=model_config,
            device=device,
            best_model_path=best_model_path,
            outputs_path=outputs_path
        )

    if config.compute_metrics:

        print("#" * 50)
        print("Computing metrics for each life and for each sensor in the test set")
        print("#" * 50)

        metrics_df = lifes_metrics(
            config=config,
            outputs_path=outputs_path,
            metrics_path=metrics_path,
        )
        print("#" * 50)
        print(f"metrics_df shape: {metrics_df.shape}")

    if config.obsidian_table:

       print("#" * 50)
       print("Producing the obsidian table")
       print("#" * 50)

       metrics_path = get_most_recent_file(metrics_path, file_pos=config.file_pos)
       metrics_df = open_element(metrics_path)
       print(df_with_index_to_obsidian_table(metrics_df))

    if config.plot_preds:

        print("#" * 50)
        print("Producing grid plot of the predictions")
        print("#" * 50)

        _ = plot_predictions_grid(
            config=config,
            sensor_idx=config.sensor_idx,
            outputs_path=outputs_path,
            plot_path=plot_path,
        )


    return model,model_info
