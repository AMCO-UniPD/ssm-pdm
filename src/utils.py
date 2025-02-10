"""
Script containing some utility functions for the `chronos-pdm` project
"""

import os
import sys
import time
import yaml
import ipdb
import pandas as pd
from typing import Tuple, List
from dataclasses import dataclass

chronos_path = os.path.join(os.path.dirname(__file__),"chronos-forecasting","src")
sys.path.append(chronos_path)

from chronos import ChronosConfig, MeanScaleUniformBins
from transformers import AutoModelForSequenceClassification

# ceruleo imports
from ceruleo.dataset.ts_dataset import AbstractPDMDataset
from ceruleo.dataset.catalog.CMAPSS import CMAPSSDataset
from ceruleo.dataset.catalog.CMAPSS import sensor_indices
from ceruleo.transformation import Transformer
from ceruleo.transformation.features.selection import ByNameFeatureSelector, PandasVarianceThreshold
from ceruleo.transformation.functional.pipeline.pipeline import make_pipeline
from ceruleo.transformation.features.scalers import MinMaxScaler,RobustMinMaxScaler,StandardScaler,RobustStandardScaler

# torch imports
from torch.utils.data import DataLoader, ConcatDataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def get_current_time() -> str:
    """
    This function returns the current time in the format 'dd-mm-YYYY_HH-MM-SS'.
    It is used to produce the name of the files saved

    Returns:
        current_time: string representing the current time

    """

    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
    return current_time

def load_yaml_to_dict(file_path:str) -> dict:
    """
    Loads the contents of a YAML file into a Python dictionary.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: A dictionary containing the data from the YAML file, or None if an error occurs.
    """
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)  # Use safe_load to prevent arbitrary code execution
        return data
    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None

class TransData(AbstractPDMDataset):
    def __init__(self, data: AbstractPDMDataset):
        super().__init__()
        self.lives = []
        for life in data:
            current_life = pd.concat([life[0],life[1]],axis=1)
            self.lives.append(current_life)

    def get_time_series(self, i):
        return self.lives[i]

    @property
    def rul_column(self) -> str:
        return "RUL"

    @property
    def n_time_series(self):
        return len(self.lives)

class RegressionDataset(Dataset):
    def __init__(
        self,
        life: pd.DataFrame,
        sensors: List[str],
    ):
        sequences=[life[sensor].values for sensor in sensors]
        targets=life["RUL"].values

        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32).unsqueeze(-1)
        target = torch.tensor(self.targets, dtype=torch.float32)
        return sequence, target

@dataclass
class ExperimentConfig:

    def __init__(self,config:dict):
        for key in config:
            setattr(self,key,config[key])

def get_transformer(config:ExperimentConfig,df:CMAPSSDataset) -> Transformer:

    """
    Create a transformer object to preprocess the data from the CMAPSS dataset

    Args:
        config (ExperimentConfig): The configuration dictionary
        df (CMAPSSDataset): The CMAPSS dataset

    Returns:
        transformer (Transformer): The transformer object
    """

    FEATURES = [df[0].columns[i] for i in sensor_indices]

    if config.scaler == "minmax":
        scaler = MinMaxScaler(range=(config.scaler_kwargs["low_limit"], config.scaler_kwargs["high_limit"]))
    elif config.scaler == "standard":
        scaler = StandardScaler()


    transformer = Transformer(
        pipelineX=make_pipeline(
            ByNameFeatureSelector(features=FEATURES),
            scaler
        ),

        pipelineY=make_pipeline(
            ByNameFeatureSelector(features=['RUL']),
            )
    )

    return transformer

def load_reg_data(config:ExperimentConfig) -> Tuple[CMAPSSDataset, List[RegressionDataset], DataLoader]:
    """
    Load the data from a RUL dataset (e.g. CMAPSS,CMAPSS-2) and convert them
    into a DataLoader object with minibatched of size 1, each one containing a life

    Args:
        config (ExperimentConfig): The configuration dictionary

    Returns:
        df (CMAPSSDataset): The CMAPSS dataset
        regression_dataset (RegressionDataset): The RegressionDataset object
        reg_loader (DataLoader): The DataLoader object
    """

    if config.data_name == "CMAPSS":
        assert config.cmapss_models in ["FD001","FD002","FD003","FD004"], "The models must be one of FD001,FD002,FD003,FD004"
        df = CMAPSSDataset(train=config.train,models=config.cmapss_models)
    else:
        raise ValueError(f"Data name {config.data_name} not supported yet")

    # transform the data
    transformer = get_transformer(config,df)
    transformer.fit(df)
    transformed_df=df.map(transformer)

    # Create the TransData object
    lifes=TransData(transformed_df)

    # Create a RegressionDataset
    regression_datasets = [RegressionDataset(life=life,sensors=config.sensors) for life in lifes]

    # Create a DataLoader
    reg_loaders=[DataLoader(regression_dataset,batch_size=len(config.sensors),shuffle=False) for regression_dataset in regression_datasets]

    reg_loader=DataLoader(ConcatDataset(regression_datasets),batch_size=len(config.sensors),shuffle=False)

    return df, regression_datasets, reg_loader

def load_model_tokenizer(
        model_config:ChronosConfig,
        exp_config:ExperimentConfig) -> Tuple[torch.nn.Module, MeanScaleUniformBins]:
    """
    Load the model and the tokenizer for the `chronos` model

    Args:
        model_config (ChronosConfig): The configuration dictionary for the model
        exp_config (ExperimentConfig): The configuration dictionary for the experiment

    Returns:
        model (torch.nn.Module): The model object
        tokenizer (MeanScaleUniformBins): The Transformer object
    """

    model = AutoModelForSequenceClassification.from_pretrained(exp_config.model_id,num_labels=exp_config.num_labels)
    
    #NOTE: To change the classification head so that it returns the entire RUL sequence
    # We arrive to the classification head with a shape of (n_sensors,512)
    # (where 512 I think is the hidden size of the model)
    # Probably it's better to create a nn.Module for the new classification head
    # so that in the forward method I can also work with the shape of the input

    # new_classification_head = nn.Sequential(
    #     # Define here the correct layers (now I put some placeholders)
    #     nn.Linear(model.config.hidden_size, model.config.hidden_size),
    #     nn.Dropout(model.config.hidden_dropout_prob),
    #     nn.Linear(model.config.seq_length, model.config.seq_length),
    # )

    # new_classification_head = nn.Sequential(nn.Identity())

    # Substitute classification_head with the new one
    model.classification_head = new_classification_head

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

    tokenizer=MeanScaleUniformBins(
        low_limit=chronos_config.tokenizer_kwargs["low_limit"],
        high_limit=chronos_config.tokenizer_kwargs["high_limit"],
        config=chronos_config
    )

    return model, tokenizer
