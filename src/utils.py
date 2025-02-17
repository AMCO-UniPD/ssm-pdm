"""
Script containing some utility functions for the `chronos-pdm` project
"""

import os
import sys
import time
import yaml
import ipdb
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

chronos_path = os.path.join(os.path.dirname(__file__),"chronos-rul","src")
sys.path.append(chronos_path)

# chronos imports
from chronos import MeanScaleUniformBins, ChronosConfig

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
        sequence_length: int = 500,
    ):
        
        if sequence_length > life.shape[0]:
            pad_arr=np.zeros(sequence_length-life.shape[0])
            mask=[np.concatenate((np.ones(life.shape[0]),np.zeros(sequence_length-life.shape[0]))) for sensor in sensors]
            sequences=[np.concatenate((life[sensor].values,pad_arr)) for sensor in sensors]
            targets=[np.concatenate((life["RUL"].values,pad_arr)) for _ in sensors]
        else:
            sequences=[life[sensor].values[:sequence_length] for sensor in sensors]
            mask=[np.ones(sequence_length) for _ in sensors]
            targets=[life["RUL"].values[:sequence_length] for _ in sensors]

        self.sequences = sequences
        self.targets = targets
        self.mask = mask

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32).unsqueeze(-1)
        target = torch.tensor(self.targets[idx], dtype=torch.float32).unsqueeze(-1)
        mask = torch.tensor(self.mask[idx], dtype=torch.float32).unsqueeze(-1)
        return sequence, target, mask

class MeanScaleUniformBinsSensor(MeanScaleUniformBins):
    def __init__(self, low_limit:float, high_limit:float, config:ChronosConfig):
        super().__init__(low_limit, high_limit,config)
    
    def _input_transform(
        self, context: torch.Tensor, mask: torch.Tensor, scale: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = context.to(dtype=torch.float32)
        # attention_mask = ~torch.isnan(context)
        attention_mask = mask.to(torch.bool)

        if scale is None:
            scale = torch.nansum(
                torch.abs(context) * attention_mask, dim=1
            ) / torch.nansum(attention_mask, dim=1)
            scale[~(scale > 0)] = 1.0


        scaled_context = context.squeeze(-1) / scale
        token_ids = (
            torch.bucketize(
                input=scaled_context,
                boundaries=self.boundaries.to(scaled_context.device),
                # buckets are open to the right, see:
                # https://pytorch.org/docs/2.1/generated/torch.bucketize.html#torch-bucketize
                right=True,
            )
            + self.config.n_special_tokens
        )

        token_ids.clamp_(0, self.config.n_tokens - 1)

        token_ids[~attention_mask.squeeze(-1)] = self.config.pad_token_id

        return token_ids, attention_mask, scale

    def _append_eos_token(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = token_ids.shape[0]
        eos_tokens = torch.full((batch_size, 1), fill_value=self.config.eos_token_id).to(token_ids.device)
        token_ids = torch.concat((token_ids, eos_tokens), dim=1)
        eos_mask = torch.full((batch_size, 1), fill_value=True).to(attention_mask.device)
        attention_mask = torch.concat((attention_mask.squeeze(-1), eos_mask), dim=1)

        return token_ids, attention_mask

    def context_input_transform(
        self, context: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = context.shape[-1]

        if length > self.config.context_length:
            context = context[..., -self.config.context_length :]

        token_ids, attention_mask, scale = self._input_transform(context=context,mask=mask)

        if self.config.use_eos_token and self.config.model_type == "seq2seq":
            token_ids, attention_mask = self._append_eos_token(
                token_ids=token_ids, attention_mask=attention_mask
            )

        return token_ids, attention_mask, scale


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
    regression_datasets = [RegressionDataset(life=life,sensors=config.sensors,sequence_length=config.sequence_length) for life in lifes]

    # Create a DataLoader
    # reg_loaders=[DataLoader(regression_dataset,batch_size=len(config.sensors),shuffle=False) for regression_dataset in regression_datasets]

    reg_loader=DataLoader(ConcatDataset(regression_datasets),batch_size=len(config.sensors),shuffle=False)

    return df, regression_datasets, reg_loader

