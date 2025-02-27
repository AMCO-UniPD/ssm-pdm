"""
Script containing some utility functions for the `chronos-pdm` project
"""

import os
import sys
import time
from gluonts.transform import feature
import yaml
import pickle
import ipdb
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Union
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
from ceruleo.transformation.features.imputers import MeanImputer

# sklearn imports
from sklearn.model_selection import train_test_split

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

def get_most_recent_file(dirpath: str, file_pos: int = 0) -> str:
    """
    This function returns the most recent file in a directory

    Args:
        dirpath: path of the directory
        file_pos: position of the file in the list of files in the directory sorted in order of creation time, default=0
        (i.e. the most recent file is at position 0)

    Returns:
        The most recent file in the directory
    """

    assert os.path.isdir(dirpath), "The provided path  is not a directory"

    files = [f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))]
    paths = [os.path.join(dirpath, basename) for basename in files]
    sorted_paths = sorted(paths, key=os.path.getmtime)[::-1]
    return sorted_paths[file_pos]

def open_element(
    file_path: str, filetype: str = "pickle"
) -> Union[np.ndarray, list, pd.DataFrame]:
    """
    Function to open an element from a file (i.e. `npz` or `pickle` file) in the specified directory path.

    Args:
        file_path: Path to the file
        filetype: Type of the file (i.e. `npz` or `pickle`)

    Returns:
        Element stored in the file
    """

    assert filetype in ["pickle", "pth"], "filetype must be either 'pickle' or 'pth'"
    if filetype == "pickle":
        with open(file_path, "rb") as fl:
            element = pickle.load(fl)
    elif filetype == "pth":
        element = torch.load(file_path)
    else:
        raise ValueError("Invalid filetype. Please choose either 'pickle' or 'pth'")
    return element

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

def save_element(
    element: Union[dict, nn.Module, pd.DataFrame],
    dirpath: str,
    filename: str = "",
    filetype: str = "pickle",
    no_time: bool = False,
) -> None:
    """
    This function saves the model to the specified path

    Args:
        model: PyTorch model
        dirpath: path of the directory where to save the model
        filename: name of the file to save the model, default=''
        filetype: type of the file to save the model, default='pickle'
        no_time: boolean to indicate whether to include the current time in the filename, default=False

    Returns:
        The method saves the model and does not return any value
    """

    if no_time:
        path = os.path.join(dirpath, filename)
    else:
        current_time = get_current_time()
        filename = current_time + "_" + filename
        path = os.path.join(dirpath, filename)

    if filetype == "pth":
        torch.save(element.state_dict(), path + ".pth")
    elif filetype == "pickle":
        with open(path + ".pickle", "wb") as f:
            pickle.dump(element, f)

    print(f"Element successfully saved at path: {path}")

def generate_path(basepath:str = os.getcwd(),
                  folders:List[str] = []) -> str:
    """
    Generate a path starting from a basepath and a list of folders to join to the basepath.

    Args:
        basepath: The basepath from which to start to generate the path, by default os.getcwd()
        folders: A list of strings containing the ordered list of subfolders to join to the basepath, by default []
    Returns:
        path: The path generated by joining the basepath and the folders
    """

    # Verify weather basepath is a valid path in the system
    assert os.path.exists(basepath), f"Basepath {basepath} does not exist"

    path=basepath+"/"

    # Join the basepath with the folders
    for folder in folders:
        path=os.path.join(path, folder) + "/"
        # Verify weather the path exists or not
        if not os.path.exists(path):
            os.makedirs(path)

    return path[:-1]

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

# Regression dataset for chronos

class RegressionDataset(Dataset):
    def __init__(
        self,
        life: pd.DataFrame,
        sensors: List[str],
        sequence_length: int = 500,
    ):
        if life.shape[0] < sequence_length:
            pad_arr=np.zeros(sequence_length-life.shape[0])
            mask=[np.concatenate((np.ones(life.shape[0]),np.zeros(sequence_length-life.shape[0]))) for _ in sensors]
            sequences=[np.concatenate((life[sensor].values,pad_arr)) for sensor in sensors]
            targets=[np.concatenate((life["RUL"].values,pad_arr)) for _ in sensors]
        else:
            print("*"*50)
            print(f"Warning: This life is longer than {sequence_length}, removing the first {life.shape[0]-sequence_length} timesteps")
            print("*"*50)
            sequences=[life[sensor].values[life.shape[0]-sequence_length:] for sensor in sensors]
            mask=[np.ones(sequence_length) for _ in sensors]
            targets=[life["RUL"].values[life.shape[0]-sequence_length:] for _ in sensors]

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

# Regression dataset class for SMM models

class SSMRegressionDataset(Dataset):
    def __init__(
        self,
        life: pd.DataFrame,
        sequence_length: int = 500,
    ):
        
        if sequence_length > life.shape[0]:
            life,rul = life.iloc[:,:-1],life["RUL"]
            pad_arr=np.zeros(shape=(sequence_length-life.shape[0],life.shape[1]))
            mask=np.concatenate((np.ones(shape=(life.shape[0])),np.zeros(shape=(sequence_length-life.shape[0]))))
            sequences=np.concatenate((life.values,pad_arr))
            targets=np.concatenate((rul.values,pad_arr[:,-1]))
        else:
            print("*"*50)
            print(f"Warning: This life is longer than {sequence_length}, removing the first {life.shape[0]-sequence_length} timesteps")
            print("*"*50)
            sequences=life.values[life.shape[0]-sequence_length:,:]
            mask=np.ones(shape=(sequence_length,life.shape[1]))
            targets=life["RUL"].values[life.shape[0]-sequence_length:,:]

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

def get_transformer(config:ExperimentConfig,df:CMAPSSDataset) -> Tuple[Transformer,List[str]]:

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

    if config.transformer_type == 1:
        transformer = Transformer(
            pipelineX=make_pipeline(
                ByNameFeatureSelector(features=FEATURES),
                scaler
            ),

            pipelineY=make_pipeline(
                ByNameFeatureSelector(features=['RUL']),
                )
        )

    elif config.transformer_type == 2:
        transformer = Transformer(
            pipelineX=make_pipeline(
                ByNameFeatureSelector(features=FEATURES), 
                RollingStatistics(window=config.window_size,
                                  to_compute=config.features),
                MeanImputer(),
                # PandasVarianceThreshold(min_variance=min_variance),
                scaler
            ), 
            pipelineY=make_pipeline(
                ByNameFeatureSelector(features=['RUL']),
            )
        )

    return transformer

def load_reg_data(config:ExperimentConfig) -> Tuple[DataLoader,DataLoader,DataLoader]:
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
        train_data = CMAPSSDataset(train=True,models=config.cmapss_models)
        # train_data, val_data = train_test_split(train_data, test_size=config.val_size, shuffle=False)
        val_data = CMAPSSDataset(train=False,models=config.cmapss_models)[config.val_idx[0]:config.val_idx[1]]
        test_data = CMAPSSDataset(train=False,models=config.cmapss_models)[config.test_idx[0]:config.test_idx[1]]
    else:
        raise ValueError(f"Data name {config.data_name} not supported yet")

    # transform the data
    transformer = get_transformer(config,train_data)
    transformer.fit(train_data)
    feature_names = transformer.columns()
    transformed_train_data=train_data.map(transformer)
    transformed_val_data=val_data.map(transformer)
    transformed_test_data=test_data.map(transformer)

    # Create the TransData objects
    train_lifes=TransData(transformed_train_data)
    val_lifes=TransData(transformed_val_data)
    test_lifes=TransData(transformed_test_data)

    # Create the RegressionDatasets
    if config.model_name.startswith("chronos"):
        train_datasets = [RegressionDataset(life=life,sensors=feature_names,sequence_length=config.sequence_length) for life in train_lifes]
        val_datasets = [RegressionDataset(life=life,sensors=feature_names,sequence_length=config.sequence_length) for life in val_lifes]
        test_datasets = [RegressionDataset(life=life,sensors=feature_names,sequence_length=config.sequence_length) for life in test_lifes]
    else:
        train_datasets = [SSMRegressionDataset(life=life,sequence_length=config.sequence_length) for life in train_lifes]
        val_datasets = [SSMRegressionDataset(life=life,sequence_length=config.sequence_length) for life in val_lifes]
        test_datasets = [SSMRegressionDataset(life=life,sequence_length=config.sequence_length) for life in test_lifes]

    batch_size = len(feature_names) if config.model_name.startswith("chronos") else config.sequence_length

    train_loader=DataLoader(ConcatDataset(train_datasets),batch_size=batch_size,shuffle=False)
    val_loader=DataLoader(ConcatDataset(val_datasets),batch_size=batch_size,shuffle=False)
    test_loader=DataLoader(ConcatDataset(test_datasets),batch_size=batch_size,shuffle=False)

    return train_loader,val_loader,test_loader


# Function that returns the feature names in the CMAPSS dataset

def get_feature_names(
        config:ExperimentConfig,
) -> List[str]:

    """
    Function to get the feature names in the CMAPSS dataset according to the specific CeRULeO transformer used

    Args:
        config (ExperimentConfig): The configuration dictionary

    Returns:
        feature_names (List[str]): The list of feature names in the CMAPSS dataset
    """

    train_data = CMAPSSDataset(train=True,models=config.cmapss_models)
    transformer = get_transformer(config,train_data)
    transformer.fit(train_data)
    feature_names = transformer.columns()
    return feature_names
