"""
Script containing some utility functions for the `chronos-pdm` project
"""

import os
import sys
import time
import math
import json
import re
import yaml
import pickle
import ipdb
import pandas as pd
import numpy as np
import random
from typing import Tuple, List, Optional, Union

ceruleo_path = os.path.join(os.path.dirname(__file__), "ceruleo")
sys.path.append(ceruleo_path)

# ceruleo imports
from ceruleo.dataset.ts_dataset import AbstractPDMDataset
from ceruleo.dataset.catalog.CMAPSS import CMAPSSDataset
from ceruleo.dataset.catalog.PHMDataset2018 import PHMDataset2018
from ceruleo.dataset.catalog.CMAPSS import sensor_indices
from ceruleo.transformation import Transformer
from ceruleo.transformation.features.selection import (
    ByNameFeatureSelector,
    PandasVarianceThreshold,
)

# from ceruleo.transformation.features.extraction import RollingStatistics
from ceruleo.transformation.functional.pipeline.pipeline import make_pipeline
from ceruleo.transformation.features.scalers import (
    MinMaxScaler,
    RobustMinMaxScaler,
    StandardScaler,
    RobustStandardScaler,
)
from ceruleo.transformation.features.imputers import MeanImputer, RollingMeanImputer
from ceruleo.transformation.features.transformation import Clip

# sklearn imports
from sklearn.model_selection import train_test_split

# torch imports
from torch.utils.data import DataLoader, ConcatDataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from config_vars import (
    MAX_RUL,
    PHM_PATH,
    PHM_PATH_ACQ4,
    CMAPSS_MODELS,
    PHM_TOOLS,
    PHM_FAILURES,
    PHM_IN_FEATURES,
    PHM_ETCH_FEATURES,
    PHM_FAIL_TYPES,
    PHM_FEATURES,
    APPROACHES,
)
from exp_config import ExperimentConfig


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


def get_most_recent_dir(dirpath: str, file_pos: int = 0) -> str:
    """
    This function returns the most recent subdirectory inside a directory

    Args:
        dirpath: path of the directory
        file_pos: position of the directory in the list of files in the directory sorted in order of creation time, default=0
        (i.e. the most recent file is at position 0)

    Returns:
        The most recent subdirectory in the directory
    """

    assert os.path.isdir(dirpath), "The provided path  is not a directory"

    dirs = [f for f in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath, f))]
    paths = [os.path.join(dirpath, basename) for basename in dirs]
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

    FILETYPES = [
        "pickle",
        "pth",
        "json",
    ]
    assert filetype in FILETYPES, (
        f"filetype must be one of {FILETYPES}, but got {filetype}"
    )

    if filetype == "pickle":
        with open(file_path, "rb") as fl:
            element = pickle.load(fl)
    elif filetype == "pth":
        element = torch.load(file_path)
    elif filetype == "json":
        with open(file_path, "r") as fl:
            element = json.load(fl)
    else:
        raise ValueError(f"Invalid filetype. Please choose one in {FILETYPES}")
    return element


def load_yaml_to_dict(file_path: str) -> dict:
    """
    Loads the contents of a YAML file into a Python dictionary.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: A dictionary containing the data from the YAML file, or None if an error occurs.
    """
    try:
        with open(file_path, "r") as file:
            data = yaml.safe_load(
                file
            )  # Use safe_load to prevent arbitrary code execution
        return data
    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")


def save_element(
    element: Union[dict, nn.Module, pd.DataFrame, List],
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


def generate_path(basepath: str = os.getcwd(), folders: List[str] = []) -> str:
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

    path = basepath + "/"

    # Join the basepath with the folders
    for folder in folders:
        path = os.path.join(path, folder) + "/"
        # Verify weather the path exists or not
        if not os.path.exists(path):
            os.makedirs(path)

    return path[:-1]


class TransData(AbstractPDMDataset):
    def __init__(self, data: AbstractPDMDataset):
        super().__init__()
        self.lives = []
        for life in data:
            current_life = pd.concat([life[0], life[1]], axis=1)
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
            pad_arr = np.zeros(sequence_length - life.shape[0])
            mask = [
                np.concatenate(
                    (np.ones(life.shape[0]), np.zeros(sequence_length - life.shape[0]))
                )
                for _ in sensors
            ]
            sequences = [
                np.concatenate((life[sensor].values, pad_arr)) for sensor in sensors
            ]
            targets = [np.concatenate((life["RUL"].values, pad_arr)) for _ in sensors]
        else:
            print("*" * 50)
            print(
                f"Warning: This life is longer than {sequence_length}, removing the first {life.shape[0] - sequence_length} timesteps"
            )
            print("*" * 50)
            sequences = [
                life[sensor].values[life.shape[0] - sequence_length :]
                for sensor in sensors
            ]
            mask = [np.ones(sequence_length) for _ in sensors]
            targets = [
                life["RUL"].values[life.shape[0] - sequence_length :] for _ in sensors
            ]

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


# Regression dataset class for SMM models for the padding approach


class SSMRegressionDataset(Dataset):
    def __init__(
        self,
        life: pd.DataFrame,
        sequence_length: int = 500,
        max_rul: int = MAX_RUL,
        normalize_rul: bool = False,
    ):
        """
        This class implements the dataset for the padding approach. In
        this approach we set a maximum length (i.e. sequence_length) for
        the time series associated to a life and we extract a sequence of that
        length from each life. If the life is longer than sequence_length we take
        the last sequence_length samples from it, otherwise we take the entire time
        series and we use 0 padding to cover for the sequence_length - life.shape[0]
        missing samples. As usual we have to use a mask to keep track of the samples
        that are padded as zeros.

        Args:
            life (pd.DataFrame): input life
            sequence_length (int): length to use for this life
            max_rul (int): maximum RUL value used to select the windows and to normalize the RUL
            normalize_rul (bool): weather to normalize the RUL
        """

        self.max_rul = max_rul
        self.normalize_rul = normalize_rul

        life, rul = life.iloc[:, :-1], life["RUL"]

        # NOTE: Time series shorter than sequence_length, we use 0 padding

        if sequence_length > life.shape[0]:
            pad_arr = np.zeros(shape=(sequence_length - life.shape[0], life.shape[1]))
            mask = np.concatenate(
                (
                    np.ones(shape=(life.shape[0])),
                    np.zeros(shape=(sequence_length - life.shape[0])),
                )
            )
            sequences = np.concatenate((life.values, pad_arr))
            targets = np.concatenate((rul.values, pad_arr[:, -1]))

        # NOTE: Time series longer than sequence_length we take the last sequence_length samples

        else:
            sequences = life.values[life.shape[0] - sequence_length :, :]
            mask = np.ones(shape=(sequence_length))
            targets = rul[life.shape[0] - sequence_length :]

        targets = targets / self.max_rul if self.normalize_rul else targets

        self.sequences = np.expand_dims(sequences, axis=0)
        self.targets = np.expand_dims(targets, axis=0)
        self.mask = np.expand_dims(mask, axis=0)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        mask = torch.tensor(self.mask[idx], dtype=torch.float32)
        return sequence, target, mask


# SSM Regression dataset class for the windowed approach


class SSMWindowRegressionDataset(Dataset):
    def __init__(
        self,
        lifes: TransData,
        sequence_length: int = 500,
        stride: int = 1,
        max_rul: int = MAX_RUL,
        normalize_rul: bool = False,
        ad: bool = False,
    ):
        """
        This class implements the dataset for the windowed approach.
        Each life is divided into multiple subsequences of length
        equal to sequence length. In case a sequence is too small
        it is padded with zeros and a mask keeps track of that to not
        consider the predictions done on the padded indexes in the loss
        computation. The dataset can be turned into an AD dataset using the
        ad boolean flag argument.

        Args:
            lifes (TransData): lifes transformed with a transformer ceruelo object
            sequence_length (int): length of the sub sequences
            stride (int): stride between two consecutive windows
            max_rul (int): maximum RUL value used to select the windows and to normalize the RUL
            normalize_rul (bool): weather to normalize the RUL
            ad (bool): weather to use AD version or not
        """

        self.lifes = [life.iloc[:, :-1] for life in lifes]
        self.ruls = [life["RUL"] for life in lifes]
        self.targets = self.ruls if not ad else self.lifes
        self.sequence_length = sequence_length
        self.stride = stride
        self.max_rul = max_rul
        self.normalize_rul = normalize_rul
        self.ad = ad

        self.data_indices = []
        for i, df in enumerate(self.lifes):
            self.num_sequences = max(
                0, math.ceil((len(df) - self.sequence_length) / self.stride) + 1
            )

            if self.num_sequences > 0:
                # Store tuples of (life_index, start_index_of_window)
                self.data_indices.extend(
                    [(i, j * self.stride) for j in range(self.num_sequences)]
                )

            else:
                # If the signal is too short we'll create a single sub sequence starting from 0 and we'll add padding
                self.data_indices.extend([(i, 0)])

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        life_idx, start_idx = self.data_indices[idx]
        life = self.lifes[life_idx]
        rul = self.targets[life_idx]
        mask = np.ones(shape=(self.sequence_length, 1))

        # NOTE: Life longer than sequence_length: we create the sub sequence

        end_idx = start_idx + self.sequence_length
        if end_idx <= life.shape[0]:
            inputs = life.iloc[start_idx:end_idx].values
            targets = rul.iloc[start_idx:end_idx].values

        # NOTE: Life shorter than sequence_length: we use 0 padding

        else:
            inputs = life.iloc[start_idx : life.shape[0]].values
            targets = rul.iloc[start_idx : rul.shape[0]].values
            pad_idx = self.sequence_length - (life.shape[0] - start_idx)

            # NOTE: Concatenate inputs and targets with pad_idx

            inputs = np.concatenate(
                (inputs, np.zeros(shape=(pad_idx, inputs.shape[1])))
            )

            if self.ad:
                targets = np.concatenate(
                    (targets, np.zeros(shape=(pad_idx, inputs.shape[1])))
                )

            else:
                targets = np.concatenate((targets, np.zeros(shape=(pad_idx,))))

            # NOTE: From pad_idx to the end the mask becomes 0

            mask[pad_idx:] = 0

        # NOTE: Normalize the RUL if self.normalize_rul is true
        if not self.ad:
            targets = targets / self.max_rul if self.normalize_rul else targets
            targets = np.expand_dims(targets, axis=-1)

        sequence = torch.tensor(inputs, dtype=torch.float32)
        target = torch.tensor(targets, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        return sequence, target, mask

    def select_windows(
        self,
        n_const_win: int = 10,
    ):
        """
        Function to select only the windows with at least one RUL value
        lower than MAX RUL and keep the others with probability keep_long_rul_prob

        Args:
            n_const_win (int): number of constant windows to keep

        Returns:
            None: the method filters the class attributes self.lifes and self.targets
        """

        filtered_indices = []

        for idx in range(len(self.data_indices)):
            life_idx, start_idx = self.data_indices[idx]
            rul = self.ruls[life_idx]

            # Determine the target window values for this specific index
            end_idx = start_idx + self.sequence_length
            # Use the same logic as __getitem__ to get the target slice
            target_slice = rul.iloc[start_idx : min(end_idx, len(rul))].values

            # Condition: at least one RUL value < self.max_rul
            is_near_failure = np.any(target_slice < self.max_rul)

            if is_near_failure:
                filtered_indices.append(self.data_indices[idx])

        constant_seq = list(set(self.data_indices) - set(filtered_indices))
        seq_to_keep = constant_seq.copy()
        self.normal_seq = constant_seq
        self.anomalous_seq = filtered_indices

        if not self.ad:
            # seq_to_keep = seq_to_keep[-int(keep_long_rul_prob*len(seq_to_keep)):]
            # seq_to_keep = seq_to_keep[-n_const_win:]
            seq_to_keep = seq_to_keep[-len(self.anomalous_seq):]

            seq_to_keep.extend(filtered_indices)
            self.data_indices = seq_to_keep

    def get_normal_wins(self) -> List:
        """
        Function that returns a list with the raw signal in the normal
        windows

        Args:
            no input arguments required

        Returns:
            normal_wins (np.ndarray): array containing all the normal windows
        """

        normal_wins = []

        for life_idx, start_idx in self.normal_seq:
            life = self.lifes[life_idx]
            end_idx = start_idx + self.sequence_length
            normal_win = life[start_idx : min(end_idx, life.shape[0])]
            normal_wins.append(normal_win)

        return normal_wins

    def get_anomalous_wins(self) -> List:
        """
        Function that returns a list with the raw signal in the anomalous
        windows

        Args:
            no input arguments required

        Returns:
            anomalous_wins (np.ndarray): array containing all the anomalous windows
        """

        anomalous_wins = []

        for life_idx, start_idx in self.anomalous_seq:
            life = self.lifes[life_idx]
            end_idx = start_idx + self.sequence_length
            anomalous_win = life[start_idx : min(end_idx, life.shape[0])]
            anomalous_wins.append(anomalous_win)

        return anomalous_wins


class SSMFullLifeRegressionDataset(Dataset):
    def __init__(
        self,
        life: pd.DataFrame,
        max_rul: int = MAX_RUL,
        normalize_rul: bool = False,
    ):
        """
        This class implements the dataset for the full life approach.
        It uses the entire life and pads it to the closest power of 2 length.

        Args:
            life (pd.DataFrame): input life
            max_rul (int): maximum RUL value used to normalize the RUL
            normalize_rul (bool): weather to normalize the RUL
        """

        self.max_rul = max_rul
        self.normalize_rul = normalize_rul

        life, rul = life.iloc[:, :-1], life["RUL"]

        life_length = life.shape[0]
        padded_length = 2 ** math.ceil(math.log2(life_length))

        if padded_length > life_length:
            pad_arr = np.zeros(shape=(padded_length - life_length, life.shape[1]))
            mask = np.concatenate(
                (
                    np.ones(shape=(life_length)),
                    np.zeros(shape=(padded_length - life_length)),
                )
            )
            sequences = np.concatenate((life.values, pad_arr))
            targets = np.concatenate(
                (rul.values, np.zeros(padded_length - life_length))
            )
        else:
            sequences = life.values
            targets = rul.values
            mask = np.ones(shape=(padded_length))

        targets = targets / self.max_rul if self.normalize_rul else targets

        self.sequences = np.expand_dims(sequences, axis=0)
        self.targets = np.expand_dims(targets, axis=0)
        self.mask = np.expand_dims(mask, axis=0)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        mask = torch.tensor(self.mask[idx], dtype=torch.float32)
        return sequence, target, mask


def get_transformer(
    config: ExperimentConfig,
    df: Union[CMAPSSDataset, PHMDataset2018],
) -> Tuple[Transformer, List[str]]:
    """
    Create a transformer object to preprocess the data from the CMAPSS dataset

    Args:
        config (ExperimentConfig): The configuration dictionary
        df (CMAPSSDataset): The CMAPSS dataset

    Returns:
        transformer (Transformer): The ceruleo transformer object
    """

    if config.data_name == "CMAPSS":
        FEATURES = [df[0].columns[i] for i in sensor_indices]
    else:
        if config.feature_type == "etch":
            FEATURES = PHM_ETCH_FEATURES
        elif config.feature_type == "phm_in":
            FEATURES = PHM_IN_FEATURES
        else:
            FEATURES = PHM_FEATURES

    if config.scaler == "minmax":
        scaler = MinMaxScaler(
            range=(
                config.scaler_kwargs["low_limit"],
                config.scaler_kwargs["high_limit"],
            )
        )
    elif config.scaler == "standard":
        scaler = StandardScaler()

    if config.transformer_type == 0:
        transformer = Transformer(
            pipelineX=make_pipeline(ByNameFeatureSelector(features=FEATURES)),
            pipelineY=make_pipeline(ByNameFeatureSelector(features=["RUL"])),
        )
    elif config.transformer_type == 1:
        transformer = Transformer(
            pipelineX=make_pipeline(
                ByNameFeatureSelector(features=FEATURES),
                MeanImputer(),
                scaler,
            ),
            pipelineY=make_pipeline(
                ByNameFeatureSelector(features=["RUL"]), Clip(lower=0.0, upper=500.0)
            ),
        )

    elif config.transformer_type == 2:
        transformer = Transformer(
            pipelineX=make_pipeline(
                ByNameFeatureSelector(features=FEATURES),
                MeanImputer(),
            ),
            pipelineY=make_pipeline(
                ByNameFeatureSelector(features=["RUL"]), Clip(lower=0.0, upper=500.0)
            ),
        )

    elif config.transformer_type == 3:
        transformer = Transformer(
            pipelineX=make_pipeline(
                ByNameFeatureSelector(features=FEATURES),
                # RollingStatistics(
                #                     window=config.window_size, to_compute=config.features
                #                 ),
                MeanImputer(),
                # PandasVarianceThreshold(min_variance=config.min_variance),
                scaler,
            ),
            pipelineY=make_pipeline(
                ByNameFeatureSelector(features=["RUL"]),
            ),
        )

    return transformer


# Function to combined the predictions on the sub sequences in the windowed approach


def combine_values(
    predictions: np.ndarray,
    true_values: np.ndarray,
    an_scores: np.ndarray,
    original_shape: int,
    sequence_length: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine the predictions done by the model on the different sub sequences in which each life was divided in the `seq_to_seq` approach

    Args:
        predictions: np.array containing the predictions for each sub sequence
        true_values: np.array containing the true values for each sub sequence
        an_scores: np.array containing the anomaly scores for each sub sequence
        original_shape: shape of the signal before applying the sliding windows approach
        sequence_length: length of the sequences
        stride: stride used to construct the sliding windows

    Returns:
        combined_predictions: np.array containing the combined
        combined_true_vals: np.array containing the combined true values
        combined_an_scores: np.array containing the combined anomaly scores.
        In case there are no anomaly scores the function returns an array of ones
    """

    # NOTE: In case we have non overlapping windows we just need
    # to concatenate ther predictions and true values over the windows

    if sequence_length == stride:
        combined_predictions = np.concatenate(predictions)
        combined_true_vals = np.concatenate(true_values)
        combined_an_scores = (
            np.concatenate(an_scores)
            if len(an_scores) != 0
            else np.ones_like(combined_true_vals)
        )

        return combined_predictions, combined_true_vals, combined_an_scores

    n_samples = (len(predictions) * stride) + sequence_length
    combined_predictions = (
        np.zeros((n_samples, predictions.shape[-1]))
        if len(an_scores) != 0
        else np.zeros(n_samples)
    )
    combined_true_vals = (
        np.zeros((n_samples, predictions.shape[-1]))
        if len(an_scores) != 0
        else np.zeros(n_samples)
    )
    combined_an_scores = np.zeros(n_samples)
    counts = np.zeros(n_samples)

    # for i, (preds, true, an_score) in enumerate(zip(predictions, true_values, an_scores)):
    for i, (preds, true) in enumerate(zip(predictions, true_values)):
        start_index = i * stride
        end_index = start_index + sequence_length
        combined_predictions[start_index:end_index] += preds
        combined_true_vals[start_index:end_index] += true
        combined_an_scores[start_index:end_index] += (
            an_score if len(an_scores) != 0 else 1
        )
        counts[start_index:end_index] += 1

    nonzero_counts = counts != 0

    if combined_predictions[nonzero_counts].ndim == counts[nonzero_counts].ndim:
        combined_predictions[nonzero_counts] /= counts[nonzero_counts]
        combined_true_vals[nonzero_counts] /= counts[nonzero_counts]
    else:
        combined_predictions[nonzero_counts] /= np.expand_dims(
            counts[nonzero_counts], axis=1
        )
        combined_true_vals[nonzero_counts] /= np.expand_dims(
            counts[nonzero_counts], axis=1
        )

    combined_an_scores[nonzero_counts] /= counts[nonzero_counts]

    if n_samples > original_shape:
        print("-" * 50)
        print(
            f"{n_samples} grater than the original shape {original_shape} so removing the last {n_samples - original_shape} samples"
        )
        print("-" * 50)

        return (
            combined_predictions[: -(n_samples - original_shape)],
            combined_true_vals[: -(n_samples - original_shape)],
            combined_an_scores[: -(n_samples - original_shape)],
        )

    return combined_predictions, combined_true_vals, combined_an_scores


def load_reg_data(config: ExperimentConfig) -> dict:
    """
    Load the data from a RUL dataset (e.g. CMAPSS,CMAPSS-2) and convert them
    into a DataLoader object with minibatched of size 1, each one containing a life

    Args:
        config (ExperimentConfig): The configuration dictionary

    Returns:
        loaders_dict (dict): A dictionary containing the DataLoader objects for the train, validation and test sets. In the case of the windowed
        approach, it also contains a list of DataLoader objects for each life in the test set.
    """

    assert config.data_name == "CMAPSS", (
        "This function works just with the CMAPSS dataset"
    )
    assert config.cmapss_models in CMAPSS_MODELS, (
        f"The models must be one of {CMAPSS_MODELS}"
    )

    train_data = CMAPSSDataset(train=True, models=config.cmapss_models)
    # train_data, val_data = train_test_split(train_data, test_size=config.val_size, shuffle=False)
    val_data = CMAPSSDataset(train=False, models=config.cmapss_models)[
        config.val_idx[0] : config.val_idx[1]
    ]
    test_data = CMAPSSDataset(train=False, models=config.cmapss_models)[
        config.test_idx[0] : config.test_idx[1]
    ]

    # transform the data
    transformer = get_transformer(config, train_data)
    transformer.fit(train_data)
    feature_names = transformer.columns()
    transformed_train_data = train_data.map(transformer)
    transformed_val_data = val_data.map(transformer)
    transformed_test_data = test_data.map(transformer)

    # Create the TransData objects
    train_lifes = TransData(transformed_train_data)
    val_lifes = TransData(transformed_val_data)
    test_lifes = TransData(transformed_test_data)

    # Create the RegressionDatasets
    if config.model_name.startswith("chronos"):
        train_datasets = [
            RegressionDataset(
                life=life, sensors=feature_names, sequence_length=config.sequence_length
            )
            for life in train_lifes
        ]
        val_datasets = [
            RegressionDataset(
                life=life, sensors=feature_names, sequence_length=config.sequence_length
            )
            for life in val_lifes
        ]
        test_datasets = [
            RegressionDataset(
                life=life, sensors=feature_names, sequence_length=config.sequence_length
            )
            for life in test_lifes
        ]
    elif config.approach == "padding":
        train_datasets = [
            SSMRegressionDataset(life=life, sequence_length=config.sequence_length)
            for life in train_lifes
        ]
        val_datasets = [
            SSMRegressionDataset(life=life, sequence_length=config.sequence_length)
            for life in val_lifes
        ]
        test_datasets = [
            SSMRegressionDataset(life=life, sequence_length=config.sequence_length)
            for life in test_lifes
        ]
    elif config.approach == "windowed":
        train_datasets = [
            SSMWindowRegressionDataset(
                life=life,
                sequence_length=config.sequence_length,
                max_rul=MAX_RUL,
            )
            for life in train_lifes
        ]
        val_datasets = [
            SSMWindowRegressionDataset(
                life=life, sequence_length=config.sequence_length
            )
            for life in val_lifes
        ]
        test_datasets = [
            SSMWindowRegressionDataset(
                life=life, sequence_length=config.sequence_length
            )
            for life in test_lifes
        ]

    if config.model_name.startswith("chronos"):
        batch_size = len(feature_names)
    elif config.approach == "padding":
        batch_size = config.sequence_length
    else:
        batch_size = config.batch_size

    train_loader = DataLoader(
        ConcatDataset(train_datasets), batch_size=batch_size, shuffle=False
    )
    val_loader = DataLoader(
        ConcatDataset(val_datasets), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        ConcatDataset(test_datasets), batch_size=batch_size, shuffle=False
    )

    test_loaders = [DataLoader(test_dataset) for test_dataset in test_datasets]

    loaders_dict = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "test_loaders": test_loaders,
    }

    return loaders_dict


def create_padding_loaders(
    config: ExperimentConfig,
    train_lifes: TransData,
    val_lifes: TransData,
    test_lifes: TransData,
) -> dict:
    """
    Function to create the dataloaders for the padding approach.

    Args:
        config (ExperimentConfig): experiment configuration object
        train_lifes (TransData): transformed training lifes
        val_lifes (TransData): transformed validation lifes
        test_lifes (TransData): transformed test lifes

    Returns:
        loaders_dict (dict): dictionary containing the dataloaders
    """

    train_datasets = [
        SSMRegressionDataset(
            life=life,
            sequence_length=config.sequence_length,
            max_rul=config.max_rul,
            normalize_rul=config.normalize_rul,
        )
        for life in train_lifes
    ]
    val_datasets = [
        SSMRegressionDataset(
            life=life,
            sequence_length=config.sequence_length,
            max_rul=config.max_rul,
            normalize_rul=config.normalize_rul,
        )
        for life in val_lifes
    ]
    test_datasets = [
        SSMRegressionDataset(
            life=life,
            sequence_length=config.sequence_length,
            max_rul=config.max_rul,
            normalize_rul=config.normalize_rul,
        )
        for life in test_lifes
    ]

    batch_size = config.batch_size

    train_loader = DataLoader(
        ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        ConcatDataset(val_datasets), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        ConcatDataset(test_datasets), batch_size=batch_size, shuffle=True
    )
    test_loaders = [DataLoader(test_dataset) for test_dataset in test_datasets]

    loaders_dict = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "test_loaders": test_loaders,
    }

    return loaders_dict


def create_full_life_loaders(
    config: ExperimentConfig,
    train_lifes: TransData,
    val_lifes: TransData,
    test_lifes: TransData,
) -> dict:
    """
    Function to create the dataloaders for the full life approach.
    The function is very similar to create_padding_loaders but in this
    case we use the SSMFullLifeRegressionDataset class
    and we have to set the batch_size to 1 because each life has a different
    length

    Args:
        config (ExperimentConfig): experiment configuration object
        train_lifes (TransData): transformed training lifes
        val_lifes (TransData): transformed validation lifes
        test_lifes (TransData): transformed test lifes

    Returns:
        loaders_dict (dict): dictionary containing the dataloaders
    """

    train_datasets = [
        SSMFullLifeRegressionDataset(
            life=life,
            max_rul=config.max_rul,
            normalize_rul=config.normalize_rul,
        )
        for life in train_lifes
    ]
    val_datasets = [
        SSMFullLifeRegressionDataset(
            life=life,
            max_rul=config.max_rul,
            normalize_rul=config.normalize_rul,
        )
        for life in val_lifes
    ]
    test_datasets = [
        SSMFullLifeRegressionDataset(
            life=life,
            max_rul=config.max_rul,
            normalize_rul=config.normalize_rul,
        )
        for life in test_lifes
    ]

    batch_size = 1

    train_loader = DataLoader(
        ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        ConcatDataset(val_datasets), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        ConcatDataset(test_datasets), batch_size=batch_size, shuffle=True
    )
    test_loaders = [DataLoader(test_dataset) for test_dataset in test_datasets]

    loaders_dict = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "test_loaders": test_loaders,
    }

    return loaders_dict


def create_window_dataset(
    config: ExperimentConfig, lifes: TransData, eval: bool = False
) -> SSMWindowRegressionDataset:
    """
    Function to create a SSMWindowRegressionDataset object starting from a set of
    run to failure cycles

    Args:
        config (ExperimentConfig): experiment configuration object
        lifes (TransData): list of lifes
        eval (bool): weather to create the dataset in evaluation mode or not

    Returns:
        dataset (SSMWindowRegressionDataset): dataset object
    """

    dataset = SSMWindowRegressionDataset(
        lifes=lifes,
        sequence_length=config.sequence_length,
        stride=config.stride,
        max_rul=MAX_RUL,
        normalize_rul=config.normalize_rul,
        ad=config.ad,
    )
    if not eval:
        dataset.select_windows(n_const_win=config.n_const_win)

    return dataset


def create_window_loaders(
    config: ExperimentConfig,
    train_lifes: TransData,
    val_lifes: TransData,
    test_lifes: TransData,
    eval: bool = False,
) -> Union[List[DataLoader], dict]:
    """
    Function to create the dataloaders for the window approach.

    Args:
        config (ExperimentConfig): experiment configuration object
        train_lifes (TransData): transformed training lifes
        val_lifes (TransData): transformed validation lifes
        test_lifes (TransData): transformed test lifes
        eval (bool): boolean flag to decide weather to load the dataloaders in eval mode
        (i.e one loader per life) or not

    Returns:
        loaders_dict (dict): dictionary containing the dataloaders if eval=False
        test_loaders (List[DataLoader]): list of dataloaders of the test lifes if eval=True
    """

    if eval:
        print("-" * 50)
        print("Creating window dataloaders in evaluation mode")
        print("-" * 50)

        # NOTE: For the evaluation lifes we do not use select_windows because we want
        # to test the model on the entire life

        test_datasets = []
        for test_life in test_lifes:
            test_dataset = create_window_dataset(config=config, lifes=[test_life])
            test_datasets.append(test_dataset)

        test_loaders = [
            DataLoader(test_dataset, batch_size=config.batch_size)
            for test_dataset in test_datasets
        ]

        print("-" * 50)
        print("window dataloaders created successfully")
        print("-" * 50)

        return test_loaders

    else:
        print("-" * 50)
        print("Creating window dataloaders in training mode")
        print("-" * 50)

        train_datasets = create_window_dataset(config=config, lifes=train_lifes)

        val_datasets = create_window_dataset(config=config, lifes=val_lifes)

        test_datasets = create_window_dataset(config=config, lifes=test_lifes)

        train_loader = DataLoader(
            train_datasets, batch_size=config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_datasets, batch_size=config.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_datasets, batch_size=config.batch_size, shuffle=True
        )
        test_loaders = [
            DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
            for test_dataset in test_datasets
        ]

        print("-" * 50)
        print("window dataloaders created successfully")
        print("-" * 50)

        loaders_dict = {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "test_loaders": test_loaders,
        }

        return loaders_dict


def print_life_info(phm_data: PHMDataset2018) -> None:
    """
    print statement with some information on the lifes durations

    Args:
        phm_data (PHMDataset2018): PHM dataset object

    Results:
        None: the function does not return anything, it just prints some information on the shape of the
        lifes in the dataset
    """

    for i, life in enumerate(phm_data):
        print("-" * 50)
        print(f"Shape of life {i}: {life.shape}")
        print("-" * 50)


def load_phm_data(config: ExperimentConfig, eval: bool = False) -> dict:
    """
    Clone of the load_reg_data function but adapted for the PHM dataset.

    Args:
        config (ExperimentConfig): The configuration dictionary
        eval (bool): weather to load the loaders in eval mode in the windowed approach

    Returns:
        loaders_dict (dict): A dictionary containing the DataLoader objects for the train, validation and test sets. In the case of the windowed
        approach, it also contains a list of DataLoader objects for each life in the test set.
    """

    assert config.data_name == "PHM", "This function works just with the PHM dataset"
    assert set(config.train_phm_tools).issubset(PHM_TOOLS), (
        f"The set of train tools must be a subset of {PHM_TOOLS} but got {config.train_phm_tools}"
    )
    assert set(config.test_phm_tools).issubset(PHM_TOOLS), (
        f"The set of test tools must be a subset of {PHM_TOOLS} but got {config.test_phm_tools}"
    )
    assert config.failure_type in PHM_FAIL_TYPES, (
        f"Failure type name must be in {PHM_FAIL_TYPES} but got {config.failure_type}"
    )

    if os.path.exists(PHM_PATH_ACQ4):
        print("-" * 50)
        print(f"Loading PHM data from {PHM_PATH_ACQ4}")
        print("-" * 50)

        train_phm_data = PHMDataset2018(
            path=PHM_PATH_ACQ4,
            failure_types=PHM_FAILURES[config.failure_type],
            tools=config.train_phm_tools,
            train=True,
        )

        test_phm_data = PHMDataset2018(
            path=PHM_PATH_ACQ4,
            failure_types=PHM_FAILURES[config.failure_type],
            tools=config.test_phm_tools,
            train=False,
        )

    else:
        print("-" * 50)
        print(f"Loading PHM data from {PHM_PATH}")
        print("-" * 50)

        train_phm_data = PHMDataset2018(
            path=PHM_PATH,
            failure_types=PHM_FAILURES[config.failure_type],
            tools=config.train_phm_tools,
            train=True,
        )

        test_phm_data = PHMDataset2018(
            path=PHM_PATH,
            failure_types=PHM_FAILURES[config.failure_type],
            tools=config.test_phm_tools,
            train=False,
        )

    print_life_info(phm_data=test_phm_data)

    train_phm_idx = np.arange(len(train_phm_data))
    test_phm_idx = np.arange(len(test_phm_data))

    train_data, val_data, train_idx, val_idx = train_test_split(
        train_phm_data, train_phm_idx, test_size=config.val_size, random_state=42
    )

    transformer = get_transformer(
        config=config,
        df=train_phm_data,
    )
    transformer.fit(train_data)
    transformed_train_data = train_data.map(transformer)
    transformed_val_data = val_data.map(transformer)
    transformed_test_data = test_phm_data.map(transformer)

    train_lifes = TransData(transformed_train_data)
    val_lifes = TransData(transformed_val_data)
    test_lifes = TransData(transformed_test_data)

    if config.approach == "padding":
        loaders_dict = create_padding_loaders(
            config=config,
            train_lifes=train_lifes,
            val_lifes=val_lifes,
            test_lifes=test_lifes,
        )

    elif config.approach == "full_life":
        loaders_dict = create_full_life_loaders(
            config=config,
            train_lifes=train_lifes,
            val_lifes=val_lifes,
            test_lifes=test_lifes,
        )

    elif config.approach == "windowed":
        loaders_dict = create_window_loaders(
            config=config,
            train_lifes=train_lifes,
            val_lifes=val_lifes,
            test_lifes=test_lifes,
            eval=eval,
        )

    else:
        raise ValueError(
            f"Approach {config.approach} not supported. Supported approaches are {APPROACHES}"
        )

    if eval:
        return {
            "test_lifes": test_lifes,
            "test_loaders": loaders_dict["test_loaders"],
            "test_idx": test_phm_idx,
        }

    loaders_dict["train_idx"] = train_idx
    loaders_dict["val_idx"] = val_idx
    loaders_dict["test_idx"] = test_phm_idx

    return loaders_dict


# Function that returns the feature names in the CMAPSS dataset


def get_feature_names(
    config: ExperimentConfig,
) -> List[str]:
    """
    Function to get the feature names in the CMAPSS dataset according to the specific CeRULeO transformer used

    Args:
        config (ExperimentConfig): The configuration dictionary

    Returns:
        feature_names (List[str]): The list of feature names in the CMAPSS dataset
    """

    train_data = CMAPSSDataset(train=True, models=config.cmapss_models)
    transformer = get_transformer(config, train_data)
    transformer.fit(train_data)
    feature_names = transformer.columns()
    return feature_names


def get_phm_feature_names(
    config: ExperimentConfig,
) -> List[str]:
    """
    Function to get the feature names in the PHM dataset according to the specific CeRULeO transformer used

    Args:
        config (ExperimentConfig): The configuration dictionary

    Returns:
        feature_names (List[str]): The list of feature names in the CMAPSS dataset
    """

    if os.path.exists(PHM_PATH_ACQ4):
        phm_data = PHMDataset2018(
            path=PHM_PATH_ACQ4,
            failure_types=PHM_FAILURES[config.failure_type],
            tools=config.test_phm_tools,
            train=False,
        )

    else:
        phm_data = PHMDataset2018(
            path=PHM_PATH,
            failure_types=PHM_FAILURES[config.failure_type],
            tools=config.test_phm_tools,
            train=False,
        )

    transformer = get_transformer(config, phm_data)
    transformer.fit(phm_data)
    feature_names = transformer.columns()
    return feature_names


# Function to sample a quantile level for the quantile regression approach


def sample_quantile(
    quantile_dist: str = "uniform",
    bounds: List[float] = [0.1, 0.9],
    print_quantile: bool = False,
) -> float:
    """
    Function to sample a quantile level for the quantile regression approach

    Args:
        quantile_dist: The distribution to sample the quantile level from, default='uniform'
        bounds: The bounds of the distribution to sample the quantile level from. Interval [a, b] for uniform distribution and mean and standard deviation for normal distribution, default=[0.1, 0.9]
        print_quantile: Boolean to indicate whether to print the sampled quantile level, default=False

    Returns:
        quantile: The sampled quantile level
    """

    assert quantile_dist in [
        "uniform",
        "normal",
    ], "quantile_dist must be either 'uniform' or 'normal'"
    assert len(bounds) == 2, "bounds must be a list of two elements"
    if quantile_dist == "normal":
        assert bounds[1] > 0, "The standard deviation must be positive"
    if quantile_dist == "uniform":
        assert bounds[0] < bounds[1], (
            "The lower bound must be less than the upper bound"
        )

    quantile = 0.5

    if quantile_dist == "uniform":
        quantile = np.random.uniform(bounds[0], bounds[1])
    elif quantile_dist == "normal":
        quantile = np.random.normal(bounds[0], bounds[1])

    if print_quantile:
        if quantile_dist == "uniform":
            print(
                f"Sampled quantile level from U[{bounds[0]}, {bounds[1]}]: {quantile}"
            )
        elif quantile_dist == "normal":
            print(
                f"Sampled quantile level from N[{bounds[0]}, {bounds[1]}]: {quantile}"
            )

    return quantile


# Function to set the seed for reproducibility


def set_seed(seed: int = 0) -> None:
    """
    Set the seed for reproducibility on random, numpy and torch

    Args:
        seed (int): seed value, by default 0

    Returns:
        None: The function sets the seed and does not return anything
    """
    random.seed(seed)  # For Python's random module
    np.random.seed(seed)  # For NumPy
    torch.manual_seed(seed)  # For PyTorch on CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # For PyTorch on a single GPU
        torch.cuda.manual_seed_all(seed)  # For PyTorch on all GPUs
    torch.backends.cudnn.deterministic = (
        True  # Ensures deterministic behavior for cuDNN
    )
    torch.backends.cudnn.benchmark = (
        False  # Disables cuDNN auto-tuner for deterministic results
    )


def extract_number(
    text: str,
) -> Union[float, None]:
    """
    Extracts the first floating point number from a string.

    Args:
      text: The input string.

    Returns:
      A float representing the extracted number, or None if no number is found.
    """

    match = re.search(r"[-+]?\d*\.\d+|\d+", text)
    if match:
        return float(match.group(0))
    else:
        return None


def an_score_to_rul(an_scores: np.ndarray, max_rul: int = MAX_RUL) -> np.ndarray:
    """
    This function converts an array containing the anomaly scores over the samples
    of a life into a RUL signal. The current formula used to convert to the RUL is
    the following:
    - subtract the maximum anomaly score from all samples
    - multiply by max_rul

    Args:
        an_scores (np.ndarray): array with the anomaly scores
        max_rul (int): maximum RUL value

    Returns:
        rul_scores (np.ndarray): array of converted RUL scores
    """

    an_scores_norm = np.max(an_scores) - an_scores
    rul_scores = an_scores_norm * max_rul

    return rul_scores
