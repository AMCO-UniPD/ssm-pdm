"""
Python module gathering functions to perform AD EDA experiments
"""

import ipdb
import os
from typing import Tuple
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from ceruleo.dataset.catalog.PHMDataset2018 import PHMDataset2018
from exp_config import ExperimentConfig
from utils import (
    get_current_time,
    get_transformer,
    SSMWindowRegressionDataset,
    TransData
)

from config_vars import (
    PHM_PATH,
    PHM_PATH_ACQ4,
    PHM_FAILURES
)

def load_lifes(
    exp_config: ExperimentConfig
) -> Tuple[TransData, TransData, TransData]:
    """
    Function to load the PHM dataset, split it into training, validation and test set,
    apply the transformer and return the lifes in the three datasets

    Args:
        exp_config (ExperimentConfig): experiment config object

    Returns:
        train_lifes, val_lifes, test_lifes (Tuple[TransData, TransData, TransData]): training, validation and test lifes
    """

    if os.path.exists(PHM_PATH_ACQ4):

      print("-"*50)
      print(f"Loading PHM data from {PHM_PATH_ACQ4}")
      print("-"*50)

      train_phm_data = PHMDataset2018(
          path = PHM_PATH_ACQ4,
          failure_types = PHM_FAILURES[exp_config.failure_type],
          tools = exp_config.train_phm_tools,
          train = True
      )

      test_phm_data = PHMDataset2018(
          path = PHM_PATH_ACQ4,
          failure_types = PHM_FAILURES[exp_config.failure_type],
          tools = exp_config.test_phm_tools,
          train = False
      )

    else:

      print("-"*50)
      print(f"Loading PHM data from {PHM_PATH}")
      print("-"*50)

      train_phm_data = PHMDataset2018(
          path = PHM_PATH,
          failure_types = PHM_FAILURES[exp_config.failure_type],
          tools = exp_config.train_phm_tools,
          train = True
      )

      test_phm_data = PHMDataset2018(
          path = PHM_PATH,
          failure_types = PHM_FAILURES[exp_config.failure_type],
          tools = exp_config.test_phm_tools,
          train = False
      )

    train_phm_idx = np.arange(len(train_phm_data))
    test_phm_idx = np.arange(len(test_phm_data))

    train_data, val_data, train_idx, val_idx = train_test_split(
        train_phm_data,
        train_phm_idx,
        test_size=exp_config.val_size,
        random_state = 42
    )

    transformer = get_transformer(exp_config, train_phm_data)
    transformer.fit(train_data)
    transformed_train_data = train_data.map(transformer)
    transformed_val_data = val_data.map(transformer)
    transformed_test_data = test_phm_data.map(transformer)

    train_lifes = TransData(transformed_train_data)
    val_lifes = TransData(transformed_val_data)
    test_lifes = TransData(transformed_test_data)

    return train_lifes, val_lifes, test_lifes

def get_mean_ad_info(
    dataset: SSMWindowRegressionDataset,
    txt_filepath: str = os.getcwd(),
    data_type: str = "train",
) -> None:
    """
    Function to compute statistics on the normal and anomalous windows of a dataset

    Args:
        dataset (SSMWindowRegressionDataset): windowed dataset
        txt_filepath (str): path where to save the txt file with the results
        data_type (str): type of data

    Returns:
        None: the function prints some information and saves the mean_df dataframe but does not return anything
    """

    normal_wins = dataset.get_normal_wins()
    anomalous_wins = dataset.get_anomalous_wins()

    print("-"*50)
    print(f"Number of normal windows for {data_type} dataset: {len(normal_wins)}")
    print(f"Number of anomalous windows for {data_type} dataset: {len(anomalous_wins)}")
    print("-"*50)

    normal_wins_concat = pd.concat(normal_wins)
    normal_wins_stat = normal_wins_concat.describe()
    anomalous_wins_concat = pd.concat(anomalous_wins)
    anomalous_wins_stat = anomalous_wins_concat.describe()

    normal_wins_mean = normal_wins_stat.loc["mean",:]
    anomalous_wins_mean = anomalous_wins_stat.loc["mean",:]
    mean_df = pd.concat([normal_wins_mean, anomalous_wins_mean],axis=1)
    mean_df["Difference"] = np.abs(mean_df.iloc[:,0] - mean_df.iloc[:,1])
    mean_df.columns = ["Normal mean", "Anomalous mean", "Difference"]

    print("-"*50)
    print(f"Mean dataframe for {data_type} dataset: \n {mean_df.to_markdown()}")
    print("-"*50)

    filename = f"{get_current_time()}_mean_ad_info_{data_type}.txt"
    txt_filepath = os.path.join(txt_filepath, filename)

    with open(txt_filepath,"w") as f:
        f.write(mean_df.to_markdown())

    print("-"*50)
    print(f"Mean dataframe for {data_type} dataset saved at {txt_filepath}")
    print("-"*50)

def get_normal_mean_ad_info(
    dataset: SSMWindowRegressionDataset,
    txt_filepath: str = os.getcwd(),
    data_type: str = "train",
) -> None:
    """
    Function to compare the mean over the features between the two halves of the normal lifes

    Args:
        dataset (SSMWindowRegressionDataset): windowed dataset
        txt_filepath (str): path where to save the txt file with the results
        data_type (str): type of data

    Returns:
        None: the function prints some information and saves the mean_df dataframe but does not return anything
    """

    normal_wins = dataset.get_normal_wins()
    split_idx = int(len(normal_wins)/2)
    first_half = normal_wins[:split_idx]
    first_half_concat = pd.concat(first_half)
    second_half = normal_wins[split_idx:]
    second_half_concat = pd.concat(second_half)

    first_half_stat = first_half_concat.describe()
    second_half_stat = second_half_concat.describe()

    first_half_mean = first_half_stat.loc["mean",:]
    second_half_mean = second_half_stat.loc["mean",:]
    mean_df = pd.concat([first_half_mean, second_half_mean], axis=1)
    mean_df["Difference"] = np.abs(mean_df.iloc[:,0] - mean_df.iloc[:,1])
    mean_df.columns = ["Normal mean", "Anomalous mean", "Difference"]

    print("-"*50)
    print(f"Mean dataframe for normal lifes for {data_type} dataset: \n {mean_df.to_markdown()}")
    print("-"*50)

    filename = f"{get_current_time()}_mean_ad_info_{data_type}.txt"
    txt_filepath = os.path.join(txt_filepath, filename)

    with open(txt_filepath,"w") as f:
        f.write(mean_df.to_markdown())

    print("-"*50)
    print(f"Mean dataframe for normal lifes for {data_type} dataset saved at {txt_filepath}")
    print("-"*50)

