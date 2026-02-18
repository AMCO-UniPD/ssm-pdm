"""
Python script to get information on the AD dataset extracted from PHM dataset
"""

# general imports
import os
import sys
import numpy as np
import pandas as pd
import ipdb
import torch
import argparse
import setproctitle

src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.append(src_path)

from exp_config import setup_exp
from ceruleo.dataset.catalog.PHMDataset2018 import PHMDataset2018
from sklearn.model_selection import train_test_split
from config_vars import (
    PHM_PATH_ACQ4,
    PHM_PATH,
    PHM_FAILURES,
    MAX_RUL
)
from utils import (
    get_current_time,
    generate_path,
    get_transformer,
    TransData,
    SSMWindowRegressionDataset,
    create_window_dataset
)

experiment_path = os.path.dirname((os.path.realpath(__file__)))
exp_config, model_config, device, exp_name = setup_exp()

def get_ad_info(
    dataset: SSMWindowRegressionDataset,
    txt_filepath: str = experiment_path,
    data_type: str = "train",
):
    """
    Function to compute statistics on the normal and anomalous windows of a dataset
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

print("-"*50)
print("AD dataset info")
print(f"Sequence length: {exp_config.sequence_length}")
print(f"Stride: {exp_config.stride}")
print("-"*50)

txt_filepath = generate_path(
    basepath = experiment_path,
    folders = ["ad_info"]
)

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

print("-"*50)
print("Creating train dataset")
print("-"*50)

train_datasets = create_window_dataset(
    config = exp_config,
    lifes = train_lifes
)

get_ad_info(dataset=train_datasets, txt_filepath=txt_filepath, data_type="train")

print("-"*50)
print("Creating val dataset")
print("-"*50)

val_datasets = create_window_dataset(
    config = exp_config,
    lifes = val_lifes
)

get_ad_info(dataset=val_datasets, txt_filepath=txt_filepath, data_type="val")

print("-"*50)
print("Creating test dataset")
print("-"*50)

test_datasets = create_window_dataset(
    config = exp_config,
    lifes = test_lifes
)

get_ad_info(dataset=test_datasets, txt_filepath=txt_filepath, data_type="test")
