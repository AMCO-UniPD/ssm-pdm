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
from ad_info import (
    load_lifes,
    get_normal_mean_ad_info,
)

experiment_path = os.path.dirname((os.path.realpath(__file__)))
exp_config, model_config, device, exp_name = setup_exp()

print("-"*50)
print("AD dataset info")
print(f"Sequence length: {exp_config.sequence_length}")
print(f"Stride: {exp_config.stride}")
print("-"*50)

txt_filepath = generate_path(
    basepath = experiment_path,
    folders = ["ad_info"]
)

train_lifes, val_lifes, test_lifes = load_lifes(exp_config=exp_config)

print("-"*50)
print("Creating train dataset")
print("-"*50)

train_datasets = create_window_dataset(
    config = exp_config,
    lifes = train_lifes
)

# get_mean_ad_info(dataset=train_datasets, txt_filepath=txt_filepath, data_type="train")
get_normal_mean_ad_info(dataset=train_datasets, txt_filepath=txt_filepath, data_type="train")

print("-"*50)
print("Creating val dataset")
print("-"*50)

val_datasets = create_window_dataset(
    config = exp_config,
    lifes = val_lifes
)

# get_mean_ad_info(dataset=val_datasets, txt_filepath=txt_filepath, data_type="val")
get_normal_mean_ad_info(dataset=train_datasets, txt_filepath=txt_filepath, data_type="train")

print("-"*50)
print("Creating test dataset")
print("-"*50)

test_datasets = create_window_dataset(
    config = exp_config,
    lifes = test_lifes
)

# get_mean_ad_info(dataset=test_datasets, txt_filepath=txt_filepath, data_type="test")
get_normal_mean_ad_info(dataset=test_datasets, txt_filepath=txt_filepath, data_type="test")
