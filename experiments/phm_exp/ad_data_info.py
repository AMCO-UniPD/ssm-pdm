"""
Python script to get information on the AD dataset extracted from PHM dataset
"""

# general imports
import os
import sys
import numpy as np
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
    PHM_PATH,
    PHM_FAILURES,
    MAX_RUL
)
from utils import (
    get_transformer,
    TransData,
    SSMWindowRegressionDataset,
)

experiment_path = os.path.dirname((os.path.realpath(__file__)))
exp_config, model_config, device, exp_name = setup_exp()

print("-"*50)
print("AD dataset info")
print(f"Sequence length: {exp_config.sequence_length}")
print(f"Stride: {exp_config.stride}")
print("-"*50)

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

train_datasets = SSMWindowRegressionDataset(
    lifes=train_lifes,
    sequence_length=exp_config.sequence_length,
    stride=exp_config.stride,
    max_rul = MAX_RUL,
    normalize_rul = exp_config.normalize_rul
)
train_datasets.select_windows(n_const_win=exp_config.n_const_win)

print("-"*50)
print("Creating val dataset")
print("-"*50)

val_datasets = SSMWindowRegressionDataset(
    lifes=val_lifes,
    sequence_length=exp_config.sequence_length,
    stride=exp_config.stride,
    max_rul = MAX_RUL,
    normalize_rul = exp_config.normalize_rul
)
val_datasets.select_windows(n_const_win=exp_config.n_const_win)

print("-"*50)
print("Creating test dataset")
print("-"*50)

test_datasets = SSMWindowRegressionDataset(
    lifes=test_lifes,
    sequence_length=exp_config.sequence_length,
    stride=exp_config.stride,
    max_rul = MAX_RUL,
    normalize_rul = exp_config.normalize_rul
)
test_datasets.select_windows(n_const_win=exp_config.n_const_win)
