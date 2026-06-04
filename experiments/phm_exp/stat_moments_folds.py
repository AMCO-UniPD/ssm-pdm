"""
Python script to print the statistical moments for different folds
"""

# general imports
import ipdb
import os
import sys
import setproctitle
import wandb

from sklearn.model_selection import KFold

src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.append(src_path)

from exp_config import setup_exp
from utils import (
    MergeData,
    transform_phm_data,
    get_raw_phm_data,
)

experiment_path = os.path.dirname((os.path.realpath(__file__)))
exp_config, model_config, device, exp_name = setup_exp()

skf = KFold(
    n_splits = exp_config.n_folds,
    shuffle = True,
    random_state = 42
)

train_phm_data, test_phm_data = get_raw_phm_data(config=exp_config)
merged_phm_data = MergeData(data_list=[train_phm_data])

for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(merged_phm_data)):

    print("-"*50)
    print(f"Processing fold {fold_idx+1}/{exp_config.n_folds}")
    print("-"*50)

    train_val_data, test_data = merged_phm_data[train_val_idx], merged_phm_data[test_idx]

    print("-"*50)
    print(f"Computing statistical moments for fold {fold_idx+1}/{exp_config.n_folds}")
    print("-"*50)

    transform_phm_data(
        config = exp_config,
        train_data = train_val_data,
        test_data = test_data
    )
