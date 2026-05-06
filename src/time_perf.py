"""
Python module containing functions to perform time
and memory performance experiments
"""

import os
import sys
import timeit
import ipdb
import torch
import pandas as pd
import numpy as np

# torch imports
import torch
import torch.nn as nn

chronos_path_src = os.path.join(os.path.dirname(__file__), "chronos-rul", "src")
sys.path.append(chronos_path_src)

from utils import (
    ExperimentConfig,
    get_current_time,
    generate_path,
    get_feature_names,
    get_phm_feature_names,
    get_most_recent_dir,
    save_element,
    open_element,
    get_most_recent_file,
    set_seed,
)

from loss import load_loss_functions

from ssm_models import load_ssm_model
from exp_config import ModelConfig

cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
experiment_path = os.path.join(cwd, "experiments", "phm_exp")

def time_exp(
    config: ExperimentConfig,
    model_config: ModelConfig,
) -> dict:
    """
    This function computes the execution time for a call to test_loop
    for a specific model to have an estimate of its inference time.

    Args:
        config (ExperimentConfig): experiment configuration
        model_config (ModelConfig): model configuration

    Returns:
        dict_time (dict): dictionary containing inference times of the model over different runs
    """

    # Initialize time dictionary
    dict_time = {}
    dict_time["test_time"] = []
    # Define dimensions of the tensor
    batch_size, sequence_length = 32, 170
    feature_names = get_feature_names(config)
    d_input = (
        len(feature_names)
        if ((not config.quantile_reg) or (not model_config.tau_feat))
        else len(feature_names) + 1,
    )
    d_input = d_input[0]

    device = torch.device(
        f"cuda:{config.device_num}" if torch.cuda.is_available() else "cpu"
    )
    model_config.device = device

    # Load the model

    model, _, _ = load_ssm_model(
        model_config=model_config, exp_config=config, d_input=d_input
    )
    model = model.to(device)

    set_seed(seed=0)

    with torch.no_grad():
        # Create a random input tensor with the same size used in the project
        x = torch.rand(batch_size, sequence_length, d_input).to(device)
        test_time = timeit.timeit(lambda: model(x, tau=0.5), number=config.n_runs_time)
    dict_time["test_time"] = test_time

    avg_time = np.round(dict_time["test_time"] / config.n_runs_time, 3)
    dict_time["avg_time"] = avg_time
    print("#" * 50)
    print(f"Inference time for current model: {avg_time}")
    print("#" * 50)

    return dict_time


def state_dict_size(
    config: ExperimentConfig,
    basepath: str = experiment_path,
    model_name: str = "S4",
    run_id: int = 1,
    tau: float = 0.5,
) -> tuple[dict, str]:
    """
    Function to return the filesize of the `pickle` file containing the state dict of a model, to use as a metric to compare the sizes of different models

    Args:
        config (ExperimentConfig): experiment configuration
        basepath (str): starting path for defining the path where to find the `pickle` file, by default experiment_path
        model_name (str): name of the model, by default S4
        run_id (int): select a run id for multi run experiments
        tau (float): select a quantile level

    Returns:
        size_dict (dict): dictionary containing the file size
        state_dict_path (str): path to the pickle file
    """

    assert run_id <= 5, "The maximum run number is 5"
    assert (
        tau in config.quantiles
    ), f"The quantile level must be inside {config.quantiles}"

    size_dict = {}
    state_dict_dirpath = generate_path(
        basepath=basepath,
        folders=[
            "best_models",
            model_name,
            config.cmapss_models,
            config.approach,
            "quantile_reg",
        ],
    )
    state_dict_exp_dirpath = get_most_recent_dir(
        state_dict_dirpath, file_pos=config.file_pos_pickle
    )

    state_dict_quantile_dirpath = generate_path(
        basepath=state_dict_exp_dirpath,
        folders=[
            f"run_{run_id}",
            f"quantile_{tau}",
        ],
    )

    state_dict_path = get_most_recent_file(
        state_dict_quantile_dirpath, file_pos=config.file_pos
    )

    pickle_size = os.path.getsize(state_dict_path)
    size_dict["pickle_size"] = pickle_size
    size_dict["pickle_size_kb"] = pickle_size / 1024
    size_dict["pickle_size_mb"] = pickle_size / (1024**2)
    print("#" * 50)
    print(
        f"Size of the pickled state dict for model {model_name}: {size_dict['pickle_size']} bytes"
    )
    print(
        f"Size of the pickled state dict for model {model_name}: {size_dict['pickle_size_kb']} KB"
    )
    print(
        f"Size of the pickled state dict for model {model_name}: {size_dict['pickle_size_mb']} MB"
    )
    print("#" * 50)
    return size_dict, state_dict_path

