"""
Python script containing some performance evaluation functions to use
inside `best_model_perf`
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

from ssm_models import load_ssm_model, ModelConfig

cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
experiment_path = os.path.join(cwd, "experiments", "chronos_exp")


def lifes_metrics(
    config: ExperimentConfig,
    outputs_path: str = experiment_path,
    metrics_path: str = experiment_path,
    compute_stats: bool = False,
    tau: float = 0.5
) -> pd.DataFrame:
    """
    Calculate the metrics for each life and each sensor in the dataset and save them in a pd.DataFrame

    Args:
        config:dict ExperimentConfig object
        outputs_path:str Path to the outputs
        metrics_path:str Path to save the metrics
        compute_stats:bool Whether to compute the mean,median and std of the metrics
        tau (float): quantile level

    Returns:
        pd.DataFrame Metrics DataFrame
    """

    metrics_df = pd.DataFrame()

    outputs_path = get_most_recent_file(outputs_path, file_pos=config.file_pos)
    outputs_dict = open_element(file_path=outputs_path, filetype="pickle")
    print("#" * 50)
    print(f"Opened outputs_dict at path: {outputs_path}")
    print("#" * 50)
    y_pred, y_true = outputs_dict["y_pred"], outputs_dict["y_true"]

    _, eval_criterion = load_loss_functions(
        loss_name=config.loss,
        eval_loss_name=config.eval_loss,
        tau=tau,
    )

    pd.options.display.float_format = "{:.2f}".format

    for i in range(len(y_pred)):
        mask = torch.tensor(y_true[i] != 0)
        pred = torch.tensor(y_pred[i])
        true = torch.tensor(y_true[i])
        eval_loss = eval_criterion(y_pred=pred, y_true=true, mask=mask).item()
        life_name = f"Life_{i+config.test_idx[0]+1}" if config.data_name == "CMAPSS" else f"Life_{config.test_idx[i]}"
        metrics_df.at[life_name, "Eval Loss"] = round(eval_loss, 2)

    if compute_stats:
        # Add a row Life_mean with the mean of the metrics over all the columns
        metrics_df.loc["Life_mean"] = metrics_df.mean(axis=0).round(2)
        # Add a row Life_median with the median of the metrics over all the columns
        metrics_df.loc["Life_median"] = metrics_df.median(axis=0).round(2)
        # Add a row Life_std with the std of the metrics over all the columns
        metrics_df.loc["Life_std"] = metrics_df.std(axis=0).round(2)

        print("#" * 50)
        print(f"Mean eval loss over all the test lifes: {metrics_df.loc['Life_mean']}")
        print("#" * 50)
        print(f"Median eval loss over all the test lifes: {metrics_df.loc['Life_median']}")
        print("#" * 50)
        print(f"Std eval loss over all the test lifes: {metrics_df.loc['Life_std']}")
        print("#" * 50)


    if config.save_metrics_df:

        filename = f"{get_current_time()}_lifes_metrics_{config.model_name}_{config.cmapss_models}_{config.eval_loss}" if config.data_name == "CMAPSS" else f"{get_current_time()}_lifes_metrics_{config.model_name}_{config.tool_type}_{config.eval_loss}"

        save_element(
            element=metrics_df,
            dirpath=metrics_path,
            filename=filename,
        )

    pd.options.display.float_format = None

    return metrics_df


# Function to select a subset of the rows and a subset of the columns
# of a metrics_df


def sub_lifes_metrics(
    config: ExperimentConfig,
    metrics_df: pd.DataFrame,
    compute_stats: bool = False,
) -> pd.DataFrame:
    """
    Select a subset of the rows and a subset of the columns of a metrics_df

    Args:
        config:ExperimentConfig ExperimentConfig object
        metrics_df:pd.DataFrame Metrics DataFrame
        compute_stats:bool Whether to compute the mean,median and std of the metrics

    Returns:
        sub_metrics_df (pd.DataFrame): Subset of the metrics_df
    """

    if config.metrics_idx is None:
        config.metrics_idx = np.arange(config.n_metrics_lifes)
    metrics_idx = [f"Life_{i+config.test_idx[0]+1}" for i in config.metrics_idx]
    if config.model_name.startswith("chronos"):
        sub_metrics_df = metrics_df.loc[metrics_idx, config.metrics_cols]
    else:
        sub_metrics_df = metrics_df.loc[metrics_idx]

    if compute_stats:
        sub_metrics_df.loc["Life_mean"] = sub_metrics_df.mean(axis=0).round(2)
        sub_metrics_df.loc["Life_median"] = sub_metrics_df.median(axis=0).round(2)
        sub_metrics_df.loc["Life_std"] = sub_metrics_df.std(axis=0).round(2)
        if len(config.metrics_cols) > 1 and config.model_name.startswith("chronos"):
            sub_metrics_df["Sensor_mean"] = sub_metrics_df.mean(axis=1).round(2)
            sub_metrics_df["Sensor_median"] = sub_metrics_df.median(axis=1).round(2)
            sub_metrics_df["Sensor_std"] = sub_metrics_df.std(axis=1).round(2)

    return sub_metrics_df


# Function to render the data contained in a pd.DataFrame into a markdown table


def df_to_obsidian_table(df):
    markdown = "| " + " | ".join(df.columns) + " |\n"
    markdown += "| " + " | ".join("---" for _ in df.columns) + " |\n"
    for row in df.itertuples(index=False):
        markdown += "| " + " | ".join(map(str, row)) + " |\n"
    return markdown


def df_with_index_to_obsidian_table(df):
    lines = []
    # Include index column name as the first header
    lines.append("| " + " | ".join([df.index.name or ""] + list(df.columns)) + " |")
    lines.append("| " + " | ".join(["---"] * (len(df.columns) + 1)) + " |")
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + list(map(str, row))) + " |")
    return "\n".join(lines)


# Function to print the summary metrics of a metrics_df dataframe


def print_summary_metrics(metrics_df: pd.DataFrame, model_name: str = "S4") -> None:
    """
    Print the summary metrics of a metrics_df dataframe

    Args:
        metrics_df:pd.DataFrame Metrics DataFrame
        model_name:str Model name

    Returns:
        None: The function computes and prints the summary metrics and does not return anything
    """

    # Compute the mean, median and std of the metrics
    metrics_df.loc["Life_mean"] = metrics_df.mean(axis=0).round(2)
    metrics_df.loc["Life_median"] = metrics_df.median(axis=0).round(2)
    metrics_df.loc["Life_std"] = metrics_df.std(axis=0).round(2)

    print("#" * 50)
    print(f"Summary metrics for model {model_name}")
    print("#" * 50)
    print(f"Mean eval loss over all the test lifes:\n{metrics_df.loc['Life_mean']}")
    print("#" * 50)
    print(f"Median eval loss over all the test lifes:\n{metrics_df.loc['Life_median']}")
    print("#" * 50)
    print(f"Std eval loss over all the test lifes:\n{metrics_df.loc['Life_std']}")
    print("#" * 50)


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
