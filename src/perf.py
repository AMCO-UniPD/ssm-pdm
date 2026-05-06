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
from ceruleo.results.results import (
    PredictionResult,
    FittedLife,
    unexpected_breaks_from_cv,
    unexploited_lifetime_from_cv,
    excessive_life_from_cv,
    metric_J_from_cv
)

from loss import load_loss_functions
from exp_config import ModelConfig

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
    Calculate the metrics for each life in the dataset and save them in a pd.DataFrame

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
    print("-" * 50)
    print(f"Opened outputs_dict at path: {outputs_path}")
    print("-" * 50)
    y_pred, y_true = outputs_dict["y_pred"], outputs_dict["y_true"]

    _, eval_criterion = load_loss_functions(
        loss_name=config.loss,
        eval_loss_name=config.life_eval_loss,
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

        filename = f"{get_current_time()}_lifes_metrics_{config.model_name}_{config.cmapss_models}_{config.eval_loss}" if config.data_name == "CMAPSS" else f"{get_current_time()}_lifes_metrics_{config.model_name}_{config.eval_loss}"

        save_element(
            element=metrics_df,
            dirpath=metrics_path,
            filename=filename,
        )

        print("-"*50)
        print(f"Metrics df saved at {os.path.join(metrics_path,filename)}")
        print("-"*50)

    pd.options.display.float_format = None

    return metrics_df

def lifes_business_metrics(
    config: ExperimentConfig,
    outputs_path: str = experiment_path,
    metrics_path: str = experiment_path,
    is_baseline: bool = False
) -> pd.DataFrame:
    """
    Clone of lifes_metrics function to compute the business metrics
    on the different test lifes

    Args:
        config (dict): ExperimentConfig object
        outputs_path (str): Path to the outputs
        metrics_path (str): Path to save the metrics
        is_baseline (bool): weather there is a baseline model

    Returns:
        pd.DataFrame business metrics DataFrame
    """

    outputs_dicts = []

    start_run_id = 0 if is_baseline else config.start_run_id
    n_runs = 1 if is_baseline else config.n_runs

    for run in range(start_run_id, n_runs):

        run_outputs_path = generate_path(
            basepath = outputs_path,
            folders = [f"run_{run+1}"]
        )
        outputs_filepath = get_most_recent_file(run_outputs_path)
        outputs_dict = open_element(outputs_filepath,filetype="pickle")
        outputs_dicts.append(outputs_dict)

    fitted_lifes = [
        [
            FittedLife(
                y_true = outputs_dicts[j]["y_true"][i],
                y_pred = outputs_dicts[j]["y_pred"][i],
                time = np.arange(np.squeeze(outputs_dict["y_pred"][i]).shape[0]),
            )
            for j in range(start_run_id, n_runs)
        ]
        for i in range(len(outputs_dicts[0]["y_true"]))
    ]

    m_ub_values, mean_ub, std_ub = unexpected_breaks_from_cv(
        lives = fitted_lifes,
        window_size = config.max_windows,
        n = config.n_maintenance_windows
    )

    m_ul_values, mean_ul, std_ul = unexploited_lifetime_from_cv(
        lives = fitted_lifes,
        window_size = config.max_windows,
        n = config.n_maintenance_windows
    )

    m_el_values, mean_el, std_el = excessive_life_from_cv(
        lives = fitted_lifes,
        window_size = config.max_windows,
        n = config.n_maintenance_windows
    )

    m_J_values, J = metric_J_from_cv(
        lives = fitted_lifes,
        window_size = config.max_windows,
        n = config.n_maintenance_windows,
        c_ub = 10.0,
        c_ul = 1.0
    )

    metrics_dict = {
        "M values": np.round(m_ub_values,2),
        "Unexpected Breaks": mean_ub,
        "Unexpected Breaks std": std_ub,
        "Unexploited Lifetime": mean_ul,
        "Unexploited Lifetime std": std_ul,
        "Excessive Life": mean_el,
        "Excessive Life std": std_el,
        "Metric J": J,
    }
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.set_index("M values")

    if config.print_mean_metrics_df:

        print("-"*50)
        print(metrics_df.to_markdown())
        print("-"*50)

    if config.save_metrics_df:

        filename = f"{get_current_time()}_business_lifes_metrics_{config.model_name}"

        save_element(
            element=metrics_df,
            dirpath=metrics_path,
            filename=filename,
        )

        print("-"*50)
        print(f"Metrics df saved at {os.path.join(metrics_path,filename)}")
        print("-"*50)

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

