"""
Python script containing the plotting functions for the chronos pdm project
"""

import os
import traceback
import sys
import ipdb
from typing import List
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
import torch

src_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "src",
)
sys.path.append(src_path)

cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
experiment_path = os.path.join(cwd, "experiments", "chronos_exp")

from utils import (
    ExperimentConfig,
    an_score_to_rul,
    get_current_time,
    get_most_recent_file,
    open_element,
    get_feature_names,
)
from config_vars import PHM_FEATURES


def plot_forecast(
    life: pd.DataFrame,
    prompt: np.ndarray,
    quantile_levels: List[float],
    pred_quantiles: torch.Tensor,
    prediction_length: int = 12,
    sensor_num: int = 4,
    data_name: str = "CMAPSS",
    plot_path: str = os.getcwd(),
) -> plt.Figure:
    """
    Function to plot the forecast of the life

    Parameters:
    -----------
    life: pd.DataFrame
        Dataframe containing the life data
    prompt: np.ndarray
        Array containing the prompt data
    quantile_dict: List[float]
        Dictionary containing the quantile levels and the quantiles tensors
    prediction_length: int
        Number of steps to predict
    data_name: str
        Name of the dataset
    plot_path: str
        Path to save the plot

    Returns:
    --------
    fig: plt.Figure
        Figure containing the plot
    """

    forecast_index = range(len(life) - prediction_length, len(life))

    low_idx = np.argmin(quantile_levels)
    high_idx = np.argmax(quantile_levels)
    low, median, high = (
        pred_quantiles[0, :, low_idx],
        pred_quantiles[0, :, 1],
        pred_quantiles[0, :, high_idx],
    )

    plt.figure(figsize=(8, 4))
    plt.plot(prompt, color="royalblue", label="historical data")
    plt.plot(forecast_index, median, color="tomato", label="median forecast")
    plt.fill_between(
        forecast_index,
        low,
        high,
        color="tomato",
        alpha=0.3,
        label="80% prediction interval",
    )
    plt.legend()
    plt.grid()

    sensor_name = f"SensorMeasure{sensor_num}"
    filename = f"{get_current_time()}_{data_name}_{sensor_name}_forecast.png"
    plot_path = os.path.join(plot_path, filename)

    plt.savefig(plot_path)
    print(f"Plot saved at: {plot_path}")


# Re adaptation of function plot_torch_predictions_grid from `SSM_PDM` project


def plot_predictions_grid(
    config: ExperimentConfig,
    outputs_path: str = experiment_path,
    plot_path: str = experiment_path,
) -> plt.figure:
    """
    Function to plot in a grid the `RUL` prediction of each life for a specific
    feature/sensor

    Parameters:
    -----------
    config: ExperimentConfig
        Experiment configuration object
    outputs_path: str
        Path to the outputs dictionary
    plot_path: str
        Path to save the plot

    Returns:
    --------
    fig: plt.figure
        Figure containing the plot
    """

    if config.life_idx is None:
        config.life_idx = np.arange(config.nrows * config.ncols)
    else:
        assert config.nrows * config.ncols == len(config.life_idx), (
            "Number of rows and columns must match the number of lives"
        )

    # Get the name of the sensor to plot
    feature_names = get_feature_names(config)

    # Get the y_pred and y_true tensors
    outputs_path = get_most_recent_file(outputs_path, file_pos=config.file_pos)
    outputs_dict = open_element(file_path=outputs_path, filetype="pickle")

    # Select the predictions and true values for the sensor
    y_pred, y_true = outputs_dict["y_pred"], outputs_dict["y_true"]
    if config.model_name.startswith("chronos"):
        pred, true = (
            y_pred[config.life_idx][:, sensor_idx, :],
            y_true[config.life_idx][:, sensor_idx, :],
        )
    elif config.approach == "padding":
        pred, true = y_pred[config.life_idx, :], y_true[config.life_id, :]
    elif config.approach == "windowed":
        pred = [y_pred[i] for i in config.life_idx]
        true = [y_true[i] for i in config.life_idx]
    if not config.full_life:
        mask = (
            true != 0
            if config.approach == "padding"
            else [true[i] != 0 for i in range(len(true))]
        )
    else:
        np.ones(true.shape, dtype=int)

    # Produce the plot
    fig, axs = plt.subplots(config.nrows, config.ncols, figsize=(30, 20))
    for i in range(config.nrows):
        for j in range(config.ncols):
            if i * config.ncols + j < (config.nrows * config.ncols):
                ax = axs[i, j]
                ax.plot(
                    true[i * config.ncols + j][mask[i * config.ncols + j]],
                    color="blue",
                    label="True RUL",
                )
                ax.plot(
                    pred[i * config.ncols + j][mask[i * config.ncols + j]],
                    color="orange",
                    label="Predicted RUL",
                )
                ax.set_title(
                    f"Life {config.life_idx[i * config.ncols + j] + config.cmapss_test_idx[0] + 1}"
                )
                ax.set_xticks([])
                ax.set_ylabel("RUL")
                ax.legend()

    if config.save_plot:
        if config.full_life:
            filename = f"{get_current_time()}_{config.model_name}_{config.cmapss_models}_predictions_grid_full"
        else:
            filename = f"{get_current_time()}_{config.model_name}_{config.cmapss_models}_predictions_grid_pad"
        life_idx_str = "_".join(str(x) for x in config.life_idx)
        filename = f"{filename}_life_{life_idx_str}.pdf"
        plot_path = os.path.join(plot_path, filename)
        plt.savefig(plot_path, bbox_inches="tight")
        print("#" * 50)
        print(f"Plot saved at: {plot_path}")
        print("#" * 50)

    return fig


# Plot function to plot the prediction intervals of the model using the
# predictions from different quantile levels


def plot_prediction_interval(
    config: ExperimentConfig,
    outputs_path: str = experiment_path,
    plot_path: str = experiment_path,
    n_last_samples: int = 0,
) -> None:
    """
    Function to plot in a grid the `RUL` predictions of each life compared to the true `RUL`,M
    The prediction on different quantiles will be used to create some prediction intervals.

    Parameters:
    -----------
    config: ExperimentConfig
        Experiment configuration object
    outputs_path: str
        Path to the outputs dictionary
    plot_path: str
        Path to save the plot
    n_last_samples: int
        Number of last samples to show. If you pass 500 it will show the last 500 samples. By default it's 0 meaning that all the samples are shown

    Returns:
    --------
        None: the function produces the plot but does not return anything
    """

    assert config.nrows * config.ncols == len(config.life_idx), (
        "Number of rows and columns must match the number of lives"
    )

    # Get the outputs dictionary
    outputs_path = get_most_recent_file(outputs_path, file_pos=config.file_pos)
    outputs_dict = open_element(file_path=outputs_path, filetype="pickle")

    y_true = outputs_dict["y_true"]
    life_idxs = [np.where(config.test_idx == x)[0][0] for x in config.life_idx]
    if n_last_samples > 0:
        true = [y_true[i][-n_last_samples:] for i in life_idxs]
    else:
        true = [y_true[i] for i in life_idxs]

    quantile_signals = {}
    cmap = plt.get_cmap("Set1")
    colors = cmap.colors

    if not config.full_life:
        mask = [true[i] != 0 for i in range(len(true))]
    else:
        mask = [np.ones(len(t), dtype=bool) for t in true]

    # Produce the plot
    if config.nrows == config.ncols == 1:
        fig, axs = plt.subplots(config.nrows, config.ncols, figsize=(10, 8))
    else:
        fig, axs = plt.subplots(config.nrows, config.ncols, figsize=(50, 20))

    for i in range(config.nrows):
        for j in range(config.ncols):
            if i * config.ncols + j < (config.nrows * config.ncols):
                if config.nrows == 1 and config.ncols == 1:
                    ax = axs
                elif config.nrows == 1:
                    ax = axs[j]
                elif config.ncols == 1:
                    ax = axs[i]
                else:
                    ax = axs[i, j]

                ax.plot(
                    np.squeeze(true[i * config.ncols + j],axis=0) if "padding" in config.approach else true[i * config.ncols +j],
                    color="#00008B",
                    label="True RUL",
                )

                for quantile,color in zip(config.quantiles,colors):
                    quantile_signals[f"pred_quantile_{quantile}"] = [
                        np.squeeze(outputs_dict[f"pred_quantile_{quantile}"][i][-n_last_samples:],axis=0) if "padding" in config.approach else outputs_dict[f"pred_quantile_{quantile}"][i][-n_last_samples:]
                        for i in life_idxs
                    ]

                    ax.plot(
                        quantile_signals[f"pred_quantile_{quantile}"][i * config.ncols + j],
                        color=color,
                        label=f"Predicted RUL {quantile}",
                    )

                plot_title = (
                    f"Life {config.life_idx[i * config.ncols + j]}"
                    if config.data_name == "CMAPSS"
                    else f"Life {config.life_idx[i * config.ncols + j]}"
                )
                ax.set_title(plot_title)
                ax.set_xticks([])
                ax.set_ylabel("RUL")
                ax.legend()

                # Use plt.fill_between to create the prediction interval using predictions
                # from the max and min quantile levels contained in quantiles

                if len(config.quantiles) == 2:
                    quantile_min, quantile_max = (
                        np.min(config.quantiles),
                        np.max(config.quantiles),
                    )

                    ax.fill_between(
                        np.arange(len(true[i * config.ncols + j][mask[i * config.ncols + j]])),
                        quantile_signals[f"pred_quantile_{quantile_min}"][i * config.ncols + j],
                        quantile_signals[f"pred_quantile_{quantile_max}"][i * config.ncols + j],
                        color="#ADD8E6",
                        alpha=0.5,
                        label="Prediction Interval {quantile_min}-{quantile_max}",
                    )


    if config.show_plot:
        print("-" * 50)
        print("Showing the plot")
        print("-" * 50)
        plt.show()

    if config.save_plot:
        if config.full_life:
            filename = (
                f"{get_current_time()}_{config.model_name}_{config.cmapss_models}_run_{config.run_id}_interval_plot_full_life"
                if config.data_name == "CMAPSS"
                else f"{get_current_time()}_{config.model_name}_run_{config.run_id}_interval_plot_full_life"
            )
        else:
            filename = (
                f"{get_current_time()}_{config.model_name}_{config.cmapss_models}_run_{config.run_id}_interval_plot"
                if config.data_name == "CMAPSS"
                else f"{get_current_time()}_{config.model_name}_run_{config.run_id}_interval_plot"
            )

        life_idx_str = "_".join(str(x) for x in config.life_idx)
        filename = f"{filename}_life_{life_idx_str}_last_{n_last_samples}_samples.png"
        plot_path = os.path.join(plot_path, filename)
        plt.savefig(plot_path, bbox_inches="tight")
        print("#" * 50)
        print(f"Plot saved at: {plot_path}")
        print("#" * 50)


def plot_an_scores(
    config: ExperimentConfig,
    outputs_path: str = experiment_path,
    plot_path: str = experiment_path,
    n_last_samples: int = 0,
) -> None:
    """
    Function to plot the anomaly scores over the different samples.

    Parameters:
        config (ExperimentConfig): experiment configuration object
        outputs_path (str): path where the outputs are saved
        plot_path (str): path where to save the plot
        n_last_samples (int): number of samples to show. If 0 all the life is shown, otherwise
        the n_last_samples samples are shown

    Returns:
        None: the function produces the plot but does not return anything
    """

    assert config.nrows * config.ncols == len(config.life_idx), (
        "Number of rows and columns must match the number of lives"
    )

    # Get the outputs dictionary
    outputs_path = get_most_recent_file(outputs_path, file_pos=config.file_pos)
    outputs_dict = open_element(file_path=outputs_path, filetype="pickle")

    life_idxs = [np.where(config.test_idx == x)[0][0] for x in config.life_idx]
    mean_quantile_0_5 = [
        outputs_dict["mean_an_scores_quantile_0.5"][i][-n_last_samples:]
        for i in life_idxs
    ]
    max_quantile_0_5 = [
        outputs_dict["max_an_scores_quantile_0.5"][i][-n_last_samples:]
        for i in life_idxs
    ]

    # Produce the plot
    if config.nrows == config.ncols == 1:
        fig, axs = plt.subplots(config.nrows, config.ncols, figsize=(10, 8))
    else:
        fig, axs = plt.subplots(config.nrows, config.ncols, figsize=(50, 20))

    for i in range(config.nrows):
        for j in range(config.ncols):
            if i * config.ncols + j < (config.nrows * config.ncols):
                if config.nrows == 1 and config.ncols == 1:
                    ax = axs
                elif config.nrows == 1:
                    ax = axs[j]
                elif config.ncols == 1:
                    ax = axs[i]
                else:
                    ax = axs[i, j]

                ax.plot(
                    mean_quantile_0_5[i * config.ncols + j],
                    color="orange",
                    label="Mean Anomaly Score quantile 0.5",
                )

                ax.plot(
                    max_quantile_0_5[i * config.ncols + j],
                    color="blue",
                    label="Max Anomaly Score quantile 0.5",
                )

                # ax.plot(
                #     rul_quantile_0_5[i * config.ncols + j],
                #     color="green",
                #     label="RUL from Anomaly Score quantile 0.5",
                # )

                plot_title = (
                    f"Life {config.life_idx[i * config.ncols + j]}"
                    if config.data_name == "CMAPSS"
                    else f"Life {config.life_idx[i * config.ncols + j]}"
                )
                ax.set_title(plot_title)
                ax.set_xticks([])
                ax.set_ylabel("Anomaly Score")
                ax.legend()

    if config.save_plot:
        filename = (
            f"{get_current_time()}_{config.model_name}_{config.cmapss_models}_run_{config.run_id}_an_scores_plot"
            if config.data_name == "CMAPSS"
            else f"{get_current_time()}_{config.model_name}_run_{config.run_id}_an_scores_plot"
        )

        life_idx_str = "_".join(str(x) for x in config.life_idx)
        filename = f"{filename}_life_{life_idx_str}_last_{n_last_samples}_samples.png"
        plot_path = os.path.join(plot_path, filename)
        plt.savefig(plot_path, bbox_inches="tight")
        print("-" * 50)
        print(f"Plot saved at: {plot_path}")
        print("-" * 50)


def plot_combined_signals(
    config: ExperimentConfig,
    outputs_path: str = experiment_path,
    plot_path: str = experiment_path,
    col_idx: int = 0,
    n_last_samples: int = 0,
) -> None:
    """
    Function to plot the predicted and true signals for a specific column.

    Parameters:
        config (ExperimentConfig): experiment configuration object
        outputs_path (str): path where the outputs are saved
        plot_path (str): path where to save the plot
        col_idx (int): index of the feature to plot
        n_last_samples (int): number of samples to show. If 0 all the life is shown, otherwise
        the n_last_samples samples are shown

    Returns:
        None: the function produces the plot but does not return anything
    """

    assert config.nrows * config.ncols == len(config.life_idx), (
        "Number of rows and columns must match the number of lives"
    )

    # Get the outputs dictionary
    outputs_path = get_most_recent_file(outputs_path, file_pos=config.file_pos)
    outputs_dict = open_element(file_path=outputs_path, filetype="pickle")

    life_idxs = [np.where(config.test_idx == x)[0][0] for x in config.life_idx]
    column = PHM_FEATURES[col_idx]
    y_true = [outputs_dict["y_true"][i][-n_last_samples:, col_idx] for i in life_idxs]

    y_pred_quantile_0_5 = [
        outputs_dict[f"pred_quantile_0.5"][i][-n_last_samples:, col_idx]
        for i in life_idxs
    ]

    # Produce the plot
    if config.nrows == config.ncols == 1:
        _, axs = plt.subplots(config.nrows, config.ncols, figsize=(10, 8))
    else:
        _, axs = plt.subplots(config.nrows, config.ncols, figsize=(50, 20))

    for i in range(config.nrows):
        for j in range(config.ncols):
            if i * config.ncols + j < (config.nrows * config.ncols):
                if config.nrows == 1 and config.ncols == 1:
                    ax = axs
                elif config.nrows == 1:
                    ax = axs[j]
                elif config.ncols == 1:
                    ax = axs[i]
                else:
                    ax = axs[i, j]

                ax.plot(
                    y_true[i * config.ncols + j],
                    color="blue",
                    label="True Signal",
                )

                ax.plot(
                    y_pred_quantile_0_5[i * config.ncols + j],
                    color="orange",
                    label="Predicted Signal quantile 0.5",
                )

                plot_title = (
                    f"Life {config.life_idx[i * config.ncols + j]}"
                    if config.data_name == "CMAPSS"
                    else f"Life {config.life_idx[i * config.ncols + j]}"
                )
                ax.set_title(plot_title)
                ax.set_xticks([])
                ax.set_ylabel(f"Signal Column {column}")
                ax.legend()

    if config.save_plot:
        filename = (
            f"{get_current_time()}_{config.model_name}_{config.cmapss_models}_run_{config.run_id}_{column}_plot"
            if config.data_name == "CMAPSS"
            else f"{get_current_time()}_{config.model_name}_run_{config.run_id}_signals_col_{column}_plot"
        )

        life_idx_str = "_".join(str(x) for x in config.life_idx)
        filename = f"{filename}_life_{life_idx_str}_last_{n_last_samples}_samples.png"
        plot_path = os.path.join(plot_path, filename)
        plt.savefig(plot_path, bbox_inches="tight")
        print("-" * 50)
        print(f"Plot saved at: {plot_path}")
        print("-" * 50)


# Blob plot


def blob_plot(
    plot_dict: dict,
    config: ExperimentConfig,
    plot_path: str,
    no_mult_adds: bool = False,
    no_test_time: bool = False,
) -> plt.figure:
    """
    Function to produce a plot that represents the number of parameters, number of mult-adds,
    and test metric for each model in the experiment.

    Args:
    plot_dict (dict): dictionary with the number of parameters, mult-adds, model names and test metrics
    config (ExperimentConfig): experiment configuration
    plot_path (str): path to save the plot
    no_mult_adds (bool): if set the mult_adds metric will not be used to produce the blob plot, by default False
    no_test_time (bool): if set the inference time will not be considered as a metric to produce the blob plot, by default False

    Returns:
    plt.figure: figure with the blob plot
    """

    # Convert plot_dict into a pandas DataFrame
    df = pd.DataFrame(plot_dict)
    df = df.rename(
        columns={
            "params_float": "Parameters (K)",
            "test_metric": "Test Loss",
            "pickle_size_kb": "Model Size (kB)",
        }
    )

    if not no_mult_adds:
        plot_dict["mult_adds_ln"] = np.abs(np.log(plot_dict["mult_adds_float"]))
        blob_size = "Log Mult-Adds"
        df = df.rename(
            columns={
                "mult_adds_float": "Mult-Adds (MMACs)",
                "mult_adds_ln": "Log Mult-Adds",
            }
        )
    elif not no_test_time:
        blob_size = "Inference Time (s)"
        df = df.rename(
            columns={
                "test_time": "Inference Time (s)",
            }
        )
    else:
        blob_size = "Model Size (kB)"

    fig = px.scatter(
        data_frame=df,
        x="Parameters (K)",
        y="Test Loss",
        color=blob_size,
        size=blob_size,
        hover_name="model_name",
        log_x=True,
        size_max=60,
        color_continuous_scale="Viridis",
        text="model_name",
        opacity=0.9,
    )
    fig.update_traces(
        textposition="bottom center",
        textfont=dict(size=10),
    )

    if config.save_plot:
        filename = f"{get_current_time()}_blob_plot_{config.cmapss_models}_quantile_{config.quantile_run}_{config.quantile_approach}_{config.eval_loss}.png"
        plot_path = os.path.join(plot_path, filename)
        fig.write_image(plot_path, scale=3)
        print("#" * 50)
        print(f"Plot saved at: {plot_path}")
        print("#" * 50)

    return fig
