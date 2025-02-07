"""
Python script containing the plotting functions for the chronos pdm project 
"""

import os 
import sys
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

src_path = os.path.join(os.path.dirname(__file__),"..","..","src",
)
sys.path.append(src_path)

from utils import get_current_time

def plot_forecast(life:pd.DataFrame,
                  prompt:np.ndarray,
                  quantile_levels:List[float],
                  pred_quantiles:torch.Tensor,
                  prediction_length:int=12,
                  sensor_num:int=4,
                  data_name:str="CMAPSS",
                  plot_path:str=os.getcwd()) -> plt.Figure:

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

    forecast_index = range(len(life)-prediction_length,len(life))

    low_idx=np.argmin(quantile_levels)
    high_idx=np.argmax(quantile_levels)
    low, median, high = pred_quantiles[0, :, low_idx], pred_quantiles[0, :, 1], pred_quantiles[0, :, high_idx]

    plt.figure(figsize=(8, 4))
    plt.plot(prompt, color="royalblue", label="historical data")
    plt.plot(forecast_index, median, color="tomato", label="median forecast")
    plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
    plt.legend()
    plt.grid()

    sensor_name=f"SensorMeasure{sensor_num}"
    filename=f"{get_current_time()}_{data_name}_{sensor_name}_forecast.png"
    plot_path=os.path.join(plot_path,filename)

    plt.savefig(plot_path)
    print(f"Plot saved at: {plot_path}")


