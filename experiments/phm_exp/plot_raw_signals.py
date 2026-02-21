"""
Python script to plot raw signals from PHM dataset lifes in a grid format.
Each column represents a single life, with rows showing different features (excluding RUL).
"""

import ipdb
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.append(src_path)

from exp_config import setup_exp
from ad_info import load_lifes
from utils import generate_path, get_current_time
from config_vars import PHM_FEATURES

experiment_path = os.path.dirname((os.path.realpath(__file__)))
exp_config, model_config, device, exp_name = setup_exp()

plot_path = generate_path(
    basepath=experiment_path,
    folders=[
        "plots",
        exp_config.failure_type,
        "raw_signals",
    ],
)

train_lifes, val_lifes, test_lifes = load_lifes(exp_config=exp_config)

if hasattr(exp_config, "use_test_set") and exp_config.use_test_set:
    lifes = test_lifes
    data_split = "test"
else:
    lifes = train_lifes
    data_split = "train"

if exp_config.life_idx is None:
    life_indices = list(
        range(min(exp_config.nrows * exp_config.ncols, lifes.n_time_series))
    )
else:
    life_indices = exp_config.life_idx

lifes = [lifes[i] for i in life_indices]

print("-" * 50)
print("Creating raw signals grid plot")
print(f"Data split: {data_split}")
print(f"Life indices: {life_indices}")
print("-" * 50)

for feature in PHM_FEATURES:

    print("-"*50)
    print(f"Producing raw signal for feature {feature}")
    print("-"*50)

    fig, axs = plt.subplots(exp_config.nrows, exp_config.ncols, figsize=(50, 20))

    for i in range(exp_config.nrows):
        for j in range(exp_config.ncols):
            ax = axs[i, j]

            ax.plot(
                lifes[i*exp_config.ncols+j].loc[:,feature],
                color = "blue",
                label = "Raw Signal"
            )

            plot_title = f"Life {life_indices[i*exp_config.ncols+j]}"
            ax.set_title(plot_title)
            ax.set_xticks([])
            ax.set_ylabel(feature)
            ax.legend()

    if exp_config.save_plot:
        filename = f"{get_current_time()}_raw_signals_{feature}_{data_split}.png"
        filepath = os.path.join(plot_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print("-" * 50)
        print(f"Plot saved at: {filepath}")
        print("-" * 50)

