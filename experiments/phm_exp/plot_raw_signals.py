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
    life_indices = exp_config.life_idx[: exp_config.ncols]

features = (
    PHM_FEATURES[: exp_config.nrows]
    if exp_config.nrows <= len(PHM_FEATURES)
    else PHM_FEATURES
)
ipdb.set_trace()

print("-" * 50)
print(f"Creating raw signals grid plot")
print(f"Data split: {data_split}")
print(f"Number of lives: {len(life_indices)}")
print(f"Number of features: {len(features)}")
print(f"Life indices: {life_indices}")
print(f"Features: {features}")
print("-" * 50)

fig, axs = plt.subplots(
    len(features), len(life_indices), figsize=(4 * len(life_indices), 3 * len(features))
)

if len(features) == 1 and len(life_indices) == 1:
    axs = np.array([[axs]])
elif len(features) == 1:
    axs = axs.reshape(1, -1)
elif len(life_indices) == 1:
    axs = axs.reshape(-1, 1)

for col_idx, life_idx in enumerate(life_indices):
    if life_idx >= lifes.n_time_series:
        print(f"Warning: Life index {life_idx} out of range, skipping...")
        continue

    life_data = lifes.get_time_series(life_idx)

    for row_idx, feature in enumerate(features):
        ax = axs[row_idx, col_idx]

        if feature in life_data.columns:
            ax.plot(life_data[feature].values, color="steelblue", linewidth=0.8)
        else:
            ax.text(
                0.5,
                0.5,
                f"{feature}\nnot found",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )

        if row_idx == 0:
            ax.set_title(f"Life {life_idx}", fontsize=10, fontweight="bold")

        if col_idx == 0:
            ax.set_ylabel(feature, fontsize=8, rotation=45, ha="right")

        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

plt.tight_layout()

if exp_config.save_plot:
    filename = f"{get_current_time()}_raw_signals_grid_{data_split}.png"
    filepath = os.path.join(plot_path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print("-" * 50)
    print(f"Plot saved at: {filepath}")
    print("-" * 50)

if exp_config.show_plot:
    plt.show()
