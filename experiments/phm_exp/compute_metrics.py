"""
Python script to compute the metrics of a quantile regression model
"""

# general imports
import os
import sys
import ipdb
import torch
import argparse
import setproctitle

src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.append(src_path)

from exp_config import setup_exp
from utils import (
    generate_path,
    save_element,
    get_current_time,
)
from perf import lifes_metrics

experiment_path = os.path.dirname((os.path.realpath(__file__)))

exp_config, model_config, device, exp_name = setup_exp()

outputs_path = generate_path(
    basepath=experiment_path,
    folders=[
        "outputs",
        exp_config.model_name,
        f"{exp_config.tool_type}_tools",
        exp_config.failure_type,
        exp_config.approach,
        exp_name
    ],
)

metrics_path = generate_path(
    basepath=experiment_path,
    folders=[
        "metrics",
        exp_config.model_name,
        f"{exp_config.tool_type}_tools",
        exp_config.failure_type,
        exp_config.approach,
        exp_name
    ],
)

setproctitle.setproctitle(f"compute_metrics_{exp_name}")

for quantile in exp_config.quantiles:
    print("#" * 50)
    print(f"Computing metrics for quantile level: {quantile}")
    print("#" * 50)

    quantile_reg_folders = [
        "quantile_reg",
        exp_name,
        f"quantile_{quantile}",
    ]
    quantile_metrics_path = generate_path(
        basepath=metrics_path, folders=quantile_reg_folders
    )
    quantile_outputs_path = generate_path(
        basepath=outputs_path, folders=quantile_reg_folders
    )

    metrics_df = lifes_metrics(
        config=exp_config,
        outputs_path=quantile_outputs_path,
        metrics_path=quantile_metrics_path,
    )
