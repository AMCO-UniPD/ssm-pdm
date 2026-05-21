"""
Python script to print the business metrics tables for a quantile regression model
"""

# general imports
import os
import sys

import ipdb
import numpy as np
import pandas as pd
import setproctitle

src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.append(src_path)

from exp_config import setup_exp
from utils import generate_path, get_most_recent_file, open_element

experiment_path = os.path.dirname((os.path.realpath(__file__)))

exp_config, model_config, device, exp_name = setup_exp()

is_baseline_model = "quantile" in exp_config.model_name

if is_baseline_model:

    business_metrics_path = generate_path(
        basepath=experiment_path,
        folders=[
            "baseline_business_metrics",
            exp_config.model_name,
        ],
    )

else:

    business_metrics_path = generate_path(
        basepath=experiment_path,
        folders=[
            "business_metrics",
            exp_config.model_name,
            exp_config.failure_type,
            exp_config.approach,
            exp_name,
        ],
    )

setproctitle.setproctitle(f"print_business_metrics_{exp_name}")

for quantile in exp_config.quantiles:

    business_metrics_path_quantile = (
        generate_path(basepath=business_metrics_path, folders=[f"quantile_{quantile}"])
        if not is_baseline_model
        else business_metrics_path
    )

    business_metrics_df_path = get_most_recent_file(business_metrics_path_quantile)
    business_metrics_df = open_element(business_metrics_df_path)

    print("-"*50)
    print(f"Mean business metrics table for quantile {quantile}")
    print("-"*50)

    print("-"*50)
    print(business_metrics_df.to_markdown())
    print("-"*50)
