"""
Python script to group together the mean metrics df produced by all the models
to produce the table to insert in the paper
"""

# general imports
import os
import sys
import ipdb
import setproctitle
import numpy as np
import pandas as pd
from scipy import stats

src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.append(src_path)

from exp_config import setup_exp
from utils import (
    generate_path,
    get_most_recent_file,
    open_element,
    save_element
)

experiment_path = os.path.dirname((os.path.realpath(__file__)))
metrics_path = generate_path(basepath=experiment_path,folders=["metrics"])

exp_config, _ , _, exp_name = setup_exp()

setproctitle.setproctitle(f"quantile_metrics_{exp_name}")

metrics_dfs = []

for model_name in exp_config.model_names:

    print("-"*50)
    print(f"Retrieving mean metrics for model {model_name}")
    print("-"*50)

    model_metrics_path = generate_path(
        basepath = metrics_path,
        folders = [
            model_name,
            exp_config.failure_type,
            exp_config.approach,
            exp_name
        ]
    )

    mean_metrics_df_path = get_most_recent_file(model_metrics_path,file_pos=exp_config.file_pos)
    mean_metrics_df = open_element(mean_metrics_df_path,filetype="pickle")

    metrics_dfs.append(mean_metrics_df.loc["Life_mean"])

paper_metrics_df = pd.DataFrame(metrics_dfs)
paper_metrics_df.index.name = "Models"
paper_metrics_df.index = exp_config.model_names

print("-"*50)
print("Paper metrics df:")
print(paper_metrics_df.to_markdown())
print("-"*50)

