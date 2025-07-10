"""
Python script to create a blob plot of model parameters, mult-adds and test metrics
"""

import os
import sys
import ipdb
import argparse
from typing import List
import numpy as np
import matplotlib.pyplot as plt

src_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "src",
)
sys.path.append(src_path)

experiment_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "chronos_exp"
)

from utils import (
    generate_path,
    load_yaml_to_dict,
    ExperimentConfig,
    get_most_recent_file,
    get_most_recent_dir,
    open_element,
    save_element,
    extract_number,
)

from plots import blob_plot
from ssm_models import ModelConfig
from perf import state_dict_size, time_exp

config_path = os.path.join(experiment_path, "config", "ssm_exp_config.yaml")
config = load_yaml_to_dict(config_path)
config = ExperimentConfig(config)
model_config = load_yaml_to_dict(config.model_config_path)
model_config = ModelConfig(model_config)
model_config.quantile_reg = config.quantile_reg

parser = argparse.ArgumentParser(description="Blob Plot Experiment")
parser.add_argument(
    "--no_mult_adds",
    action="store_true",
    help="If set, do not use the Mult Adds metric for the blob plot",
)
args = parser.parse_args()

# Load the summary dict for all the models and save the number of parameters
# and the mult-adds in a dictionary.
plot_dict = {}
(
    plot_dict["params"],
    plot_dict["mult_adds"],
    plot_dict["model_name"],
    plot_dict["test_metric"],
    plot_dict["pickle_size_kb"],
    plot_dict["test_time"],
) = [], [], [], [], [], []
# plot_dict["metrics_dirpath"]=[]
summary_dict_dirpath = generate_path(basepath=experiment_path, folders=["summary_dict"])
metrics_dirpath = generate_path(basepath=experiment_path, folders=["metrics"])

for model_name in config.model_names:
    if model_name == "LSTM":
        summary_dict_dirpath_model = generate_path(
            basepath=summary_dict_dirpath, folders=[model_name, "torchinfo"]
        )
    else:
        summary_dict_dirpath_model = generate_path(
            basepath=summary_dict_dirpath, folders=[model_name, config.summary_func]
        )

    print("#" * 50)
    print(f"Getting the summary dict for model: {model_name}")
    print("#" * 50)
    summary_dict_path = get_most_recent_file(
        summary_dict_dirpath_model, file_pos=config.file_pos
    )
    summary_dict = open_element(summary_dict_path)
    params, mult_adds = summary_dict["params"], summary_dict["mult_adds"]
    plot_dict["params"].append(params)
    plot_dict["mult_adds"].append(mult_adds)
    plot_dict["model_name"].append(model_name)

    print("#" * 50)
    print(f"Performing time experiment for model: {model_name}")
    print("#" * 50)
    dict_time = time_exp(config=config, model_config=model_config)
    plot_dict["test_time"].append(dict_time["avg_time"])

    print("#" * 50)
    print(f"Computing state dict size for model {model_name}")
    print("#" * 50)
    size_dict, state_dict_path = state_dict_size(
        config=config, basepath=experiment_path, model_name=model_name
    )
    print("#" * 50)
    print(f"Getting pickle size from file: {state_dict_path}")
    print("#" * 50)
    plot_dict["pickle_size_kb"].append(size_dict["pickle_size_kb"])

    metrics_dirpath_model = generate_path(
        basepath=metrics_dirpath,
        folders=[model_name, config.cmapss_models, config.approach, "quantile_reg"],
    )

    print("#" * 50)
    print(f"Getting the metrics dataframe for model: {model_name}")
    print("#" * 50)
    metrics_exp_dirpath_model = get_most_recent_dir(
        metrics_dirpath_model, file_pos=config.file_pos
    )
    # plot_dict["metrics_dirpath"].append(os.path.basename(metrics_exp_dirpath_model))
    metrics_df_path = get_most_recent_file(
        metrics_exp_dirpath_model, file_pos=config.file_pos
    )
    metrics_df = open_element(metrics_df_path)
    plot_dict["test_metric"].append(
        metrics_df.loc["Life_mean", f"quantile_{config.quantile_run}"]
    )

# plot_dict["params_float"] = [extract_number(param) for param in plot_dict["params"]] if all(isinstance(item,str) for item in plot_dict["params"]) else plot_dict["params"]

params_list = []
for param in plot_dict["params"]:
    if isinstance(param, str):
        params_list.append(extract_number(param))
    else:
        params_list.append(param)

plot_dict["params_float"] = params_list

if args.no_mult_adds:
    del plot_dict["mult_adds"]
else:
    mult_adds_list = []
    for macs in plot_dict["mult_adds"]:
        if isinstance(macs, str):
            if "KM" in macs:
                mult_adds_list.append(extract_number(macs) / 1000)
            else:
                mult_adds_list.append(extract_number(macs))
        else:
            mult_adds_list.append(macs)

    plot_dict["mult_adds_float"] = mult_adds_list

ipdb.set_trace()

plot_path = generate_path(
    basepath=experiment_path, folders=["plots", "blob_plot", config.cmapss_models]
)

# Create the blob plot
blob_plot(
    plot_dict=plot_dict,
    config=config,
    plot_path=plot_path,
    no_mult_adds=args.no_mult_adds,
)
