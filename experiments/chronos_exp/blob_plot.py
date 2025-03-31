"""
Python script to create a blob plot of model parameters, mult-adds and test metrics
"""

import os
import sys
import ipdb
from typing import List
import numpy as np
import matplotlib.pyplot as plt

src_path = os.path.join(os.path.dirname(__file__),"..","..","src",
)
sys.path.append(src_path)

experiment_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"chronos_exp")

from utils import (
    generate_path,
    load_yaml_to_dict,
    ExperimentConfig,
    get_most_recent_file,
    get_most_recent_dir,
    open_element,
    save_element,
    extract_number
)

from plots import blob_plot

config_path=os.path.join(experiment_path,"config","ssm_exp_config.yaml")
config=load_yaml_to_dict(config_path)
config=ExperimentConfig(config)

# Load the summary dict for all the models and save the number of parameters
# and the mult-adds in a dictionary.
plot_dict={}
plot_dict["params"],plot_dict["mult_adds"],plot_dict["model_name"],plot_dict["test_metric"] = [],[],[],[]
summary_dict_dirpath = generate_path(basepath=experiment_path,folders=["summary_dict"])
metrics_dirpath = generate_path(basepath=experiment_path,folders=["metrics"])


for model_name in config.model_names:

    summary_dict_dirpath_model = generate_path(basepath=summary_dict_dirpath,
                                            folders=[model_name])
    summary_dict_path = get_most_recent_file(summary_dict_dirpath_model,file_pos=config.file_pos)
    summary_dict = open_element(summary_dict_path)
    params,mult_adds = summary_dict["params"],summary_dict["mult_adds"]
    plot_dict["params"].append(params)
    plot_dict["mult_adds"].append(mult_adds)
    plot_dict["model_name"].append(model_name)

    metrics_dirpath_model = generate_path(basepath=metrics_dirpath,
                                          folders=[model_name,
                                                   config.cmapss_models,
                                                   config.approach,
                                                   "quantile_reg"
                                                   ])

    metrics_exp_dirpath_model = get_most_recent_dir(metrics_dirpath_model,file_pos=1)

    metrics_df_path = get_most_recent_file(metrics_exp_dirpath_model,file_pos=config.file_pos)
    metrics_df = open_element(metrics_df_path)
    plot_dict["test_metric"].append(metrics_df.loc["Life_mean",f"quantile_{config.quantile_run}"])

plot_dict["params_float"] = [extract_number(param) for param in plot_dict["params"]]

mult_adds_list=[]
for macs in plot_dict["mult_adds"]:
    if "KM" in macs:
        mult_adds_list.append(extract_number(macs)/1000)
    else:
        mult_adds_list.append(extract_number(macs))
plot_dict["mult_adds_float"] = mult_adds_list

plot_path = generate_path(basepath=experiment_path,
                          folders=["plots",
                                   "blob_plot",
                                   config.cmapss_models])

# Create the blob plot
blob_plot(
    plot_dict=plot_dict,
    config=config,
    plot_path=plot_path,
)
