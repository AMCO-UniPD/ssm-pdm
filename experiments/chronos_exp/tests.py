"""
Python script to perform some quick tests
"""

# general imports
import os
import sys
import ipdb
import setproctitle
setproctitle.setproctitle("chronos-rul-tests")

src_path = os.path.join(os.path.dirname(__file__),"..","..","src",
)
sys.path.append(src_path)
experiment_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"chronos_exp")

from perf import df_with_index_to_obsidian_table, sub_lifes_metrics
from utils import (
        ExperimentConfig,
        get_most_recent_file,
        open_element,
        load_yaml_to_dict,
        generate_path,
        save_element,
        get_current_time,
)

exp_config_path="config/exp_config.yaml"
exp_config=load_yaml_to_dict(exp_config_path)
exp_config=ExperimentConfig(exp_config)

outputs_path = generate_path(basepath=experiment_path,
                                   folders=["outputs",
                                            exp_config.model_name,
                                            exp_config.cmapss_models])

metrics_path = generate_path(basepath=experiment_path,
                             folders=["metrics",
                                      exp_config.model_name,
                                      exp_config.cmapss_models])

if exp_config.save_outputs:

    outputs_path = get_most_recent_file(outputs_path,file_pos=exp_config.file_pos)
    print(f"Opened outputs_dict at path: {outputs_path}")
    outputs_dict = open_element(outputs_path)
    y_pred,y_true = outputs_dict["y_pred"],outputs_dict["y_true"]
    ipdb.set_trace()

    print("#"*50)
    print(f"Ouputs dict keys: {outputs_dict.keys()}")
    print(f"y_pred shape: {y_pred.shape}")
    print(f"y_true shape: {y_true.shape}")
    print("#"*50)

if exp_config.compute_metrics:

    # n_files = len(os.listdir(metrics_path))
    metrics_df_path = get_most_recent_file(metrics_path,file_pos=exp_config.file_pos)
    metrics_df = open_element(metrics_df_path)

    print(f"Opened metrics_df at path: {metrics_df_path}")

    metrics_df = sub_lifes_metrics(
        config=exp_config,
        metrics_df=metrics_df,
    )
    metrics_df=df_with_index_to_obsidian_table(metrics_df)

    print("#"*50)
    print(f"Metrics dataframe in markdown format")
    print("#"*50)
    print(metrics_df)
    print()


# ipdb.set_trace()

