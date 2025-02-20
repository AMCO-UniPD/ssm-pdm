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


from utils import (
        ExperimentConfig,
        get_most_recent_file,
        open_element,
        load_yaml_to_dict,
        generate_path
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

    outputs_dict = open_element(outputs_path)

    print("#"*50)
    print(f"Ouputs dict keys: {outputs_dict.keys()}")
    print(f"y_pred shape: {outputs_dict['y_pred'].shape}")
    print(f"y_true shape: {outputs_dict['y_true'].shape}")
    print("#"*50)

if exp_config.compute_metrics:
    metrics_path = get_most_recent_file(metrics_path,file_pos=exp_config.file_pos)

    metrics_df = open_element(metrics_path)

    print("#"*50)
    print(f"Metrics dict shape: {metrics_df.shape}")
    print(f"Metrics dict columns: {metrics_df.columns}")
    print("#"*50)

ipdb.set_trace()
