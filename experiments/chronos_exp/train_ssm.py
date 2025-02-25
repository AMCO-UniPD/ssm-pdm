"""
Training script for the `SSM` models in the `chronos-pdm` project
"""

# general imports
import os
import sys
import ipdb
import torch
import argparse
import setproctitle

chronos_path = os.path.join(os.path.dirname(__file__),"..","..","src",
                            "chronos-rul","src")
src_path = os.path.join(os.path.dirname(__file__),"..","..","src",
)
sys.path.append(src_path)
sys.path.append(chronos_path)

from utils import (
    ExperimentConfig,
    generate_path,
    get_most_recent_file,
    load_yaml_to_dict,
    open_element,
)

from models import (
    wandb_run,
    best_model_perf
)

from loss import load_loss_functions
from perf import (
        lifes_metrics,
        df_with_index_to_obsidian_table,
)
from plots import plot_predictions_grid

experiment_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"chronos_exp")

parser=argparse.ArgumentParser(description="Training script for SSM models in chronos-pdm project")
parser.add_argument("--exp_config_path",type=str,default="config/exp_config.yaml",help="Path to the experiment config file")
args=parser.parse_args()

exp_config=load_yaml_to_dict(args.exp_config_path)
exp_config=ExperimentConfig(exp_config)
model_config=load_yaml_to_dict(exp_config.model_config_path)

device = torch.device(f"cuda:{exp_config.device_num}" if torch.cuda.is_available() else "cpu")

print("#"*50)
print(f"Using device: {device}")
print("#"*50)

best_model_path = generate_path(basepath=experiment_path,
                                   folders=["best_models",
                                            exp_config.model_name,
                                            exp_config.cmapss_models])

outputs_path = generate_path(basepath=experiment_path,
                                   folders=["outputs",
                                            exp_config.model_name,
                                            exp_config.cmapss_models])

metrics_path = generate_path(basepath=experiment_path,
                             folders=["metrics",
                                      exp_config.model_name,
                                      exp_config.cmapss_models])

plot_path = generate_path(basepath=experiment_path,
                             folders=["plots",
                                      exp_config.model_name,
                                      exp_config.cmapss_models])

ipdb.set_trace()

if exp_config.test_script:

    print("#"*50)
    print("Running best model performance test")
    print("#"*50)

    setproctitle.setproctitle(f"{exp_config.model_name}-test-script")

    if exp_config.save_outputs:

        print("#"*50)
        print("Saving outputs")
        print("#"*50)

        best_model_perf(
            config=exp_config,
            model_config=model_config,
            device=device,
            best_model_path=best_model_path,
            outputs_path=outputs_path,
        )
