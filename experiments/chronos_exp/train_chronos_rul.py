"""
Training script for the `chronos-pdm` project
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
    load_yaml_to_dict,
    load_reg_data,
)

from models import (
    wandb_run,
    best_model_perf
)

from loss import load_loss_functions
from perf import lifes_metrics
from plots import plot_predictions_grid

experiment_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"chronos_exp")

parser=argparse.ArgumentParser(description="Training script chronos-rul")
parser.add_argument("--exp_config_path",type=str,default="config/exp_config.yaml",help="Path to the experiment config file")
args=parser.parse_args()

exp_config=load_yaml_to_dict(args.exp_config_path)
exp_config=ExperimentConfig(exp_config)
model_config=load_yaml_to_dict(exp_config.model_config_path)

device = torch.device(f"cuda:{exp_config.device_num}" if torch.cuda.is_available() else "cpu")

best_model_path = generate_path(basepath=experiment_path,
                                   folders=["best_models",
                                            config.model_name,
                                            config.cmapss_models])

outputs_path = generate_path(basepath=experiment_path,
                                   folders=["outputs",
                                            config.model_name,
                                            config.cmapss_models])

metrics_path = generate_path(basepath=experiment_path,
                             folders=["metrics",
                                      exp_config.model_name,
                                      exp_config.cmapss_models])

plot_path = generate_path(basepath=experiment_path,
                             folders=["plots",
                                      exp_config.model_name,
                                      exp_config.cmapss_models])

if exp_config.test_script:
    
    print("#"*50)
    print("Running best model performance test")
    print("#"*50)

    best_model_perf(
        config=exp_config,
        model_config=model_config,
        device=device,
        best_model_path=best_model_path,
        outputs_path=outputs_path,
    )

    print("#" * 50)
    print("Computing metrics for each life and for each sensor in the test set")
    print("#" * 50)

    metrics_df = lifes_metrics(
        config=exp_config,
        outputs_path=outputs_path,
        metrics_path=metrics_path,
    )
    print("#" * 50)
    print(f"metrics_df shape: {metrics_df.shape}")

    print("#" * 50)
    print("Producing grid plot of the predictions")
    print("#" * 50)
    fig = plot_predictions_grid(
        config=exp_config,
        sensor_idx=exp_config.sensor_idx,
        outputs_path=outputs_path,
        plot_path=plot_path,
    )


print("#"*50)
print("Model training started")
print("#"*50)

run_name=f"{exp_config.model_name}_{exp_config.cmapss_models}"
setproctitle.setproctitle(run_name)
model,model_info = wandb_run(
    run_name=run_name,
    config=exp_config,
    model_config=model_config,
    device=device,
    best_model_path=best_model_path,
    metrics_path=metrics_path,
    plot_path=plot_path
)
