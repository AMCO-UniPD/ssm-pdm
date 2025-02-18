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
    load_yaml_to_dict,
    load_reg_data,
)

from models import (
    wandb_run,
    best_model_perf
)

from loss import load_loss_functions

experiment_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"chronos_exp")

parser=argparse.ArgumentParser(description="Training script chronos-rul")
parser.add_argument("--exp_config_path",type=str,default="config/exp_config.yaml",help="Path to the experiment config file")
args=parser.parse_args()

exp_config=load_yaml_to_dict(args.exp_config_path)
exp_config=ExperimentConfig(exp_config)
model_config=load_yaml_to_dict(exp_config.model_config_path)

device = torch.device(f"cuda:{exp_config.device_num}" if torch.cuda.is_available() else "cpu")

if exp_config.test_script:
    
    print("#"*50)
    print("Running best model performance test")
    print("#"*50)

    train_loader,_,_=load_reg_data(exp_config)

    criterion,_=load_loss_functions(
        loss_name=exp_config.loss,
        eval_loss_name=exp_config.eval_loss,
        tau=exp_config.tau
    )

    best_model_perf(
        config=exp_config,
        model_config=model_config,
        train_loader=train_loader,
        criterion=criterion,
        experiment_path=experiment_path,
        device=device,
    )


run_name=f"{exp_config.model_name}_{exp_config.cmapss_models}"
setproctitle.setproctitle(run_name)
model,model_info = wandb_run(
    run_name=run_name,
    config=exp_config,
    model_config=model_config,
    device=device,
    best_model_path=os.getcwd()
)
