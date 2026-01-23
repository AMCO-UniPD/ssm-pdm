"""
Training script to perform a wandb sweep
"""

# general imports
import os
import sys
import wandb
import ipdb
import torch
import argparse
import setproctitle

src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.append(src_path)

from exp_config import setup_exp
from wandb_sweep import wandb_run_sweep, define_sweep_config

experiment_path = os.path.dirname((os.path.realpath(__file__)))
sweep_config_path = os.path.join(experiment_path,"config","sweep_config.yaml")
exp_config, model_config, device, exp_name = setup_exp()

#NOTE: Sweep configuration

sweep_config = define_sweep_config(sweep_config_path = sweep_config_path)

#NOTE: Start the sweep

setproctitle.setproctitle(exp_config.sweep_name)

sweep_id = wandb.sweep(sweep=sweep_config , project=exp_config.project_name)
wandb.agent(sweep_id , function=wandb_run_sweep, count=10)
