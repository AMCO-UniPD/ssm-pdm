"""
Training script to perform a wandb sweep
"""

# general imports
import os
import sys
from pandas.io.formats.format import printing
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

print("-"*50)
print(f"Sweep configuration:\n {sweep_config}")
print("-"*50)

#NOTE: Start the sweep

setproctitle.setproctitle(exp_config.sweep_name)

if exp_config.continue_old_sweep:

    sweep_id = os.environ.get("WANDB_SWEEP_ID", exp_config.sweep_id)
    print("-"*50)
    print(f"Resuming old sweep with id {sweep_id}")
    print("-"*50)

else:

    sweep_id = wandb.sweep(sweep=sweep_config , project=exp_config.project_name)
    print("-"*50)
    print(f"Starting new sweep with id {sweep_id}")
    print("-"*50)

wandb.agent(
    sweep_id,
    function = wandb_run_sweep,
    count = 10,
    project = exp_config.project_name,
    entity = exp_config.wandb_entity
)

#NOTE: Print the final result of the sweep printing the best hyperparameters found

# After the agent finishes:
api = wandb.Api()
sweep = api.sweep(f"{exp_config.project_name}/{sweep_id}")

# Get the best run based on the metric defined in your sweep_config
best_run = sweep.best_run()

print("Sweep finished!")
print(f"Best run ID: {best_run.id}")
print(f"Best Score ({sweep.config.get('metric', {}).get('name')}): {best_run.summary.get(sweep.config.get('metric', {}).get('name'))}")
print("Best Hyperparameters:")
for k, v in best_run.config.items():
    print("-"*50)
    print(f"  {k}: {v}")
    print("-"*50)
