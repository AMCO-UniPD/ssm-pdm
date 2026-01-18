"""
Training script for PHM dataset experiments
"""

# general imports
import os
import sys
import ipdb
import torch
import argparse
import setproctitle

src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.append(src_path)

from utils import (
    ExperimentConfig,
    generate_path,
    get_most_recent_file,
    load_yaml_to_dict,
    open_element,
    get_current_time,
    save_element,
    set_seed,
    load_phm_data
)

from models import wandb_run, best_model_perf

from ssm_models import ModelConfig

experiment_path = os.path.dirname((os.path.realpath(__file__)))

parser = argparse.ArgumentParser(description="Training script for SSM models in chronos-pdm project")
parser.add_argument(
    "--exp_config_path",
    type=str,
    default="config/exp_config.yaml",
    help="Path to the experiment config file",
)
args = parser.parse_args()

exp_config = load_yaml_to_dict(args.exp_config_path)
exp_config = ExperimentConfig(exp_config)
model_config = load_yaml_to_dict(exp_config.model_config_path)
model_config = ModelConfig(model_config)
model_config.quantile_reg = exp_config.quantile_reg

device = torch.device(f"cuda:{exp_config.device_num}" if torch.cuda.is_available() else "cpu")
model_config.device = device

print("#" * 50)
print(f"Using device: {device}")
print("#" * 50)

loaders_dict = load_phm_data(config=exp_config)

