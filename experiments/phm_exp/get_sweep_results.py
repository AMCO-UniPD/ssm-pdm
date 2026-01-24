"""
Python script to get the results of a wandb sweep after it has finished
"""

import argparse
import wandb

api = wandb.Api()

parser = argparse.ArgumentParser(description="Get the best config of an already finished wandb sweep")

parser.add_argument(
    "--wandb_username",
    type = str,
    default = "frizzo-davide-Univeristy of Padova",
    help = "wandb username"
)

parser.add_argument(
    "--project_name",
    type = str,
    default = "ssm-pdm",
    help = "wandb project name"
)

parser.add_argument(
    "--sweep_id",
    type = str,
    default = "hxol4h28",
    help = "wandb sweep id"
)

args = parser.parse_args()

sweep_path = f"{args.wandb_username}/{args.project_name}/{args.sweep_id}"
sweep = api.sweep(sweep_path)

best_run = sweep.best_run()
print("-"*50)
print(f"Best config: {best_run.config}")
print("-"*50)

