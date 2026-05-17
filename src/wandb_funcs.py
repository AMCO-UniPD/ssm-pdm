"""
Python module containing some functions related
to wandb logging
"""

import os
from typing import Union

# wandb imports
import wandb
from wandb.sdk.wandb_run import Run as WandbRun

from exp_config import ExperimentConfig, ModelConfig

def set_wandb_run_name(
    config: ExperimentConfig,
    run: WandbRun
) -> WandbRun:
    """
    Set the wandb run name

    Args:
        config (ExperimentConfig): experiment configuration object
        run (WandbRun): wandb run object

    Returns:
        run (WandbRun): updated WandbRun instance
    """

    if config.exp_name == "wandb_run":
        run.name = f"{config.exp_name}_test"
    else:
        run.name = config.exp_name

    return run

def init_wandb(
    config: ExperimentConfig,
    model_config: ModelConfig
) -> Union[WandbRun, None]:
    """
    Function to initialize a wandb run using the wandb API key
    to avoid multi login problems

    Args:
        config (ExperimentConfig): experiment configuration object
        model_config (ModelConfig): model configuration

    Returns:
        run (WandbRun): WandbRun instance
    """

    # from dotenv import load_dotenv
    # load_dotenv()

    if not config.use_wandb:
        print("-"*50)
        print("Performing experiment without wandb logging")
        print("-"*50)
        return None

    print("-"*50)
    print("Initializing wandb logging")
    print("-"*50)

    runWB = wandb.init(
        project=config.project_name,
        save_code=False,
    )

    runWB = set_wandb_run_name(
        config = config,
        run = runWB
    )

    print("-"*50)
    print(f"Run {runWB.name} started")
    print(f"View at: {runWB.url}")
    print("-"*50)

    return runWB

