"""
Python module containing utility functions for the models migrated from the
`SSM_PDM` project into the `chronos_pdm` project.
"""

import os
import sys
import ipdb
import traceback
import numpy as np
from typing import (
    Tuple,
    Union,
    Optional,
    Callable,
)
from dataclasses import (
    dataclass,
    fields,
    field,
)

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from transformer_encoder import TransformerEncoder
from transformer_encoder.utils import PositionalEncoding

# model summary imports
from torchinfo import summary
from calflops import calculate_flops

from utils import ExperimentConfig, print_life_info, save_element, generate_path

from model_classes import RULModel, QuantileRULModel, MonotonicRULModel, MonoQuantileRULModel

chronos_path_src = os.path.join(os.path.dirname(__file__), "chronos-rul", "src")
imports_path = os.path.join(os.path.dirname(__file__), "AD_MG", "src")
sys.path.append(chronos_path_src)
sys.path.append(imports_path)

# s4 imports
from s4 import DropoutNd
from s4 import S4Block as S4

# s4d imports
from s4d import S4D

# s5 imports
from s5 import S5Block

# informer imports
from informer import *

cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
experiment_path = os.path.join(cwd, "experiments", "chronos_exp")


def setup_optimizer(model, lr, weight_decay, epochs):
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s)
        for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group({"params": params, **hp})

    # Create a lr scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        # print(' | '.join([
        #     f"Optimizer group {i}",
        #     f"{len(g['params'])} tensors",
        # ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler

class MonotonicLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device = None,
        dtype = None,
        pre_activation=nn.Identity(),
    ):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.act = pre_activation

    def forward(self, x):
        w_pos = self.weight.clamp(min=0.0)
        w_neg = self.weight.clamp(max=0.0)
        x_pos = F.linear(self.act(x), w_pos, self.bias)
        x_neg = F.linear(self.act(-x), w_neg, self.bias)
        return x_pos + x_neg

# Function to create the model

def load_ssm_model(
    exp_config: ExperimentConfig,
    model_config: ModelConfig,
    mono_mask: np.ndarray = np.zeros(shape=(10,1)),
    tau: float = 0.5,
    **kwargs
) -> Tuple[nn.Module, optim.Optimizer, optim.lr_scheduler]:
    """
    Function to create the model based on the configuration.
    The function also sets up the optimizer and the scheduler.

    Args:
        exp_config (ExperimentConfig): experiment configuration object
        model_config (ModelConfig): model configuration object
        mono_mask (np.ndarray): boolean mask to identify monotonic features
        tau (float): The quantile level on which the model will be evaluated if the quantile regression approach is used

    Returns:
        model: nn.Module object
        optimizer: torch.optim object
        scheduler: torch.optim.lr_scheduler object
    """

    if exp_config.quantile_reg and exp_config.monotonic:
        model = MonoQuantileRULModel(tau=tau, mono_mask=mono_mask, **kwargs)
    elif not exp_config.quantile_reg and exp_config.monotonic:
        model = MonotonicRULModel(mono_mask=mono_mask, **kwargs)
    elif exp_config.quantile_reg and not exp_config.monotonic:
        model = QuantileRULModel(tau=tau, **kwargs)
    else:
        model = RULModel(**kwargs)

    optimizer, scheduler = setup_optimizer(
        model,
        lr=exp_config.lr,
        weight_decay=exp_config.weight_decay,
        epochs=exp_config.epochs,
    )

    return model, optimizer, scheduler
