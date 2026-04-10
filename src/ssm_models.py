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

# dataclass for the model configuration

@dataclass
class ModelConfig:
    """
    This dataclass contains all the configuration parameters for the
    models used in the project
    """

    random_init: bool = True
    d_model: int = 128
    n_layers: int = 5
    dropout: float = 0.0
    dropout_fn: Optional[Callable] = field(default=torch.nn.modules.dropout.Dropout1d)
    gap: bool = True
    # quantile regression
    quantile_reg: bool = True
    tau_mult: bool = True
    tau_feat: bool = True
    # monotonic
    n_mono_layers: int = 2
    n_neurons: int = 128
    n_mono_neurons: int = 128
    # S4 config
    lr: float = 1.0e-3
    activation: str = "relu"
    gate_act: str = "null"
    mult_act: str = "null"
    final_act: str = "glu"
    prenorm: bool = False
    # S4D config
    d_state: int = 64
    act: str = "gelu"
    # S5 config
    bidir: bool = False
    ff_dropout: float = 0.0
    attn_dropout: float = 0.0
    # RULTransformer config
    d_ff: int = 64
    n_heads: int = 8
    # RULInformer config
    factor: int = 5
    attn: str = "prob"
    inf_activation: str = "gelu"
    distil: bool = True
    output_attention: bool = False
    device: str = "cpu"

    @classmethod
    def from_dict(cls, config: dict) -> "ModelConfig":
        valid_keys = {f.name for f in fields(cls)}
        unknown = config.keys() - valid_keys
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")
        return cls(**config)

    def add_params(self, args: dict):
        """
        This function let's us to add some additional parameters
        to the dataclass (i.e. parameters passed through the command line)
        """
        for key in args:
            setattr(self, key, args[key])


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

def concat_tau(
    x: torch.Tensor,
    tau: float = 0.5,
    device: str = "cpu",
    tau_feat: bool = False
) -> torch.Tensor:
    """
    This function checks some stuff on the quantile level tau
    and concatenates it to the input in case the tau_feat flag is true,
    otherwise the tensor passed in input is returned.

    Args:
        x (torch.Tensor): input tensor
        tau (float): quantile level
        device (str): CUDA device
        tau_feat (bool): boolean flag to decide weather to concatenate the quantile
        level to the input or not

    Returns:
        x (torch.Tensor): input tensor if tau_feat=False, otherwise the input tensor
        with tau added as an additional feature
    """

    assert tau is not None, "tau must be provided for quantile regression"
    assert isinstance(tau, float), "tau must be a float"
    assert 0 <= tau <= 1, "tau must be between 0 and 1"

    if tau_feat:
        x = torch.cat(
            [x, torch.ones(x.shape[0], x.shape[1], 1).to(device) * tau],
            dim=-1,
        )

    return x

# Manual parameter count computation in case torchinfo does not work

def model_summary_manual(model: nn.Module) -> int:
    """
    Manual version of torchinfo summary module.

    Args:
        model: nn.Module object

    Returns:
        total_params: the number of parameters in the model
    """

    total_params = 0
    trainable_params = 0
    non_trainable_params = 0
    for name, param in model.named_parameters():
        num_params = param.numel()  # Number of elements in the parameter
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        else:
            non_trainable_params += num_params
        print(
            f"{name}: Shape={list(param.shape)}, Num params={num_params},"
            f" Trainable={param.requires_grad}"
        )
        print("#" * 50)

    print("#" * 50)
    print("Model Summary:")
    print("#" * 50)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")

    return total_params


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
