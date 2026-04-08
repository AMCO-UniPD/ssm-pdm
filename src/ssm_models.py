"""
Python module containing utility functions for the models migrated from the
`SSM_PDM` project into the `chronos_pdm` project.
"""

import os
import sys
import ipdb
import traceback
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

class GapHead(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        d_output: int,
    ):
        """
        Model Regression head for the GAP approach
        """
        super().__init__()

        self.decoder = nn.Linear(config.d_model, d_output)
        self.tau_mult = config.tau_mult

    def forward(self,x, tau=0.5):

        x = x.mean(dim=1) # (B,L,d_model) → (B,d_model)
        x = self.decoder(x) * tau if self.tau_mult else self.decoder(x)

        return x

class ADHead(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        d_input: int,
    ):
        """
        Model Regression head for the AD approach
        """
        super().__init__()

        self.decoder = nn.Linear(config.d_model, d_input) if not config.quantile_reg else nn.Linear(config.d_model, d_input-1)
        self.tau_mult = config.tau_mult

    def forward(self, x, tau=0.5):

        x = self.decoder(x) * tau if self.tau_mult else self.decoder(x)  # (B,L,d_model) -> (B,L,d_input)
        return x

class ModelHead(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        sequence_length: int,
    ):
        """
        Model Regression head for all the padding and window approaches
        that are not using the gap approach. In this case the embedding of the
        last time stamp is used to produce the output
        """
        super().__init__()

        self.decoder = nn.Linear(config.d_model, sequence_length)
        self.tau_mult = config.tau_mult

    def forward(self, x, tau=0.5):

        x = x[:,-1,:].squeeze(1) # (B, L, d_model) -> (B,1,d_model) → (B, d_model)
        x = self.decoder(x) * tau if self.tau_mult else self.decoder(x)  # (B, d_model) -> (B, d_output)
        return x

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

class MonotonicHead(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        sequence_length: int,
    ):
        """
        Model Head for monotonic neural networks. With this model head (that needs to have at least 4 layers
        to be a universal approximator) the model should be monotonic, so it should produce always non increasing
        predictions for the RUL
        """
        super().__init__()

        self.tau_mult = config.tau_mult

        self.decoder = nn.Sequential([
            MonotonicLinear(config.d_model, config.d_model, pre_activation=nn.Identity()),
            MonotonicLinear(config.d_model, config.d_model, pre_activation=nn.ReLU()),
            MonotonicLinear(config.d_model, config.d_model, pre_activation=nn.ReLU()),
            MonotonicLinear(config.d_model, sequence_length, pre_activation=nn.ReLU()),
        ])

    def forward(self, x, tau=0.5):

        x = x[:,-1,:].squeeze(1) # (B, L, d_model) -> (B,1,d_model) → (B, d_model)
        x = self.decoder(x) * tau if self.tau_mult else self.decoder(x)  # (B, d_model) -> (B, d_output)
        return x

class FullLifeHead(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        d_output: int,
    ):
        """
        Model Regression head for the full_life approach
        """
        super().__init__()

        self.decoder = nn.Linear(config.d_model, d_output)
        self.tau_mult = config.tau_mult

    def forward(self, x, tau=0.5):

        x = self.decoder(x) * tau if self.tau_mult else self.decoder(x)  # (B, d_model) -> (B, d_output)
        return x.squeeze(-1)

def select_model_head(
    config: ModelConfig,
    d_input: int,
    d_output: int,
    sequence_length: int,
    ad: bool,
    gap: bool,
    full_life: bool,
    monotonic: bool
) -> nn.Module:
    """
    Function used to select the model head depending on the approach
    used.

    Args:
        ad (bool): weather to use the AD approach
        gap (bool): weather to use Global Average Pooling (GAP)
        full_life (bool): weather to use the full_life approach
        monotonic (bool): weather to use the monotonic approach
        d_input (int): number of input features of the model
        d_output (int): number of output features of the model
        sequence_length (int): length of the sequences used

    Returns:
        head (nn.Module): One of the possible model heads
    """

    if ad:
        head = ADHead(
            config = config,
            d_input = d_input
        )

    if gap:
        head = GapHead(
            config = config,
            d_output = d_output
        )
    else:
        head = ModelHead(
            config = config,
            sequence_length = sequence_length
        )

    if full_life:
        head = FullLifeHead(
            config = config,
            d_output = d_output
        )

    if monotonic:
        head = MonotonicHead(
            config = config,
            sequence_length = sequence_length
        )

    return head

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

class LinearModel(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        input_size: int,
        output_size: int,
        sequence_length: int,
        ad: bool = False,
        monotonic: bool = False,
        full_life: bool = False
    ):
        super(LinearModel, self).__init__()
        self.quantile_reg = config.quantile_reg
        self.tau_feat = config.tau_feat
        self.tau_mult = config.tau_mult
        self.gap = config.gap
        self.device = config.device
        self.ad = ad
        self.monotonic = monotonic
        self.full_life = full_life

        self.fc1 = nn.Linear(input_size, config.d_model)

        self.head = select_model_head(
            config = config,
            d_input = input_size,
            d_output = output_size,
            sequence_length = sequence_length,
            ad = self.ad,
            gap = self.gap,
            monotonic = self.monotonic,
            full_life = self.full_life
        )

    def forward(self, x, tau=0.5):
        if self.quantile_reg:
            x = concat_tau(
                x = x,
                tau = tau,
                device = self.device,
                tau_feat = self.tau_feat
            )

        x = self.fc1(x) # (B,L,D) → (B,L,H)
        x = self.head(x, tau = tau) # (B,L,H) -> (B,L)

        return x

class MLPModel(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        input_size: int,
        output_size: int,
        sequence_length: int,
        ad: bool = False,
        monotonic: bool = False,
        full_life: bool = False
    ):
        super(MLPModel, self).__init__()
        self.quantile_reg = config.quantile_reg
        self.tau_feat = config.tau_feat
        self.tau_mult = config.tau_mult
        self.gap = config.gap
        self.device = config.device
        self.ad = ad
        self.monotonic = monotonic
        self.full_life = full_life

        self.fc1 = nn.Linear(input_size, config.d_model)
        self.hidden_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.acts = nn.ModuleList()

        if config.act == "gelu":
            self.activation = nn.GELU()
        elif config.act == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()

        for _ in range(config.n_layers):
            self.hidden_layers.append(nn.Linear(config.d_model, config.d_model))
            self.norms.append(nn.LayerNorm(config.d_model))
            self.acts.append(self.activation)
            self.dropouts.append(nn.Dropout(config.dropout))
            self.hidden_layers.append(self.activation)

        self.head = select_model_head(
            config = config,
            d_input = input_size,
            d_output = output_size,
            sequence_length = sequence_length,
            ad = self.ad,
            gap = self.gap,
            monotonic = self.monotonic,
            full_life = self.full_life
        )

    def forward(self, x, tau=0.5):
        if self.quantile_reg:
            x = concat_tau(
                x = x,
                tau = tau,
                device = self.device,
                tau_feat = self.tau_feat
            )

        x = self.fc1(x) # (B,L,D) → (B,L,H)

        # In this loop it's all (B,L,H) → (B,L,H)

        for layer, norm, act, dropout in zip(self.hidden_layers, self.norms, self.acts, self.dropouts):
            x = layer(x)
            x = norm(x)
            x = act(x)
            x = dropout(x)

        x = self.head(x, tau=tau) # (B, L, d_model) -> (B, L)

        return x

# RNN based models

class Recurrent_PDM(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        model_name: str,
        input_size: int,
        output_size: int,
        sequence_length: int,
        ad: bool = False,
        monotonic: bool = False,
        full_life: bool = False
    ):
        super(Recurrent_PDM, self).__init__()

        self.quantile_reg = config.quantile_reg
        self.tau_feat = config.tau_feat
        self.tau_mult = config.tau_mult
        self.device = config.device
        self.gap = config.gap
        self.ad = ad
        self.monotonic = monotonic
        self.full_life = full_life

        if model_name == "LSTM":
            self.recurrent = nn.LSTM(
                input_size=input_size,
                hidden_size=config.d_model,
                num_layers=config.n_layers,
                batch_first=True,
                dropout=config.dropout,
            )
        elif model_name == "GRU":
            self.recurrent = nn.GRU(
                input_size=input_size,
                hidden_size=config.d_model,
                num_layers=config.n_layers,
                batch_first=True,
                dropout=config.dropout,
            )
        elif model_name == "RNN":
            self.recurrent = nn.RNN(
                input_size=input_size,
                hidden_size=config.d_model,
                num_layers=config.n_layers,
                batch_first=True,
                dropout=config.dropout,
            )

        self.norm = nn.LayerNorm(config.d_model)

        self.head = select_model_head(
            config = config,
            d_input = input_size,
            d_output = output_size,
            sequence_length = sequence_length,
            ad = self.ad,
            gap = self.gap,
            monotonic = self.monotonic,
            full_life = self.full_life
        )

    def forward(self, x, tau=0.5):
        if self.quantile_reg:
            x = concat_tau(
                x = x,
                tau = tau,
                device = self.device,
                tau_feat = self.tau_feat
            )

        x = self.recurrent(x)[0]  # (B, L, D) -> (B, L, H)
        x = self.norm(x) # (B, L, H) -> (B, L, H)
        x = self.head(x, tau=tau) # (B, L, H) → (B, L)

        return x

# SSM model classes


class S4Model(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        d_input: int,
        d_output: int,
        sequence_length: int,
        ad: bool = False,
        full_life: bool = False,
        monotonic: bool = False,
    ):
        super().__init__()

        self.prenorm = config.prenorm
        self.gap = config.gap
        self.tau_mult = config.tau_mult
        self.tau_feat = config.tau_feat
        self.quantile_reg = config.quantile_reg
        self.device = config.device
        self.ad = ad
        self.full_life = full_life
        self.monotonic = monotonic
        d_model = config.d_model
        n_layers = config.n_layers
        dropout = config.dropout
        activation = config.activation
        gate_act = config.gate_act
        mult_act = config.mult_act
        final_act = config.final_act

        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4(
                    d_model,
                    dropout=dropout,
                    activation=activation,
                    gate_act=gate_act,
                    mult_act=mult_act,
                    final_act=final_act,
                    transposed=True,
                    lr=min(0.001, config.lr),
                )
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout(dropout))

        self.head = select_model_head(
            config = config,
            d_input = d_input,
            d_output = d_output,
            sequence_length = sequence_length,
            ad = self.ad,
            gap = self.gap,
            full_life = self.full_life,
            monotonic = self.monotonic
        )

    def forward(self, x, tau=0.5):
        if self.quantile_reg:
            x = concat_tau(
                x = x,
                tau = tau,
                device = self.device,
                tau_feat = self.tau_feat
            )

        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x

            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)  # (B, d_model, L) -> (B, L, d_model)

        x = self.head(x,tau=tau)

        return x

class S4DModel(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        d_input: int,
        d_output: int,
        sequence_length: int,
        ad: bool = False,
        monotonic: bool = False,
        full_life: bool = False
    ):
        super().__init__()

        self.gap = config.gap
        self.tau_mult = config.tau_mult
        self.tau_feat = config.tau_feat
        self.quantile_reg = config.quantile_reg
        self.device = config.device
        self.ad = ad
        self.monotonic = monotonic
        self.full_life = full_life

        d_state = config.d_state
        act = config.act
        d_model = config.d_model
        n_layers = config.n_layers
        dropout = config.dropout

        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4D layers as residual blocks
        self.s4d_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4d_layers.append(
                S4D(
                    d_model=d_model,
                    d_output=d_output,
                    d_state=d_state,
                    dropout=dropout,
                    act=act,
                    transposed=True,
                )
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout(dropout))

        self.head = select_model_head(
            config = config,
            d_input = d_input,
            d_output = d_output,
            sequence_length = sequence_length,
            ad = self.ad,
            gap = self.gap,
            full_life = self.full_life,
            monotonic = self.monotonic
        )

    def forward(self, x, tau=0.5):
        if self.quantile_reg:
            x = concat_tau(
                x = x,
                tau = tau,
                device = self.device,
                tau_feat = self.tau_feat
            )

        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4d_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x

            # Apply S4D block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4D block
            z = dropout(z)

            # Residual connection
            x = z + x

            # layer norm
            x = norm(x.transpose(-1,-2)).transpose(-1,-2)

        x = x.transpose(-1, -2)

        x = self.head(x, tau = tau)

        return x

class S5Model(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        d_input: int,
        d_output: int,
        sequence_length: int,
        ad: bool = False,
        monotonic: bool = False,
        full_life: bool = False
    ):
        super().__init__()

        self.gap = config.gap
        self.tau_mult = config.tau_mult
        self.tau_feat = config.tau_feat
        self.quantile_reg = config.quantile_reg
        self.device = config.device
        self.ad = ad
        self.monotonic = monotonic
        self.full_life = full_life

        self.encoder = nn.Linear(d_input, config.d_model)

        # Stack S5 layers as residual blocks
        self.s5_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(config.n_layers):
            self.s5_layers.append(
                S5Block(
                    dim=config.d_model,
                    state_dim=config.d_state,
                    bidir=config.bidir,
                    ff_dropout=config.ff_dropout,
                    attn_dropout=config.attn_dropout
                )
            )
            self.norms.append(nn.LayerNorm(config.d_model))
            self.dropouts.append(nn.Dropout(config.dropout))

        self.head = select_model_head(
            config = config,
            d_input = d_input,
            d_output = d_output,
            sequence_length = sequence_length,
            ad = self.ad,
            gap = self.gap,
            full_life = self.full_life,
            monotonic = self.monotonic
        )

    def forward(self, x, tau=0.5):
        if self.quantile_reg:
            x = concat_tau(
                x = x,
                tau = tau,
                device = self.device,
                tau_feat = self.tau_feat
            )

        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        for layer, norm, dropout in zip(self.s5_layers, self.norms, self.dropouts):  # (B, L, H) -> (B, L, H). The P is used inside here (black box we do not care)

            # 1. layer
            x = layer(x)

            if torch.isnan(x).any():
                print("-"*50)
                print("Obtained NaN after x=layer(x) operation, check better inside the S5Block module")
                print("-"*50)
                ipdb.set_trace()

            # 2. dropout
            x = dropout(x)
            # 3. norm
            x = norm(x)

        x = self.head(x, tau=tau)

        return x

# Transformer based models

class RULTransformer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        input_size: int,
        output_size: int,
        sequence_length: int,
        ad: bool = False,
        monotonic: bool = False,
        full_life: bool = False
    ):
        super(RULTransformer, self).__init__()

        self.quantile_reg = config.quantile_reg
        self.tau_feat = config.tau_feat
        self.tau_mult = config.tau_mult
        self.device = config.device
        self.gap = config.gap
        self.ad = ad
        self.monotonic = monotonic
        self.full_life = full_life

        self.embedding = nn.Sequential(
            nn.Embedding(num_embeddings=input_size, embedding_dim=config.d_model),
            PositionalEncoding(
                d_model=config.d_model, dropout=config.dropout, max_len=output_size
            ),
        )

        self.encoder = TransformerEncoder(
            d_model=config.d_model,
            d_ff=config.d_ff,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dropout=config.dropout,
        )

        self.head = select_model_head(
            config = config,
            d_input = input_size,
            d_output = output_size,
            sequence_length = sequence_length,
            ad = self.ad,
            gap = self.gap,
            monotonic = self.monotonic,
            full_life = self.full_life
        )

    def forward(self, x, mask=None, tau=0.5):
        if mask is None:
            mask = torch.zeros(x.size(0), x.size(1)).to(x.device)

        if self.quantile_reg:
            x = concat_tau(
                x = x,
                tau = tau,
                device = self.device,
                tau_feat = self.tau_feat
            )

        x = x.argmax(dim=-1)  # (B, L, d_input) -> (B, L)
        x = self.embedding(x)  # (B, L) -> (B, L, d_model)
        x = self.encoder(x, mask)  # (B, L, d_model) -> (B, L, d_model)
        x = self.head(x, tau=tau) # (B, L, H) → (B, L)

        return x


# Informer based model


class RULInformer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        d_input: int,
        d_output: int,
        sequence_length: int,
        ad: bool = False,
        monotonic: bool = False,
        full_life: bool = False
    ):
        super(RULInformer, self).__init__()

        self.quantile_reg = config.quantile_reg
        self.tau_feat = config.tau_feat
        self.tau_mult = config.tau_mult
        self.device = config.device
        self.output_attention = config.output_attention
        self.gap = config.gap
        self.ad = ad
        self.monotonic = monotonic
        self.full_life = full_life

        # Encoding
        self.enc_embedding = DataEmbedding(
            c_in=d_input, d_model=config.d_model, dropout=config.dropout
        )
        # Attention
        Attn = ProbAttention if config.attn == "prob" else FullAttention
        # Encoder
        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    attention=AttentionLayer(
                        attention=Attn(
                            mask_flag=False,
                            factor=config.factor,
                            attention_dropout=config.dropout,
                            output_attention=True,
                        ),
                        d_model=config.d_model,
                        n_heads=config.n_heads,
                        mix=False,
                    ),
                    d_model=config.d_model,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                    activation=config.inf_activation,
                )
                for _ in range(config.n_layers)
            ],
            conv_layers=[ConvLayer(config.d_model) for _ in range(config.n_layers - 1)]
            if config.distil
            else None,
            norm_layer=torch.nn.LayerNorm(config.d_model),
        )

        self.head = select_model_head(
            config = config,
            d_input = d_input,
            d_output = d_output,
            sequence_length = sequence_length,
            ad = self.ad,
            gap = self.gap,
            monotonic = self.monotonic,
            full_life = self.full_life
        )

    def forward(self, x_enc, output_attention=False, enc_self_mask=None, tau=0.5):
        if self.quantile_reg:
            x_enc = concat_tau(
                x = x_enc,
                tau = tau,
                device = self.device,
                tau_feat = self.tau_feat
            )

        enc_out = self.enc_embedding(x_enc)  # [B,L,D] -> [B,L,H]
        enc_out, attns = self.encoder(
            enc_out, attn_mask=enc_self_mask
        )  # [B,L,H] -> [B,L,H]

        # enc_out = enc_out.mean(dim=1)  # [B,L,H] -> [B,H]

        dec_out = self.head(enc_out, tau=tau) # (B, L, H) → (B, L)

        if output_attention:
            return dec_out, attns
        else:
            return dec_out

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
    d_input: int,
) -> Union[
    Tuple[nn.Module, optim.Optimizer, optim.lr_scheduler], Tuple[nn.Module, dict]
]:
    """
    Function to create the model based on the configuration.
    The function also sets up the optimizer and the scheduler.

    Args:
        exp_config: ExperimentConfig object
        model_config: ModelConfig object

    Returns:
        model: nn.Module object
        optimizer: torch.optim object
        scheduler: torch.optim.lr_scheduler object
    """

    if exp_config.model_name == "S4":
        model = S4Model(
            config=model_config,
            d_input=d_input,
            d_output=1 if not model_config.gap else exp_config.sequence_length,
            sequence_length = exp_config.sequence_length,
            ad = exp_config.ad,
            monotonic = exp_config.monotonic,
            full_life = True if exp_config.approach == "full_life" else False
        )
    elif exp_config.model_name == "S4D":
        model = S4DModel(
            config=model_config,
            d_input=d_input,
            d_output=1 if not model_config.gap else exp_config.sequence_length,
            sequence_length = exp_config.sequence_length,
            ad=exp_config.ad,
            monotonic = exp_config.monotonic,
            full_life = True if exp_config.approach == "full_life" else False
        )
    elif exp_config.model_name == "S5":
        model = S5Model(
            config=model_config,
            d_input=d_input,
            d_output=1 if not model_config.gap else exp_config.sequence_length,
            sequence_length = exp_config.sequence_length,
            ad = exp_config.ad,
            monotonic = exp_config.monotonic,
            full_life = True if exp_config.approach == "full_life" else False
        )
    elif exp_config.model_name == "MLP":
        model = MLPModel(
            config=model_config,
            input_size=d_input,
            output_size=1 if not model_config.gap else exp_config.sequence_length,
            sequence_length = exp_config.sequence_length,
            ad = exp_config.ad,
            monotonic = exp_config.monotonic,
            full_life = True if exp_config.approach == "full_life" else False
        )
    elif exp_config.model_name == "Linear":
        model = LinearModel(
            config = model_config,
            input_size=d_input,
            output_size=1 if not model_config.gap else exp_config.sequence_length,
            sequence_length = exp_config.sequence_length,
            ad = exp_config.ad,
            monotonic = exp_config.monotonic,
            full_life = True if exp_config.approach == "full_life" else False
        )
    elif exp_config.model_name in ["RNN", "LSTM", "GRU"]:
        model = Recurrent_PDM(
            config=model_config,
            model_name=exp_config.model_name,
            input_size=d_input,
            output_size=exp_config.sequence_length,
            sequence_length = exp_config.sequence_length,
            ad = exp_config.ad,
            monotonic = exp_config.monotonic,
            full_life = True if exp_config.approach == "full_life" else False
        )
    elif exp_config.model_name == "RULTransformer":
        model = RULTransformer(
            config=model_config,
            input_size=d_input,
            output_size=exp_config.sequence_length,
            sequence_length = exp_config.sequence_length,
            ad = exp_config.ad,
            monotonic = exp_config.monotonic,
            full_life = True if exp_config.approach == "full_life" else False
        )
    elif exp_config.model_name == "RULInformer":
        model = RULInformer(
            config=model_config,
            d_input=d_input,
            d_output=exp_config.sequence_length,
            sequence_length = exp_config.sequence_length,
            ad = exp_config.ad,
            monotonic = exp_config.monotonic,
            full_life = True if exp_config.approach == "full_life" else False
        )
    else:
        raise ValueError(f"Model {exp_config.model_name} not recognized")

    if exp_config.model_summary:
        # Obtain the model summary with torchsummary
        try:
            if exp_config.summary_func == "torchinfo":
                print("#" * 50)
                print(f"Model summary computation with torchinfo:")
                print("#" * 50)
                input_size = (
                    (1, exp_config.sequence_length, d_input)
                    if ((not exp_config.quantile_reg) or (not model_config.tau_feat))
                    else (1, exp_config.sequence_length, d_input - 1)
                )
                model_summary = summary(
                    model=model, input_size=input_size, device=model_config.device
                )
                params = model_summary.total_params
                mult_adds = model_summary.total_mult_adds

                print("#" * 50)
                print(f"Total params: {params}")
                print(f"Total mult adds: {mult_adds}")
                print("#" * 50)

            elif exp_config.summary_func == "calflops":
                print("#" * 50)
                print(f"Model summary computation with calflops:")
                print("#" * 50)

                input_size = (
                    (1, exp_config.sequence_length, d_input)
                    if ((not exp_config.quantile_reg) or (not model_config.tau_feat))
                    else (1, exp_config.sequence_length, d_input - 1)
                )
                flops, mult_adds, params = calculate_flops(
                    model=model.to(model_config.device),
                    input_shape=input_size,
                    output_as_string=True,
                    output_precision=4,
                )

                print("#" * 50)
                print(f"{exp_config.model_name} model summary with calflops:")
                print(f"FLOPS: {flops}")
                print(f"mult_adds: {mult_adds}")
                print(f"Params: {params}")
                print("#" * 50)

        except Exception as e:
            mult_adds = None
            print("#" * 50)
            print(
                f"{exp_config.summary_func} not working, let's compute the model summary manually"
            )
            print("#" * 50)
            traceback.print_exc()  # Print the full traceback of the error
            params = model_summary_manual(model)

        if exp_config.save_summary_dict:
            summary_dict = {"params": params, "mult_adds": mult_adds}

            print("#" * 50)
            print(f"Model summary for {exp_config.model_name} model")
            print("#" * 50)

            summary_dict_dirpath = generate_path(
                basepath=experiment_path,
                folders=["summary_dict", exp_config.model_name],
            )
            save_element(
                element=summary_dict,
                dirpath=summary_dict_dirpath,
                filename=f"{exp_config.model_name}_summary_dict_{exp_config.summary_func}",
                filetype="pickle",
            )

            return model, summary_dict

    if exp_config.model_summary_manual:
        model_summary_manual(model)

    optimizer, scheduler = setup_optimizer(
        model,
        lr=exp_config.lr,
        weight_decay=exp_config.weight_decay,
        epochs=exp_config.epochs,
    )

    return model, optimizer, scheduler
