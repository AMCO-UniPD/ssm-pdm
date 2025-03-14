"""
Python script containing utility functions for the models migrated from the
`SSM_PDM` project into the `chronos_pdm` project.
"""

import os
import sys
import ipdb
from typing import Tuple
from dataclasses import dataclass

# torch imports
import torch
import torch.nn as nn
from torch.nn.functional import dropout
import torch.optim as optim
from torch.optim import lr_scheduler
from torchinfo import summary
from transformer_encoder import TransformerEncoder
from transformer_encoder.utils import PositionalEncoding

from utils import ExperimentConfig


chronos_path_src = os.path.join(os.path.dirname(__file__),"chronos-rul","src")
imports_path = os.path.join(os.path.dirname(__file__),"AD_MG","src")
sys.path.append(chronos_path_src)
sys.path.append(imports_path)

# s4 imports
from s4 import DropoutNd
from s4 import S4Block as S4
# s4d imports
from s4d import S4D
# s5 imports
from s5 import S5, S5Block
# informer imports
from informer import *

cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
experiment_path = os.path.join(cwd, "experiments", "chronos_exp")

# dataclass for the model configuration

@dataclass
class ModelConfig:

    def __init__(self,config:dict):

        for key,value in config.items():
            setattr(self,key,value)

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
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

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

# RNN based models
    
class Recurrent_PDM(nn.Module):
    def __init__(
            self,
            config: ModelConfig,
            model_name: str,
            input_size: int,
            output_size: int,
    ):
        super(Recurrent_PDM, self).__init__()

        self.quantile_reg = config.quantile_reg
        self.device = config.device

        if model_name == "LSTM":
            self.recurrent = nn.LSTM(
                input_size=input_size,
                hidden_size=config.d_model,
                num_layers=config.n_layers,
                batch_first=True,
                dropout=config.dropout
            )
            # self.bn = nn.BatchNorm1d(hidden_size)
        elif model_name == "GRU":
            self.recurrent = nn.GRU(
                input_size=input_size,
                hidden_size=config.d_model,
                num_layers=config.n_layers,
                batch_first=True,
                dropout=config.dropout
            )
        elif model_name == "RNN":
            self.recurrent = nn.RNN(
                input_size=input_size,
                hidden_size=config.d_model,
                num_layers=config.n_layers,
                batch_first=True,
                dropout=config.dropout
            )

        self.fc = nn.Linear(config.d_model, output_size)
    
    def forward(self, x, tau=None):
        
        if self.quantile_reg:
            assert tau is not None, "tau must be provided for quantile regression"
            assert isinstance(tau, float), "tau must be a float"
            assert 0 <= tau <= 1, "tau must be between 0 and 1"
            # Concatenate the tau value to the inputs
            x = torch.cat([x, torch.ones(x.shape[0],x.shape[1],1).to(self.device) * tau], dim=-1)

        out = self.recurrent(x) # (B, L, D) -> (B, L, H)
        out = out[0].mean(dim=1) # (B, L, H) -> (B, H)
        out = self.fc(out) # (B, H) -> (B, L)
        # out = self.fc(out[0]).squeeze(-1) # (B, L, H) -> (B, L)
        return out

# SSM model classes

class S4Model(nn.Module):

    def __init__(
        self,
        config:ModelConfig,
        d_input:int,
        d_output:int,
        lr,
    ):
        super().__init__()

        self.prenorm = config.prenorm
        self.gap = config.gap
        self.quantile_reg = config.quantile_reg
        self.device = config.device
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
                S4(d_model,
                   dropout=dropout,
                   activation=activation,
                   gate_act=gate_act,
                   mult_act=mult_act,
                   final_act=final_act,
                   transposed=True,
                   lr=min(0.001, lr)))
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(DropoutNd(dropout))

        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x, tau=None):

        if self.quantile_reg:
            assert tau is not None, "tau must be provided for quantile regression"
            assert isinstance(tau, float), "tau must be a float"
            assert 0 <= tau <= 1, "tau must be between 0 and 1"
            # Concatenate the tau value to the inputs
            x = torch.cat([x, torch.ones(x.shape[0],x.shape[1],1).to(self.device) * tau], dim=-1)

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

        x = x.transpose(-1, -2) # (B, d_model, L) -> (B, L, d_model)

        if self.gap:
            x = x.mean(dim=1) # (B, L, d_model) -> (B, d_model)
            x = self.decoder(x)  # (B, d_model) -> (B, d_output)
            return x

        # Decode the outputs
        x = self.decoder(x).squeeze(-1)  # (B,L,d_model) -> (B,L)
        return x

class S4DModel(nn.Module):

    def __init__(
        self,
        config:ModelConfig,
        d_input:int,
        d_output:int,
    ):
        super().__init__()

        self.gap = config.gap
        self.quantile_reg = config.quantile_reg
        self.device = config.device
        d_state = config.d_state
        act = config.act
        d_model = config.d_model
        n_layers = config.n_layers
        dropout = config.dropout
        d_input = d_input

        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4D layers as residual blocks
        self.s4d_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4d_layers.append(
                S4D(d_model=d_model,
                    d_output=d_output,
                    d_state=d_state,
                    dropout=dropout,
                    act=act,
                    transposed=True)
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(DropoutNd(dropout))

        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x, tau=None):

        if self.quantile_reg:
            assert tau is not None, "tau must be provided for quantile regression"
            assert isinstance(tau, float), "tau must be a float"
            assert 0 <= tau <= 1, "tau must be between 0 and 1"
            # Concatenate the tau value to the inputs
            x = torch.cat([x, torch.ones(x.shape[0],x.shape[1],1).to(self.device) * tau], dim=-1)

        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4d_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

        x = x.transpose(-1, -2)

        if self.gap:
            x = x.mean(dim=1) # (B, L, d_model) -> (B, d_model)
            x = self.decoder(x)  # (B, d_model) -> (B, d_output)
            return x

        # Decode the outputs
        x = self.decoder(x).squeeze(-1)  # (B,L,d_model) -> (B,L)
        return x

class S5Model(nn.Module):

    def __init__(
        self,
        config:ModelConfig,
        d_input:int,
        d_output:int,
    ):
        super().__init__()

        self.gap = config.gap
        self.quantile_reg = config.quantile_reg
        self.device = config.device
        d_model = config.d_model
        n_layers = config.n_layers
        bidir = config.bidir

        self.encoder = nn.Linear(d_input, d_model)

        # Stack S5 layers as residual blocks
        self.s5_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.s5_layers.append(
                S5Block(dim=d_input,
                        state_dim=d_model,
                        bidir=bidir))


        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x, tau=None):

        if self.quantile_reg:
            assert tau is not None, "tau must be provided for quantile regression"
            assert isinstance(tau, float), "tau must be a float"
            assert 0 <= tau <= 1, "tau must be between 0 and 1"
            # Concatenate the tau value to the inputs
            x = torch.cat([x, torch.ones(x.shape[0],x.shape[1],1).to(self.device) * tau], dim=-1)

        for layer in self.s5_layers: # (B, L, H) -> (B, L, H). The P is used inside here (black box we do not care)
            x = layer(x)

        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        if self.gap:
            x = x.mean(dim=1) # (B, L, d_model) -> (B, d_model)
            x = self.decoder(x)  # (B, d_model) -> (B, d_output)
            return x

        # Decode the outputs
        x = self.decoder(x).squeeze(-1)  # (B,L,d_model) -> (B,L)
        return x

# Transformer based models

class RULTransformer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        input_size: int,
        output_size: int,
    ):
        super(RULTransformer, self).__init__()

        self.quantile_reg = config.quantile_reg
        self.device = config.device

        self.embedding = nn.Sequential(
            nn.Embedding(
                num_embeddings=input_size,
                embedding_dim=config.d_model
            ),
            PositionalEncoding(
                d_model=config.d_model,
                dropout=config.dropout,
                max_len=output_size
            )
        )

        self.encoder = TransformerEncoder(
            d_model=config.d_model,
            d_ff=config.d_ff,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dropout=config.dropout
        )
        
        self.decoder = nn.Linear(
            in_features=config.d_model,
            out_features=output_size
        )
        
    def forward(self, x, mask=None, tau=None):

        if mask is None:
            mask = torch.zeros(x.size(0),x.size(1)).to(x.device)

        if self.quantile_reg:
            assert tau is not None, "tau must be provided for quantile regression"
            assert isinstance(tau, float), "tau must be a float"
            assert 0 <= tau <= 1, "tau must be between 0 and 1"
            # Concatenate the tau value to the inputs
            x = torch.cat([x, torch.ones(x.shape[0],x.shape[1],1).to(self.device) * tau], dim=-1)

        x = x.argmax(dim=-1) # (B, L, d_input) -> (B, L)
        x = self.embedding(x) # (B, L) -> (B, L, d_model)
        x = self.encoder(x, mask) # (B, L, d_model) -> (B, L, d_model)
        x = x.mean(dim=1) # (B, L, d_model) -> (B, d_model)
        x = self.decoder(x) # (B, d_model) -> (B, d_output)
        # x = self.decoder(x).squeeze(-1) # (B, L, d_model) -> (B, L)
        return x
        

# Informer based model

class RULInformer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        d_input:int,
        d_output:int,
        output_attention:bool=False
       ):
        super(RULInformer,self).__init__()

        self.quantile_reg = config.quantile_reg
        self.device = config.device
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(
            c_in=d_input,
            d_model=config.d_model,
            dropout=config.dropout
        )
        # Attention
        Attn = ProbAttention if config.attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            attn_layers = [
                EncoderLayer(
                    attention = AttentionLayer(
                        attention = Attn(
                            mask_flag = False,
                            factor = config.factor,
                            attention_dropout=config.dropout,
                            output_attention=True
                        ),
                        d_model = config.d_model,
                        n_heads = config.n_heads,
                        mix=False),
                    d_model = config.d_model,
                    d_ff = config.d_ff,
                    dropout=config.dropout,
                    activation=config.inf_activation
                ) for _ in range(config.n_layers)
            ],
            conv_layers = [
                ConvLayer(
                    config.d_model
                ) for _ in range(config.n_layers-1)
            ] if config.distil else None,
            norm_layer=torch.nn.LayerNorm(config.d_model)
        )

        self.projection = nn.Linear(config.d_model, d_output)

    def forward(self, x_enc, output_attention=False, enc_self_mask=None, tau=None):

        if self.quantile_reg:
            assert tau is not None, "tau must be provided for quantile regression"
            assert isinstance(tau, float), "tau must be a float"
            assert 0 <= tau <= 1, "tau must be between 0 and 1"
            # Concatenate the tau value to the inputs
            x_enc = torch.cat([x_enc, torch.ones(x_enc.shape[0],x_enc.shape[1],1).to(self.device) * tau], dim=-1)

        enc_out = self.enc_embedding(x_enc) # [B,L,D] -> [B,L,H]
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask) # [B,L,H] -> [B,L,H]

        enc_out = enc_out.mean(dim=1) # [B,L,H] -> [B,H]
        dec_out = self.projection(enc_out) # [B,L,H] -> [B,L]

        if output_attention:
            return dec_out, attns
        else:
            return dec_out

# Manual parameter count computation in case torchinfo does not work

def model_summary_manual(
    model: nn.Module
) -> None:
    
    """
    Manual version of torchinfo summary module.

    Args:
        model: nn.Module object

    Returns:
        The function prints out the number of trainable and non trainable parameters of the model and does not return anything.
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
        print('#'* 50)

    print("#" * 50)
    print("Model Summary:")
    print("#" * 50)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")

# Function to create the model

def load_ssm_model(
        exp_config:ExperimentConfig,
        model_config:ModelConfig,
        d_input: int,
) -> Tuple[nn.Module, optim.Optimizer, optim.lr_scheduler]:

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
        model = S4Model(config=model_config,
                        d_input=d_input,
                        d_output=1 if not model_config.gap else exp_config.sequence_length,
                        lr=exp_config.lr)
    elif exp_config.model_name == "S4D":
        model = S4DModel(config=model_config,
                         d_input=d_input,
                         d_output=1 if not model_config.gap else exp_config.sequence_length)
    elif exp_config.model_name == "S5":
        model = S5Model(config=model_config,
                        d_input=d_input,
                        d_output=1 if not model_config.gap else exp_config.sequence_length)
    elif exp_config.model_name in ["RNN","LSTM","GRU"]:
        model = Recurrent_PDM(config=model_config,
                              model_name=exp_config.model_name,
                              input_size=d_input,
                              output_size=exp_config.sequence_length)
    elif exp_config.model_name == "RULTransformer":
        model = RULTransformer(
            config=model_config,
            input_size=d_input,
            output_size=exp_config.sequence_length
        )
    elif exp_config.model_name == "RULInformer":
        model = RULInformer(
            config=model_config,
            d_input=d_input,
            d_output=exp_config.sequence_length,
            output_attention=exp_config.output_attention
        )
    else:
        raise ValueError(f"Model {exp_config.model_name} not recognized")

    if exp_config.model_summary:

        # Obtain the model summary with torchsummary
        try:
            model_summary=summary(model, input_size=(1, exp_config.sequence_length, d_input))

            print('#'*50)
            print(f"Total params: {model_summary.total_params}")
            print(f"Total mult adds: {model_summary.total_mult_adds}")
            print('#'*50)
        except Exception as e:
            print('#'*50)
            print("torchinfo summary not working, let's use the manual computation")
            print('#'*50)
            model_summary_manual(model)

    if exp_config.model_summary_manual:
        model_summary_manual(model)

    optimizer, scheduler = setup_optimizer(
                                       model,
                                       lr=exp_config.lr,
                                       weight_decay=exp_config.weight_decay,
                                       epochs=exp_config.epochs)

    return model, optimizer, scheduler
