"""
Python module containing the implementation of the model head
blocks for the RUL models
"""

import torch
import torch.nn as nn

from exp_config import ModelConfig
from ssm_models import MonotonicLinear

class Head(nn.Module):
    def __init__(self, config: ModelConfig, d_output: int):
        """
        The head is responsible for combining the features
        extracted throught the extractor and perform the downstream
        task. In this case predicting the RUL signal.
        Normally the transformation done at this stage is:
        (B,L,H) → (B,O)
        """
        super().__init__()
        self.config = config
        self.d_output = d_output

        self.decoder = nn.LeakyLinear(self.d_output)

    def forward(self, x):

        x = x[:,-1,:].squeeze(1) # (B, L, H) -> (B,1,H) → (B, H)
        x = self.decoder(x) # (B,H) → (B,O)

        return x

class QuantileHead(Head):
    def __init__(self, tau: float = 0.5, *args, **kwargs):
        super().__init__(*args, *kwargs)
        self.tau = tau

    def forward(self, x):
        x = x[:,-1,:].squeeze(1) # (B, L, H) -> (B,1,H) → (B, H)
        x = self.decoder(x) * self.tau if self.config.tau_mult else self.decoder(x) # (B,H) → (B,O)

class MonotonicHead(Head):
    def __init__(self, d_output: int = 1000, *args, **kwargs):
        super().__init__(*args, *kwargs)

        if self.config.act == "gelu":
            self.activation = nn.GELU()
        elif self.config.act == "relu":
            self.activation = nn.ReLU()
        elif self.config.act == "silu":
            self.activation = nn.SiLU()
        elif self.config.act == "selu":
            self.activation = nn.SELU()
        elif self.config.act == "celu":
            self.activation = nn.CELU()
        else:
            self.activation = nn.Identity()

        self.mono = nn.ModuleList([
            MonotonicLinear(self.config.d_model, self.config.n_mono_neurons,pre_activation=nn.Identity()),
            *[MonotonicLinear(self.config.n_mono_neurons, self.config.n_mono_neurons, pre_activation=self.activation) for _ in range(self.config.n_mono_layers)],
            MonotonicLinear(self.config.n_mono_neurons, d_output, pre_activation=self.activation)
        ])

        def forward(self, x):

            for layer in self.mono:
                x = layer(x)

            return x

class MonoQuantileHead(MonotonicHead):

    def __init__(self, tau: float = 0.5, *args, **kwargs):
        super().__init__(*args, *kwargs)
        self.tau = tau

    def forward(self, x):

        for layer in self.mono:
            x = layer(x)

        x = x * self.tau if self.config.tau_mult else x
        return x

