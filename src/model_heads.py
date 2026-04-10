"""
Python module with different type of model heads
"""

import torch
import torch.nn as nn

from ssm_models import MonotonicLinear, ModelConfig

class GapHead(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        d_output: int,
    ):
        """
        Model Regression head for the GAP approach. The embeddings
        coming from the feature extractor are averaged over the time dimension
        (i.e. axis 1)
        to produce the output.
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
        Model Regression head for the AD approach. In this case the
        dimension in output is d_input because we are using a reconstruction approach
        so we want to reconstruct the input.
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

class MonotonicHead(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        sequence_length: int,
    ):
        """
        Model Head for monotonic neural networks (works both for padding and monotonic approach).
        With this model head (that needs to have at least 4 layers
        to be a universal approximator) the model should be monotonic, so it should produce always
        non increasing predictions for the RUL
        """
        super().__init__()

        self.tau_mult = config.tau_mult

        self.decoder = nn.Sequential(
            MonotonicLinear(config.d_model, config.d_model, pre_activation=nn.Identity()),
            MonotonicLinear(config.d_model, config.d_model, pre_activation=nn.ReLU()),
            MonotonicLinear(config.d_model, config.d_model, pre_activation=nn.ReLU()),
            MonotonicLinear(config.d_model, sequence_length, pre_activation=nn.ReLU()),
        )

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
