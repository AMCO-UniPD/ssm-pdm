"""
Python module containing model classes (test for a new implementation)
"""

import os
import sys
import ipdb
import numpy as np

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

from utils import (
    ExperimentConfig,
    split_input,
    concat_tau,
    model_summary_manual
)
from exp_config import ModelConfig
from projector import Projector
from extractor import (
    Extractor,
    RecurrentExtractor,
    S4Extractor,
    S4DExtractor,
    S5Extractor,
    LinearExtractor,
    MLPExtractor,
    TransformerExtractor,
    InformerExtractor,
)

from head import (
    Head,
    QuantileHead,
    QuantileScaleHead,
    MonotonicHead,
    MonoQuantileHead,
)

cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
experiment_path = os.path.join(cwd, "experiments", "chronos_exp")


class RULModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        model_config: ModelConfig,
        output_size: int,
    ):
        """
        RULModel base class

        Args:
            model_name (str): name of the model
            model_config (ModelConfig): model configuration object
            output_size (int): output size
        """

        super().__init__()
        self.model_name = model_name
        self.model_config = model_config
        self.output_size = output_size

        self.projector = Projector(d_model=self.model_config.d_model)
        self.get_extractor()
        self.head = Head(config=self.model_config, d_output=self.output_size)

    @property
    def name(self) -> str:
        """
        Property to return the model name
        """
        return self.model_name

    def get_extractor(self) -> Extractor:
        """
        Method to select the specific extractor
        to use based on the model name
        """

        if self.name == "S4":
            self.extractor = S4Extractor(config=self.model_config)
        elif self.name == "S4D":
            self.extractor = S4DExtractor(config=self.model_config)
        elif self.name == "S5":
            self.extractor = S5Extractor(config=self.model_config)
        elif self.name == "Linear":
            self.extractor = LinearExtractor(config=self.model_config)
        elif self.name == "MLP":
            self.extractor = MLPExtractor(config=self.model_config)
        elif self.name in ["LSTM", "RNN", "GRU"]:
            self.extractor = RecurrentExtractor(
                model_name=self.name, config=self.model_config)
        elif self.name == "Transformer":
            self.extractor = TransformerExtractor(
                output_size = self.output_size, config=self.model_config
            )
        elif self.name == "Informer":
            self.extractor = InformerExtractor(
                 config=self.model_config
            )
        else:
            self.extractor = Extractor(config=self.model_config)

    def model_summary(self):
        """
        Function to manually compute the
        model summary
        """

        modules = [self.projector, self.extractor, self.head]
        total_params = 0

        for module in modules:
            print("-"*50)
            print(f"Computing model summary for {module.name}")
            print("-"*50)
            total_params += model_summary_manual(model=module)

        print("-"*50)
        print(
            f"Total number of parameters for model {self.name}: {total_params}")
        print("-"*50)

    def forward(self, x):
        """
        Typical method for implementing the forward pass of the model
        """

        if getattr(self, "debug_tau", False):
            print(f"[{type(self).__name__} backbone] tau={self.tau} "
                  "(backbone does not condition on tau)")

        x = self.projector(x)  # (B,L,D) → (B,L,H)
        x = self.extractor(x)  # (B,L,H) → (B,L,H)
        x = self.head(x)  # (B,L,H) → (B,O)

        return x


class QuantileRULModel(RULModel):
    """
    QuantileRULModel class used to define models using Quantile
    Regression

    Args:
        tau (float): The quantile level on which the model will be evaluated if the quantile regression approach is used
    """

    def __init__(self, tau: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tau = tau
        self.head = QuantileHead(
            tau=self.tau, config=self.model_config, d_output=self.output_size)

    def forward(self, x):
        if getattr(self, "debug_tau", False):
            print(f"[{type(self).__name__} backbone] tau={self.tau}, "
                  f"tau_feat={self.model_config.tau_feat}")

        x = concat_tau(
            x=x,
            tau=self.tau,
            device=self.model_config.device,
            tau_feat=self.model_config.tau_feat
        )

        x = self.projector(x)
        x = self.extractor(x)
        # Training and evaluation update model.tau after construction.
        self.head.tau = self.tau
        x = self.head(x)

        return x


class QuantileScaleRULModel(RULModel):
    """
    Quantile regression model using a location-and-scale head.

    The quantile level is consumed by the head, so it is neither concatenated
    to the input features nor multiplied with the prediction.
    """

    def __init__(self, tau: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tau = tau
        self.head = QuantileScaleHead(
            tau=self.tau,
            config=self.model_config,
            d_output=self.output_size,
        )

    def forward(self, x):
        # The backbone is independent of tau; the head must use the current level.
        self.head.tau = self.tau
        return super().forward(x)


class MonotonicRULModel(RULModel):
    """
    MonotonicRULModel class used to define models using Monotonic
    Regression

    Args:
        mono_mask (np.ndarray): boolean mask to identify monotonic features
    """

    def __init__(self, mono_mask: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mono_mask = mono_mask
        self.n_mono_neurons = len(np.where(mono_mask == 1)[0])
        self.head = MonotonicHead(
            config=self.model_config,
            d_output=self.output_size,
            n_mono_neurons=self.n_mono_neurons
        )

    def forward(self, x):

        x, x_mono = split_input(mask_mono=self.mono_mask,
                                inputs=x, device=self.model_config.device)

        x = self.projector(x)  # (B,L,D_nm) → (B,L,H)
        x = self.extractor(x)  # (B,L,H) → (B,L,H)

        # [(B,L,D_m), (B,L,H)] → (B,L,D_m+H)
        x = torch.cat((x, x_mono), dim=-1)
        x = self.head(x)  # (B,L,D_m+H)

        return x


class MonoQuantileRULModel(RULModel):
    """
    MonoQuantileRULModel class used to define monotonic models using Quantile Regression

    Args:
        tau (float): The quantile level on which the model will be evaluated if the quantile regression approach is used
        mono_mask (np.ndarray): boolean mask to identify monotonic features
    """

    def __init__(self, tau: float, mono_mask: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tau = tau
        self.mono_mask = mono_mask

        # NOTE: Here we have to use +1 in n_mono_neurons because the additional
        # feature added with tau_feat is
        # inserted in the monotonic features

        if self.model_config.tau_feat:
            self.n_mono_neurons = len(np.where(mono_mask == 1)[0]) + 1
        else:
            self.n_mono_neurons = len(np.where(mono_mask == 1)[0])

        self.head = MonoQuantileHead(
            config=self.model_config,
            d_output=self.output_size,
            tau=self.tau,
            n_mono_neurons=self.n_mono_neurons
        )

    def forward(self, x):

        x, x_mono = split_input(mask_mono=self.mono_mask,
                                inputs=x, device=self.model_config.device)

        # NOTE: The feature containing the quantile level is concatenated
        # to the monotonic features because it's constant and thus monotonic

        x_mono = concat_tau(
            x=x_mono,
            tau=self.tau,
            device=self.model_config.device,
            tau_feat=self.model_config.tau_feat
        )

        x = self.projector(x)  # (B,L,D_nm) → (B,L,H)
        x = self.extractor(x)  # (B,L,H) → (B,L,H)

        # [(B,L,D_m), (B,L,H)] → (B,L,D_m+H)
        x = torch.cat((x, x_mono), dim=-1)
        x = self.head(x)  # (B,L,D_m+H)

        return x
