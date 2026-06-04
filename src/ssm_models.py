"""
Python module containing utility functions for the models migrated from the
`SSM_PDM` project into the `chronos_pdm` project.
"""

from informer import *
from s5 import S5Block
from s4d import S4D
from s4 import S4Block as S4
from s4 import DropoutNd
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

from utils import save_element, generate_path
from exp_config import ExperimentConfig, ModelConfig


chronos_path_src = os.path.join(
    os.path.dirname(__file__), "chronos-rul", "src")
imports_path = os.path.join(os.path.dirname(__file__), "AD_MG", "src")
sys.path.append(chronos_path_src)
sys.path.append(imports_path)

# s4 imports

# s4d imports

# s5 imports

# informer imports

cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
experiment_path = os.path.join(cwd, "experiments", "chronos_exp")


