"""
Python script containing utility functions for the models of the `chronos-pdm` project
"""

import os
import sys
import ipdb
import torch
import torch.nn as nn
from typing import Tuple

chronos_path_src = os.path.join(os.path.dirname(__file__),"chronos-rul","src")
chronos_path_scripts = os.path.join(os.path.dirname(__file__),"chronos-rul","scripts")
sys.path.append(chronos_path_src)
sys.path.append(chronos_path_scripts)

from training.train import load_model
from chronos import ChronosModel, ChronosConfig, MeanScaleUniformBins
from transformers import AutoModelForSequenceClassification
from utils import ExperimentConfig, MeanScaleUniformBinsSensor

class RegressionHead(nn.Module):
    def __init__(self, sequence_length:int, hidden_size:int):
        super(RegressionHead, self).__init__()

        self.fc = nn.Linear(hidden_size, sequence_length)
 
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.fc(x) # (n_sensors,hidden_size) -> (n_sensors,sequence_length)
        return x

def load_model_tokenizer(
        model_config:ChronosConfig,
        exp_config:ExperimentConfig) -> Tuple[nn.Module, MeanScaleUniformBins]:
    """
    Load the model and the tokenizer for the `chronos` model

    Args:
        model_config (ChronosConfig): The configuration dictionary for the model
        exp_config (ExperimentConfig): The configuration dictionary for the experiment
        chronos_model (bool): Whether to load the `chronos` model or not

    Returns:
        model (torch.nn.Module): The model object
        tokenizer (MeanScaleUniformBins): The Transformer object
    """

    # Create the ChronosConfig object
    chronos_config=ChronosConfig(
        tokenizer_class=model_config["tokenizer_class"],
        tokenizer_kwargs=model_config["tokenizer_kwargs"],
        n_tokens=model_config["n_tokens"],
        n_special_tokens=model_config["n_special_tokens"],
        pad_token_id=model_config["pad_token_id"],
        eos_token_id=model_config["eos_token_id"],
        use_eos_token=model_config["use_eos_token"],
        model_type=model_config["model_type"],
        context_length=model_config["context_length"],
        prediction_length=model_config["prediction_length"],
        num_samples=model_config["num_samples"],
        temperature=model_config["temperature"],
        top_k=model_config["top_k"],
        top_p=model_config["top_p"],
    )

    tokenizer=MeanScaleUniformBinsSensor(
        low_limit=chronos_config.tokenizer_kwargs["low_limit"],
        high_limit=chronos_config.tokenizer_kwargs["high_limit"],
        config=chronos_config
    )

    model = AutoModelForSequenceClassification.from_pretrained(exp_config.model_id,num_labels=exp_config.num_labels)
    
    # The classification head can also be defined with a nn.Module
    new_classification_head = RegressionHead(sequence_length=exp_config.sequence_length, hidden_size=model.config.hidden_size)

    # Substitute classification_head with the new one
    model.classification_head = new_classification_head

    return model, tokenizer
