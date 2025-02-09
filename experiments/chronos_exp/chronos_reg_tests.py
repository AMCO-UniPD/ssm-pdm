"""
Python script to try out some thing to make CHRONOS a regression model to perform directly RUL estimation
"""

import os
import sys
import ipdb
import torch

chronos_path = os.path.join(os.path.dirname(__file__),"..","..","src",
                            "chronos-forecasting","src")
chronos_path_train = os.path.join(os.path.dirname(__file__),"..","..","src",
                            "chronos-forecasting","scripts")
src_path = os.path.join(os.path.dirname(__file__),"..","..","src",
)
sys.path.append(src_path)
sys.path.append(chronos_path)
sys.path.append(chronos_path_train)

from utils import load_yaml_to_dict
from transformers import AutoModelForSequenceClassification
from chronos import MeanScaleUniformBins, ChronosConfig
from training.train import load_model
from ceruleo.dataset.catalog.CMAPSS import CMAPSSDataset

# Load the model with num_labels=1 to perform regression
chronos=AutoModelForSequenceClassification.from_pretrained("amazon/chronos-t5-small",num_labels=1)
# Load the model config file
config=load_yaml_to_dict("config/config.yaml")
# Create the ChronosConfig object
chronos_config=ChronosConfig(
    tokenizer_class=config["tokenizer_class"],
    tokenizer_kwargs=config["tokenizer_kwargs"],
    n_tokens=config["n_tokens"],
    n_special_tokens=config["n_special_tokens"],
    pad_token_id=config["pad_token_id"],
    eos_token_id=config["eos_token_id"],
    use_eos_token=config["use_eos_token"],
    model_type=config["model_type"],
    context_length=config["context_length"],
    prediction_length=config["prediction_length"],
    num_samples=config["num_samples"],
    temperature=config["temperature"],
    top_k=config["top_k"],
    top_p=config["top_p"],
)

model = load_model(
    model_id=config["model_id"],
    model_type=chronos_config.model_type,
    vocab_size=chronos_config.n_tokens,
    random_init=config["random_init"],
    tie_embeddings=config["tie_embeddings"],
    pad_token_id=chronos_config.pad_token_id,
    eos_token_id=chronos_config.eos_token_id,
)

chronos_tokenizer=MeanScaleUniformBins(
                                       low_limit=chronos_config.tokenizer_kwargs["low_limit"],
                                       high_limit=chronos_config.tokenizer_kwargs["high_limit"],
                                       config=chronos_config
                                      )


df = CMAPSSDataset(train=True,models="FD001")
life = df[0]
prompt=torch.tensor(life["SensorMeasure4"]).unsqueeze(0)

input_ids,attention_mask,scale=chronos_tokenizer.context_input_transform(prompt)

# Do inference
output=chronos(input_ids=input_ids,attention_mask=attention_mask)
