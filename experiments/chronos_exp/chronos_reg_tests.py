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

from utils import *

# Load the configuration files
model_config_path="config/config.yaml"
exp_config_path="config/exp_config.yaml"
config=load_yaml_to_dict(model_config_path)
exp_config=load_yaml_to_dict(exp_config_path)

exp_config=ExperimentConfig(exp_config)

device = torch.device(f"cuda:{exp_config.device_num}" if torch.cuda.is_available() else "cpu")

df,regression_dataset,reg_loader=load_reg_data(exp_config)

model,tokenizer=load_model_tokenizer(model_config=config,exp_config=exp_config)
model.to(device)

for life,rul in reg_loader:
    life = life.to(device)
    rul = rul.to(device)
    input_ids,attention_mask,scale=tokenizer.context_input_transform(life)
    output=model(input_ids=input_ids,attention_mask=attention_mask)
    break
