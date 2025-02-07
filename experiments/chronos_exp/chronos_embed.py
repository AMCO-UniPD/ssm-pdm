"""
Python script to access encoder embedding and test some things on the chronos internals
Also here using the code suggested in the github repo
"""

import os
import sys
import ipdb

# Add the path to the sys path
chronos_path = os.path.join(os.path.dirname(__file__),"..","..","src",
                            "chronos-forecasting","src")
sys.path.append(chronos_path)

import pandas as pd
import torch
from chronos import ChronosPipeline, ChronosBoltPipeline

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

pipeline_bolt = ChronosBoltPipeline.from_pretrained(
    "amazon/chronos-bolt-small",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

df = pd.read_csv("https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv")

# context must be either a 1D tensor, a list of 1D tensors,
# or a left-padded 2D tensor with batch as the first dimension
context = torch.tensor(df["#Passengers"])
embeddings, tokenizer_state = pipeline.embed(context)
embeddings_bolt, tokenizer_state_bolt = pipeline_bolt.embed(context)

