"""
Python scirpt to test out a simple forecasting with chronos
I will use the example code provided in the github repo
"""

import os
import sys
import ipdb
import argparse

# Add the path to the sys path
chronos_path = os.path.join(os.path.dirname(__file__),"..","..","src",
                            "chronos-forecasting","src")
sys.path.append(chronos_path)

import pandas as pd
import matplotlib.pyplot as plt  # requires: pip install matplotlib
import torch
from chronos import BaseChronosPipeline, ChronosPipeline

parser=argparse.ArgumentParser(description="Simple forecasting with chronos")
parser.add_argument("--device_num",type=int,default=0,help="Device to use for inference")
parser.add_argument("--prediction_length",type=int,default=12,help="Number of steps to predict")
parser.add_argument("--quantile_levels",nargs='+', type=float, default=[0.1,0.5,0.9],help="Quantile levels to predict")

args=parser.parse_args()
config=args.__dict__

device=torch.device(f"cuda:{args.device_num}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pipeline
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-small",
    device_map=device,
    torch_dtype=torch.bfloat16,
)

# Load the data
df = pd.read_csv(
    "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
)

context=torch.tensor(df["#Passengers"])
embeddings,tokenizer_state=pipeline.embed(context)

ipdb.set_trace()

# Do inference
quantiles, mean = pipeline.predict_quantiles(
    context=torch.tensor(df["#Passengers"]),
    prediction_length=12,
    quantile_levels=[0.1, 0.5, 0.9],
)

filename="forecast.png"
plot_path=os.path.join(os.getcwd(),"plots",filename)

forecast_index = range(len(df), len(df) + 12)
low, median, high = quantiles[0, :, 0], quantiles[0, :, 1], quantiles[0, :, 2]

plt.figure(figsize=(8, 4))
plt.plot(df["#Passengers"], color="royalblue", label="historical data")
plt.plot(forecast_index, median, color="tomato", label="median forecast")
plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
plt.legend()
plt.grid()

plt.savefig(plot_path)
print(f"Plot saved at: {plot_path}")


