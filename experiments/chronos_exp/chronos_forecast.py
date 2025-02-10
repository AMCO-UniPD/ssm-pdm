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
src_path = os.path.join(os.path.dirname(__file__),"..","..","src",
)
sys.path.append(src_path)
sys.path.append(chronos_path)

import pandas as pd
import matplotlib.pyplot as plt  # requires: pip install matplotlib
import torch
from utils import get_current_time
from plots import plot_forecast
from chronos import BaseChronosPipeline, ChronosPipeline
from ceruleo.dataset.catalog.CMAPSS import CMAPSSDataset

parser=argparse.ArgumentParser(description="Simple forecasting with chronos")
parser.add_argument("--device_num",type=int,default=0,help="Device to use for inference")
parser.add_argument("--prediction_length",type=int,default=12,help="Number of steps to predict")
parser.add_argument("--quantile_levels",nargs='+', type=float, default=[0.1,0.5,0.9],help="Quantile levels to predict")
parser.add_argument("--dataset", type=str, default="CMAPSS", help="Dataset to use for forecasting")
parser.add_argument("--life_idx", type=int, default=0, help="Index of the CMAPSS life to consider")
parser.add_argument("--cmapss_model", type=str, default="FD001", help="CMAPSS model to use for forecasting")
parser.add_argument("--sensor_num", type=int, default=4, help="Number of the sensor to use for forecasting")

args=parser.parse_args()
config=args.__dict__

device=torch.device(f"cuda:{args.device_num}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pipeline
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-base",
    device_map=device,
    torch_dtype=torch.bfloat16,
)

# Load the data
if args.dataset=="CMAPSS":
    df = CMAPSSDataset(train=True,models=args.cmapss_model)
    life = df[args.life_idx]
    sensors_idx=[f"SensorMeasure{args.sensor_num}",f"SensorMeasure{args.sensor_num+1}"]
    prompt=torch.tensor(life[sensors_idx].values)
else:
    df = pd.read_csv(
        "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
    )
    prompt=torch.tensor(df["#Passengers"])

prompt_list = [prompt[:-args.prediction_length,i] for i in range(prompt.shape[1])]

# Do inference
quantiles, mean = pipeline.predict_quantiles(
    context=prompt_list,
    prediction_length=args.prediction_length,
    quantile_levels=args.quantile_levels,
)

ipdb.set_trace()

plot_path=os.path.join(os.getcwd(),"plots")
# Plot the forecast
fig = plot_forecast(life=life,
                    prompt=prompt,
                    quantile_levels=args.quantile_levels,
                    pred_quantiles=quantiles,
                    prediction_length=args.prediction_length,
                    sensor_num=args.sensor_num,
                    data_name=args.dataset,
                    plot_path=plot_path)

