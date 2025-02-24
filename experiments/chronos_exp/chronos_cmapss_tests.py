"""
Python script to try out the first pre processing steps on the `CMAPSS`
dataset for the `chronos-pdm`project
"""

import os
import sys
import ipdb
import torch
import pandas as pd

src_path = os.path.join(os.path.dirname(__file__),"..","..","src",
)
sys.path.append(src_path)

# imports from other files
from utils import TransData, RegressionDataset

# ceruleo imports
from ceruleo.dataset.catalog.CMAPSS import CMAPSSDataset
from ceruleo.dataset.catalog.CMAPSS import sensor_indices
from ceruleo.transformation import Transformer
from ceruleo.transformation.features.selection import ByNameFeatureSelector, PandasVarianceThreshold
from ceruleo.transformation.functional.pipeline.pipeline import make_pipeline
from ceruleo.transformation.features.scalers import MinMaxScaler,RobustMinMaxScaler,StandardScaler,RobustStandardScaler

# torch imports
from torch.utils.data import DataLoader

# Load the CMAPSS dataset
df = CMAPSSDataset(train=True,models="FD001")
# ceruleo transformation to select only the sensor_indices
FEATURES = [df[0].columns[i] for i in sensor_indices]
transformer = Transformer(
pipelineX=make_pipeline(
	ByNameFeatureSelector(features=FEATURES),
	MinMaxScaler(range=(-1, 1))
	# StandardScaler()
),

pipelineY=make_pipeline(
	ByNameFeatureSelector(features=['RUL']),
	)
)

transformer.fit(df)
transformed_df=df.map(transformer)

# Create the TransData object
lifes=TransData(transformed_df)

# Create a RegressionDataset
regression_dataset = RegressionDataset(lifes)

# Create a DataLoader
reg_loader=DataLoader(regression_dataset,batch_size=1,shuffle=False)


