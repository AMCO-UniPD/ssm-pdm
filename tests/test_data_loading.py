import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "src" / "ceruleo"))

from ceruleo.dataset.ts_dataset import PDMInMemoryDataset
from ceruleo.transformation import Transformer
from ceruleo.transformation.features.imputers import MeanImputer
from ceruleo.transformation.features.scalers import MinMaxScaler
from ceruleo.transformation.features.selection import ByNameFeatureSelector
from ceruleo.transformation.functional.pipeline.pipeline import make_pipeline
from utils import (
    SSMRegressionDataset,
    SSMWindowRegressionDataset,
    TransData,
    _load_trans_data,
    _save_trans_data,
    fit_phm_transformer,
)


def _dataset():
    lives = [
        pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0], "RUL": [2, 1, 0]}),
        pd.DataFrame({"a": [2.0, 4.0], "b": [8.0, 10.0], "RUL": [1, 0]}),
    ]
    return PDMInMemoryDataset(lives, rul_column="RUL")


def _transformer():
    return Transformer(
        pipelineX=make_pipeline(
            ByNameFeatureSelector(features=["a", "b"]),
            MeanImputer(),
            MinMaxScaler(range=(0, 1)),
        ),
        pipelineY=make_pipeline(ByNameFeatureSelector(features=["RUL"])),
    )


def test_parallel_fit_preserves_transform_values():
    dataset = _dataset()
    legacy = _transformer().fit(dataset)
    parallel = fit_phm_transformer(_transformer(), dataset, workers=2)

    for life in dataset:
        expected_x, expected_y, _ = legacy.transform(life)
        actual_x, actual_y, _ = parallel.transform(life)
        np.testing.assert_array_equal(actual_x.values, expected_x.values)
        np.testing.assert_array_equal(actual_y.values, expected_y.values)


def test_tensor_backed_datasets_match_legacy_float32_values():
    dataset = _dataset()
    transformer = _transformer().fit(dataset)
    transformed = dataset.map(transformer)
    data = TransData(transformed, workers=2)

    windowed = SSMWindowRegressionDataset(data, sequence_length=4, stride=2)
    sequence, target, mask = windowed[0]
    expected_x, expected_y, _ = transformer.transform(dataset[0])
    expected_x = np.concatenate((expected_x.values, np.zeros((1, 2))))
    expected_y = np.concatenate((expected_y.values.reshape(-1), np.zeros(1)))
    np.testing.assert_array_equal(sequence.numpy(), expected_x.astype(np.float32))
    np.testing.assert_array_equal(target.squeeze(-1).numpy(), expected_y.astype(np.float32))
    assert sequence.dtype == target.dtype == mask.dtype == torch.float32

    padded = SSMRegressionDataset(data, sequence_length=4)
    sequence, target, mask = padded[0]
    np.testing.assert_array_equal(sequence.numpy(), expected_x.astype(np.float32))
    np.testing.assert_array_equal(target.numpy(), expected_y.astype(np.float32))


def test_transformed_array_cache_round_trip(tmp_path):
    transformer = _transformer().fit(_dataset())
    data = TransData(_dataset().map(transformer))
    metadata = _save_trans_data(tmp_path, "train", data)
    cached = _load_trans_data(tmp_path, metadata)

    assert cached.feature_names == data.feature_names
    assert len(cached) == len(data)
    for expected_x, expected_y, actual_x, actual_y in zip(
        data.feature_arrays,
        data.target_arrays,
        cached.feature_arrays,
        cached.target_arrays,
    ):
        np.testing.assert_array_equal(actual_x, expected_x)
        np.testing.assert_array_equal(actual_y, expected_y)
