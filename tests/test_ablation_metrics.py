"""Checks for RMSE on the two PHM RUL regions."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ablation_metrics import rmse_by_rul_region


def test_region_rmse_splits_transition_window_by_target_value():
    # The first window crosses the 500-RUL plateau boundary; the final zero
    # is padding, matching the mask used by the existing life RMSE metric.
    target = np.array([[500, 500, 400, 300], [200, 100, 0, 0]])
    prediction = np.array([[490, 490, 380, 280], [180, 80, 999, 999]])

    scores = rmse_by_rul_region(prediction, target, max_rul=500)

    assert scores["constant"] == pytest.approx(10.0)
    assert scores["decreasing"] == pytest.approx(20.0)
    assert scores["overall"] == pytest.approx(np.sqrt(300.0))


def test_missing_plateau_does_not_hide_decreasing_rmse():
    scores = rmse_by_rul_region([190, 90], [200, 100], max_rul=500)

    assert scores["constant"] is None
    assert scores["decreasing"] == pytest.approx(10.0)
    assert scores["overall"] == pytest.approx(10.0)


def test_empty_or_mismatched_life_is_rejected():
    with pytest.raises(ValueError):
        rmse_by_rul_region([1, 2], [0, 0], max_rul=500)
    with pytest.raises(ValueError):
        rmse_by_rul_region([1], [100, 200], max_rul=500)
