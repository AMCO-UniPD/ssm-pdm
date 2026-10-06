"""Checks for RMSE on the two PHM RUL regions."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ablation_metrics import (
    interval_by_rul_region,
    pinball_by_rul_region,
    rmse_by_rul_region,
)


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


def test_pinball_is_asymmetric_and_excludes_padding():
    target = [500, 500, 400, 300, 0]
    prediction = [490, 510, 380, 320, 999]

    scores = pinball_by_rul_region(prediction, target, 500, tau=0.1)

    assert scores == pytest.approx({"overall": 7.5, "constant": 5, "decreasing": 10})
    assert pinball_by_rul_region([190], [200], 500, 0.1)["overall"] == 1
    assert pinball_by_rul_region([210], [200], 500, 0.1)["overall"] == 9


def test_interval_inclusive_coverage_width_and_crossings():
    scores = interval_by_rul_region(
        [500, 490, 410, 290, -999],
        [510, 500, 390, 295, 999],
        [500, 500, 400, 300, 0],
        500,
    )

    assert scores["coverage"] == pytest.approx(
        {"overall": 0.5, "constant": 1, "decreasing": 0}
    )
    assert scores["width"] == pytest.approx(
        {"overall": 1.25, "constant": 10, "decreasing": -7.5}
    )
    assert scores["crossing_rate"] == pytest.approx(
        {"overall": 0.25, "constant": 0, "decreasing": 0.5}
    )


def test_uncertainty_missing_region_returns_none():
    pinball = pinball_by_rul_region([190], [200], 500, 0.1)
    intervals = interval_by_rul_region([190], [210], [200], 500)

    assert pinball["constant"] is None
    assert all(metric["constant"] is None for metric in intervals.values())
    assert intervals["coverage"]["decreasing"] == 1


@pytest.mark.parametrize("tau", [0, 1, -0.1, float("nan")])
def test_invalid_quantile_is_rejected(tau):
    with pytest.raises(ValueError, match="tau"):
        pinball_by_rul_region([190], [200], 500, tau)


def test_uncertainty_rejects_bad_shapes_and_nonfinite_valid_values():
    with pytest.raises(ValueError, match="shape"):
        interval_by_rul_region([190], [210, 220], [200], 500)
    with pytest.raises(ValueError, match="finite"):
        pinball_by_rul_region([float("nan")], [200], 500, 0.1)
    # Invalid padded predictions have no effect on the score.
    scores = pinball_by_rul_region([190, float("nan")], [200, 0], 500, 0.1)
    assert scores["overall"] == 1
