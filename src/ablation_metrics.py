"""Unweighted test metrics for the PHM window-weight ablation."""

import numpy as np


def _region_masks(target, max_rul):
    target = np.asarray(target, dtype=np.float64)
    if not np.isfinite(max_rul) or max_rul <= 0:
        raise ValueError("max_rul must be positive and finite")
    if not np.all(np.isfinite(target)):
        raise ValueError("Targets must be finite")
    valid = target != 0
    if not np.any(valid):
        raise ValueError("The test life has no valid RUL samples")
    plateau = np.isclose(target, max_rul, rtol=0, atol=1e-5)
    return {
        "overall": valid,
        "constant": valid & plateau,
        "decreasing": valid & (target < max_rul) & ~plateau,
    }


def _prediction_array(prediction, target, masks):
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target must have the same shape")
    if not np.all(np.isfinite(prediction[masks["overall"]])):
        raise ValueError("Valid predictions must be finite")
    return prediction


def _region_means(values, masks):
    return {
        region: float(np.mean(values[mask])) if np.any(mask) else None
        for region, mask in masks.items()
    }


def rmse_by_rul_region(prediction, target, max_rul):
    """Return RMSE for all valid samples and the two clipped-RUL regions.

    The saved PHM outputs use zero targets for padding. Match the existing
    ``lifes_metrics`` mask by excluding those entries. The constant region is
    the clipped RUL plateau, and the decreasing region is below that plateau.
    A region without valid samples has an RMSE of ``None``.
    """
    target = np.asarray(target, dtype=np.float64)
    masks = _region_masks(target, max_rul)
    prediction = _prediction_array(prediction, target, masks)
    means = _region_means(np.square(prediction - target), masks)
    return {
        region: np.sqrt(value) if value is not None else None
        for region, value in means.items()
    }


def pinball_by_rul_region(prediction, target, max_rul, tau):
    """Unweighted sample-mean pinball loss in original RUL units."""
    if not np.isfinite(tau) or not 0 < tau < 1:
        raise ValueError("tau must be strictly between zero and one")
    target = np.asarray(target, dtype=np.float64)
    masks = _region_masks(target, max_rul)
    prediction = _prediction_array(prediction, target, masks)
    error = target - prediction
    return _region_means(np.maximum(tau * error, (tau - 1) * error), masks)


def interval_by_rul_region(lower, upper, target, max_rul):
    """Inclusive coverage, raw width, and crossing rate, without sorting bounds.

    Rates are fractions in [0, 1]. Crossed bounds never cover the target;
    their signed widths remain visible rather than silently repairing them.
    """
    target = np.asarray(target, dtype=np.float64)
    masks = _region_masks(target, max_rul)
    lower = _prediction_array(lower, target, masks)
    upper = _prediction_array(upper, target, masks)
    return {
        "coverage": _region_means((lower <= target) & (target <= upper), masks),
        "width": _region_means(upper - lower, masks),
        "crossing_rate": _region_means(lower > upper, masks),
    }
