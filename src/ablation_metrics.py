"""Unweighted test metrics for the PHM window-weight ablation."""

import numpy as np


def rmse_by_rul_region(prediction, target, max_rul):
    """Return RMSE for all valid samples and the two clipped-RUL regions.

    The saved PHM outputs use zero targets for padding. Match the existing
    ``lifes_metrics`` mask by excluding those entries. The constant region is
    the clipped RUL plateau, and the decreasing region is below that plateau.
    A region without valid samples has an RMSE of ``None``.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target must have the same shape")
    if max_rul <= 0:
        raise ValueError("max_rul must be positive")

    valid = target != 0
    if not np.any(valid):
        raise ValueError("The test life has no valid RUL samples")

    plateau = np.isclose(target, max_rul, rtol=0, atol=1e-5)
    masks = {
        "overall": valid,
        "constant": valid & plateau,
        "decreasing": valid & (target < max_rul) & ~plateau,
    }
    squared_error = np.square(prediction - target)
    return {
        region: float(np.sqrt(np.mean(squared_error[mask]))) if np.any(mask) else None
        for region, mask in masks.items()
    }
