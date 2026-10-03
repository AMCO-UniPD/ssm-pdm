"""Focused checks for the constant/decreasing window weight ablation."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loss import WindowedPinballLoss, WindowedQuantileLoss, window_group_weights


@pytest.mark.parametrize("ratio", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_group_weight_ratio_and_fixed_total(ratio):
    groups = torch.tensor([True] * 100 + [False] * 10)
    weights = window_group_weights(groups, 100, 10, ratio)
    constant_total = weights[groups].sum().item()
    decreasing_total = weights[~groups].sum().item()

    assert constant_total + decreasing_total == pytest.approx(2.0)
    assert decreasing_total / constant_total == pytest.approx(ratio)


def test_current_and_unweighted_baselines():
    groups = torch.tensor([True] * 100 + [False] * 10)
    balanced = window_group_weights(groups, 100, 10, 1.0)
    uniform = window_group_weights(groups, 100, 10, None)

    assert balanced[0].item() == pytest.approx(0.01)
    assert balanced[-1].item() == pytest.approx(0.1)
    assert uniform[0].item() == pytest.approx(uniform[-1].item())
    assert uniform.sum().item() == pytest.approx(2.0)
    assert uniform[~groups].sum().item() / uniform[groups].sum().item() == pytest.approx(0.1)


@pytest.mark.parametrize("ratio", [0, -1, float("nan"), float("inf")])
def test_invalid_ratios_rejected(ratio):
    with pytest.raises(ValueError):
        window_group_weights(torch.tensor([True, False]), 1, 1, ratio)


def test_ratio_applies_to_windowed_pinball_loss():
    target = torch.tensor([[2.0, 2.0], [6.0, 0.0]])
    prediction = torch.zeros_like(target, requires_grad=True)
    mask = torch.ones_like(target, dtype=torch.bool)

    balanced = WindowedQuantileLoss(ratio=1.0)(prediction, target, mask, 2, 1, 0.5)
    decreasing_emphasis = WindowedQuantileLoss(ratio=2.0)(
        prediction, target, mask, 2, 1, 0.5
    )

    assert balanced.item() == pytest.approx(2.0)
    assert decreasing_emphasis.item() == pytest.approx(7 / 3)
    decreasing_emphasis.backward()
    assert prediction.grad is not None


def test_evaluation_pinball_uses_same_ratio():
    target = torch.tensor([[2.0, 2.0], [6.0, 0.0]])
    prediction = torch.zeros_like(target)
    mask = torch.ones_like(target, dtype=torch.bool)

    loss = WindowedPinballLoss(tau=0.5, ratio=2.0)(
        prediction, target, mask, 2, 1
    )

    assert loss.item() == pytest.approx(7 / 3)
