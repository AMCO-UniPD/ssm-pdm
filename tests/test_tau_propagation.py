"""Regression checks for changing tau after model construction."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from exp_config import ModelConfig
from model_classes import RULModel, QuantileRULModel, QuantileScaleRULModel, MonoQuantileRULModel


class TauTests(unittest.TestCase):
    def make_model(self, cls, feat=False, mult=False):
        config = ModelConfig(d_model=4, n_layers=1, dropout=0, device='cpu', tau_feat=feat, tau_mult=mult)
        with patch.object(RULModel, 'get_extractor', lambda model: setattr(model, 'extractor', nn.Identity())):
            return cls(model_name='Linear', model_config=config, output_size=1,
                       **({'mono_mask': np.array([False, True])} if cls is MonoQuantileRULModel else {}), tau=0.5).eval()

    def test_multiplier_uses_updated_tau_in_both_models(self):
        x = torch.ones(2, 3, 2)
        for cls in (QuantileRULModel, MonoQuantileRULModel):
            with self.subTest(model=cls.__name__):
                model = self.make_model(cls, mult=True)
                model.tau = 0.25
                lower = model(x)
                model.tau = 0.75
                upper = model(x)
                torch.testing.assert_close(upper, lower * 3)
                self.assertEqual(model.head.tau, 0.75)

    def test_feature_path_receives_current_tau(self):
        model = self.make_model(QuantileRULModel, feat=True)
        captured = []
        hook = model.projector.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
        for tau in (0.1, 0.9):
            model.tau = tau
            model(torch.ones(2, 3, 2))
        hook.remove()
        torch.testing.assert_close(captured[0][:, :, -1], torch.full((2, 3), 0.1))
        torch.testing.assert_close(captured[1][:, :, -1], torch.full((2, 3), 0.9))

    def test_scale_quantiles_order_and_scale_gradients(self):
        model = self.make_model(QuantileScaleRULModel)
        x = torch.ones(2, 3, 2)
        predictions = []
        for tau in (0.1, 0.5, 0.9):
            model.tau = tau
            predictions.append(model(x))
        self.assertTrue(torch.all(predictions[0] < predictions[1]))
        self.assertTrue(torch.all(predictions[1] < predictions[2]))
        for tau, decoder in ((0.1, model.head.lower_scale_decoder), (0.9, model.head.upper_scale_decoder)):
            model.zero_grad(set_to_none=True)
            model.tau = tau
            model(x).sum().backward()
            self.assertGreater(decoder.weight.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
