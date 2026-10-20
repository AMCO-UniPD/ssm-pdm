"""Checks for the CMAPSS profiler and its recurrent model execution path."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exp_config import ModelConfig  # noqa: E402
from extractor import RecurrentExtractor  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "cmapss_profile", ROOT / "experiments/chronos_exp/profile_benchmark_models.py"
)
profiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiler)


class CmapssProfileTests(unittest.TestCase):
    def test_recurrent_layers_receive_gradients_and_use_sequence_history(self):
        torch.manual_seed(0)
        for name in ("LSTM", "GRU", "RNN"):
            with self.subTest(model=name):
                extractor = RecurrentExtractor(
                    model_name=name, config=ModelConfig(d_model=4, n_layers=2)
                )
                x = torch.randn(2, 5, 4)
                output = extractor(x)
                self.assertEqual(output.shape, x.shape)
                # A last-step prediction must depend on earlier timesteps.
                changed = x.clone()
                changed[:, 0, :] += 10
                self.assertFalse(torch.allclose(output[:, -1], extractor(changed)[:, -1]))
                output[:, -1, 0].sum().backward()
                for parameter in extractor.recurrent.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_legacy_benchmark_config_and_report(self):
        args = profiler.argparse.Namespace(
            exp_config=profiler.EXPERIMENT_PATH / "config/ssm_exp_config.yaml",
            model_config=None, models=None, input_features=None,
            sequence_length=None, tau_feat=None, tau_mult=None,
        )
        config, model_config, names, features, model_path = profiler.load_configs(args)
        self.assertEqual(features, 14)
        self.assertEqual(config.sequence_length, 170)
        self.assertEqual(config.cmapss_models, "FD001")
        self.assertIn("LSTM", names)
        report = profiler.render_markdown(
            [("LSTM", 1000, 1000, 262144, None)], config, model_config, features,
            args.exp_config, model_path,
        )
        self.assertIn("| LSTM | 1,000 | 1,000 | 0.250000 |", report)
        self.assertIn("(1, 170, 14)", report)
        self.assertIn("torch.save(model.state_dict(), buffer)", report)
        self.assertNotIn("GFLOPs", report)

    def test_torchinfo_count_and_manual_fallback(self):
        model = torch.nn.Linear(3, 2)
        count, error = profiler.torchinfo_parameters(model, (1, 3))
        self.assertEqual(count, 8)
        self.assertIsNone(error)
        with patch("torchinfo.summary", side_effect=RuntimeError("profiler failed")):
            count, error = profiler.torchinfo_parameters(model, (1, 3))
        self.assertEqual(count, 8)
        self.assertIn("RuntimeError: profiler failed", error)

    def test_legacy_s5_runs_blocks_before_projection(self):
        from exp_config import ExperimentConfig

        model = profiler.build_legacy_s5(
            ExperimentConfig(sequence_length=5),
            ModelConfig(d_model=8, n_layers=1, tau_feat=True), feature_count=2,
        )
        seen = []
        hook = model.encoder.register_forward_pre_hook(
            lambda module, args: seen.append(args[0].shape)
        )
        self.assertEqual(model(torch.zeros(1, 5, 2)).shape, (1, 5))
        hook.remove()
        self.assertEqual(seen, [torch.Size([1, 5, 3])])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
