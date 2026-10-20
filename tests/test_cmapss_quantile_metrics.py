"""Legacy raw-output compatibility, padding masks, and end-to-end CMAPSS CLI."""

# ruff: noqa: E402

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/phm_exp"))
from cmapss_quantile_metrics import (
    latest_output,
    load_output_metrics,
    output_life_metrics,
)


class CmapssQuantileMetricsTests(unittest.TestCase):
    def test_per_life_rmse_excludes_padding_and_does_not_pool_lives(self):
        result = output_life_metrics(
            {
                "y_pred": [np.array([7, 4, np.nan]), np.array([[15], [9]])],
                "y_true": [np.array([10, 8, 0]), np.array([[20], [0]])],
            },
            51,
        )
        self.assertEqual(list(result.index), ["Life_51", "Life_52"])
        np.testing.assert_allclose(result.to_numpy(), [np.sqrt(12.5), 5.0])

    def test_invalid_predictions_targets_and_lengths_fail(self):
        for output in (
            {"y_pred": [], "y_true": []},
            {"y_pred": [[1]], "y_true": [[1], [2]]},
            {"y_pred": [[1, 2]], "y_true": [[1]]},
            {"y_pred": [[1, 2]], "y_true": [[0, 0]]},
            {"y_pred": [[np.nan]], "y_true": [[1]]},
            {"y_pred": [[1]], "y_true": [[np.inf]]},
        ):
            with self.subTest(output=output), self.assertRaises(ValueError):
                output_life_metrics(output)

    def test_file_position_and_missing_quantile_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name, timestamp in [("old", 1), ("new", 2)]:
                path = folder / f"{name}.pickle"
                pd.to_pickle({}, path)
                os.utime(path, (timestamp, timestamp))
            self.assertEqual(latest_output(folder).name, "new.pickle")
            self.assertEqual(latest_output(folder, 1).name, "old.pickle")
            with self.assertRaises(FileNotFoundError):
                latest_output(folder, 2)
            with self.assertRaises(FileNotFoundError):
                load_output_metrics(folder, range(1), [0.1])

    def test_target_or_life_order_changes_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for quantile, target in [(0.1, [[10, 5]]), (0.9, [[11, 5]])]:
                folder = root / "run_1" / f"quantile_{quantile}"
                folder.mkdir(parents=True)
                pd.to_pickle(
                    {"y_pred": [[9, 4]], "y_true": target}, folder / "outputs.pickle"
                )
            with self.assertRaisesRegex(ValueError, "life order"):
                load_output_metrics(root, range(1), [0.1, 0.9])

    def test_cli_reads_environment_and_reports_explicit_experiments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = root / "external outputs"
            config = {
                "model_names": ["S4", "LSTM"],
                "exp_names": ["experiment_S4", "experiment_LSTM"],
                "quantiles": [0.1, 0.9],
                "n_runs": 2,
                "start_run_id": 0,
                "test_idx": [50, 100],
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config))
            for model in config["model_names"]:
                for run, errors in [(1, [1.0, 3.0]), (2, [3.0, 5.0])]:
                    for tau in config["quantiles"]:
                        folder = (
                            outputs
                            / model
                            / "FD001/windowed/quantile_reg"
                            / f"experiment_{model}"
                            / f"run_{run}"
                            / f"quantile_{tau}"
                        )
                        folder.mkdir(parents=True)
                        pd.to_pickle(
                            {
                                "y_pred": [
                                    [10 - errors[0], 999],
                                    [20 - errors[1], 999],
                                ],
                                "y_true": [[10, 0], [20, 0]],
                            },
                            folder / "outputs.pickle",
                        )
            report = root / "report"
            command = [
                sys.executable,
                str(ROOT / "experiments/phm_exp/quantile_metrics.py"),
                "--data_name",
                "CMAPSS",
                "--exp_config_path",
                str(config_path),
                "--report_dir",
                str(report),
            ]
            baseline = None
            for flag in ([], ["--print_summary_metrics"]):
                result = subprocess.run(
                    command + flag,
                    env={**os.environ, "CMAPSS_OUTPUTS_DIR": str(outputs)},
                    text=True,
                    capture_output=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                markdown = (report / "table.md").read_text()
                self.assertIn("3.00 +- 1.41", markdown)
                self.assertIn("LSTM", markdown)
                if baseline is not None:
                    self.assertEqual(markdown, baseline)
                baseline = markdown
            provenance = json.loads((report / "provenance.json").read_text())
            self.assertEqual(provenance["outputs_root"], str(outputs))
            self.assertEqual(provenance["run_ids"], [1, 2])
            self.assertEqual(len(provenance["models"][0]["sources"]), 4)
            self.assertEqual(len(provenance["models"][0]["sources"][0]["sha256"]), 64)
            frame = pd.read_pickle(report / "S4_mean_metrics.pickle")
            self.assertEqual(frame.loc["Life_mean", "quantile_0.1"], 3.0)
            self.assertEqual(frame.loc["Life_std", "quantile_0.1"], 1.41)
            # Explicit single-model selection retains the configured mapping.
            result = subprocess.run(
                command + ["--model_name", "LSTM"],
                env={**os.environ, "CMAPSS_OUTPUTS_DIR": str(outputs)},
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                [
                    row["model"]
                    for row in json.loads((report / "provenance.json").read_text())[
                        "models"
                    ]
                ],
                ["LSTM"],
            )


if __name__ == "__main__":
    unittest.main()
