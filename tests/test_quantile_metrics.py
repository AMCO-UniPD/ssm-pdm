"""Synthetic-artifact checks; no training data or model dependencies required."""

# ruff: noqa: E402

import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/phm_exp"))

from metrics_summary import add_life_statistics
from quantile_metrics import aggregate_metrics, comparison_table, load_model_metrics


# Exercise the real printing function without importing perf's training stack.
perf_ast = ast.parse((ROOT / "src/perf.py").read_text())
printing_ast = ast.Module(
    body=[
        node
        for node in perf_ast.body
        if isinstance(node, ast.FunctionDef) and node.name == "print_summary_metrics"
    ],
    type_ignores=[],
)
namespace = {"pd": pd, "add_life_statistics": add_life_statistics}
exec(compile(printing_ast, str(ROOT / "src/perf.py"), "exec"), namespace)
print_summary_metrics = namespace["print_summary_metrics"]


def life_frame(values, column="quantile_0.1"):
    return pd.DataFrame(
        {column: values}, index=[f"Life_{i}" for i in range(len(values))]
    )


class QuantileMetricsTests(unittest.TestCase):
    def test_summaries_ignore_old_statistics_and_are_idempotent(self):
        frame = life_frame([10.0, 20.0, 60.0])
        frame.loc["Life_mean"] = 9999
        frame.loc["Life_median"] = -1000
        frame.loc["Life_std"] = 500
        add_life_statistics(frame)
        self.assertEqual(frame.at["Life_mean", "quantile_0.1"], 30.0)
        self.assertEqual(frame.at["Life_median", "quantile_0.1"], 20.0)
        self.assertEqual(
            frame.at["Life_std", "quantile_0.1"], round(np.std([10, 20, 60], ddof=1), 2)
        )
        expected = frame.copy()
        with redirect_stdout(io.StringIO()):
            print_summary_metrics(frame)
            print_summary_metrics(frame)
        assert_frame_equal(frame, expected)

    def test_run_average_precedes_life_std_and_keeps_quantiles_separate(self):
        first = life_frame([10.0, 30.0])
        first["quantile_0.9"] = [100.0, 200.0]
        second = life_frame([30.0, 10.0])
        second["quantile_0.9"] = [200.0, 400.0]
        first.loc["Life_std"] = 9999
        second.loc["Life_mean"] = -1000
        result = aggregate_metrics([first, second.iloc[::-1]])
        self.assertEqual(result.at["Life_mean", "quantile_0.1"], 20.0)
        self.assertEqual(result.at["Life_std", "quantile_0.1"], 0.0)
        self.assertEqual(result.at["Life_mean", "quantile_0.9"], 225.0)
        self.assertEqual(
            result.at["Life_std", "quantile_0.9"], round(np.std([150, 300], ddof=1), 2)
        )

    def test_print_flag_does_not_change_rounded_statistics(self):
        frame = aggregate_metrics([life_frame([0.0, 0.01]), life_frame([0.01, 0.02])])
        expected = frame.copy()
        with redirect_stdout(io.StringIO()):
            print_summary_metrics(frame)
        assert_frame_equal(frame, expected)

    def test_empty_missing_duplicate_and_nonfinite_inputs_fail(self):
        with self.assertRaisesRegex(ValueError, "No runs"):
            aggregate_metrics([])
        for bad in (
            life_frame([1.0]),
            life_frame([np.nan, 2.0]),
            life_frame([np.inf, 2.0]),
        ):
            with self.assertRaises(ValueError):
                aggregate_metrics([life_frame([1.0, 2.0]), bad])
        duplicate = life_frame([1.0, 2.0])
        duplicate.index = ["Life_0", "Life_0"]
        with self.assertRaises(ValueError):
            aggregate_metrics([duplicate])

    def test_single_life_sd_is_undefined(self):
        result = aggregate_metrics([life_frame([42.0])])
        self.assertTrue(pd.isna(result.at["Life_std", "quantile_0.1"]))
        self.assertEqual(
            comparison_table({"S4": result}).at["S4", "quantile_0.1"], "42.00 +- nan"
        )

    def test_model_quantile_markdown(self):
        frame = aggregate_metrics([life_frame([10.0, 30.0])])
        frame["quantile_0.9"] = frame["quantile_0.1"] * 2
        table = comparison_table({"S4": frame, "S5": frame * 2})
        self.assertEqual(list(table.index), ["S4", "S5"])
        self.assertEqual(list(table.columns), ["quantile_0.1", "quantile_0.9"])
        self.assertEqual(table.at["S4", "quantile_0.1"], "20.00 +- 14.14")
        self.assertIn("40.00 +- 28.28", table.to_markdown())

    def test_loads_selected_runs_and_folds_from_pickles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for prefix in ("run", "fold"):
                for number, values in (
                    (1, [100.0, 200.0]),
                    (2, [10.0, 30.0]),
                    (3, [30.0, 10.0]),
                ):
                    folder = root / f"{prefix}_{number}" / "quantile_0.1"
                    folder.mkdir(parents=True)
                    frame = life_frame(values, "Eval Loss")
                    frame.loc["Life_std"] = 99999
                    frame.to_pickle(folder / "metrics.pickle")
            for cv in (False, True):
                config = SimpleNamespace(
                    cv=cv,
                    start_run_id=1,
                    n_runs=3,
                    start_fold_id=1,
                    stop_fold_id=3,
                    n_folds=5,
                    quantiles=[0.1],
                )
                with redirect_stdout(io.StringIO()):
                    result = load_model_metrics(config, root)
                self.assertEqual(result.at["Life_mean", "quantile_0.1"], 20.0)
                self.assertEqual(result.at["Life_std", "quantile_0.1"], 0.0)
                config.quantiles = [0.9]
                with self.assertRaisesRegex(FileNotFoundError, "quantile_0.9"):
                    load_model_metrics(config, root)

    def test_main_prints_models_and_saves_same_metrics_with_print_flag(self):
        import quantile_metrics

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = SimpleNamespace(
                model_name="S4",
                failure_type="flow",
                approach="qr",
                cv=False,
                start_run_id=0,
                n_runs=1,
                quantiles=[0.1],
                print_summary_metrics=False,
                print_mean_metrics_df=False,
                save_mean_metrics_df=True,
                data_name="PHM",
            )
            for model in ("S4", "S5"):
                folder = (
                    root
                    / "metrics"
                    / model
                    / "flow/qr"
                    / f"default_{model}"
                    / "run_1/quantile_0.1"
                )
                folder.mkdir(parents=True)
                life_frame([10.0, 30.0], "Eval Loss").to_pickle(
                    folder / "metrics.pickle"
                )
            exp = ModuleType("exp_config")
            exp.setup_exp = lambda: (config, None, None, "unused")
            exp.set_exp_name = lambda cfg: f"default_{cfg.model_name}"
            perf = ModuleType("perf")
            perf.print_summary_metrics = print_summary_metrics
            utils = ModuleType("utils")
            saved = []
            utils.save_element = lambda frame, **kwargs: saved.append(
                (frame.copy(), kwargs)
            )
            with (
                patch.dict(
                    sys.modules, {"exp_config": exp, "perf": perf, "utils": utils}
                ),
                patch("phm_paths.results_root", return_value=root),
                patch.object(
                    sys, "argv", ["quantile_metrics.py", "--model_names", "S4", "S5"]
                ),
            ):
                for flag in (False, True):
                    config.print_summary_metrics = flag
                    output = io.StringIO()
                    with redirect_stdout(output):
                        quantile_metrics.main()
                    self.assertIn("20.00 +- 14.14", output.getvalue())
                    self.assertIn("| S4", output.getvalue())
                    self.assertIn("| S5", output.getvalue())
            self.assertEqual(len(saved), 4)
            assert_frame_equal(saved[0][0], saved[2][0])
            self.assertEqual(
                saved[0][1]["filename"], "S4_qr_quantile_reg_global_metrics_df"
            )


if __name__ == "__main__":
    unittest.main()
