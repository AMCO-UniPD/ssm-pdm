"""Load existing PHM predictions and export a compact row of business metrics.

--comparison quantiles matches plot_quantile_business_metrics.py (default).
--comparison models matches multi_plot_quantile_business_metrics.py.
--comparison standard matches plot_business_metrics.py.
All remaining arguments are the usual experiment configuration arguments.
"""

# ruff: noqa: E402 -- local source paths must be configured before project imports.

import argparse
import sys
from pathlib import Path
import ipdb

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))
sys.path.append(str(SRC / "ceruleo"))

import matplotlib.pyplot as plt
from ceruleo.results.results import PredictionResult

from business_metrics_plot import plot_business_metrics_compact
from config_vars import BASELINE_MODEL_NAMES
from exp_config import setup_exp
from perf import mask_outputs
from phm_paths import results_root
from utils import generate_path, get_current_time, get_most_recent_file, open_element


def load_results(exp_config, experiment_path, comparison):
    """Keep the baseline, run/fold, quantile and life selection conventions."""
    if len(exp_config.model_names) != len(exp_config.exp_names):
        raise ValueError("Provide one experiment name per model name")
    results = {}
    for model_name, exp_name in zip(exp_config.model_names, exp_config.exp_names):
        baseline = model_name in BASELINE_MODEL_NAMES
        quantiles = (
            (
                exp_config.quantiles
                if comparison == "quantiles"
                else [exp_config.quantile_run]
            )
            if not baseline and comparison != "standard"
            else [None]
        )
        for quantile in quantiles:
            if baseline:
                folders = ["baseline_outputs", model_name]
            else:
                folders = [
                    "combined_outputs",
                    model_name,
                    exp_config.failure_type,
                    exp_config.plot_approach,
                    exp_name,
                ]
                if comparison == "standard":
                    folders.append(f"run_{exp_config.plot_run_id}")
                else:
                    # Preserve the run selection of the original quantile scripts.
                    run_id = 5 if model_name == "RULTransformer" else 1
                    folders.extend(
                        [
                            f"fold_{run_id}" if exp_config.cv else f"run_{run_id}",
                            f"quantile_{quantile}",
                        ]
                    )
            outputs_path = generate_path(basepath=str(experiment_path), folders=folders)
            outputs_file = get_most_recent_file(outputs_path)
            print(f"Loading {model_name}: {outputs_file}")
            outputs = open_element(outputs_file, filetype="pickle")
            outputs = mask_outputs(outputs_dict=outputs, life_idx=exp_config.life_idx)
            name = (
                f"{model_name}_quantile_{quantile}"
                if comparison == "quantiles" and not baseline
                else model_name
            )
            results[name] = [
                PredictionResult(
                    name=f"Life_{i + 1}", true_RUL=true, predicted_RUL=pred
                )
                for i, (true, pred) in enumerate(
                    zip(outputs["y_true"], outputs["y_pred"])
                )
            ]
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument(
        "--comparison", choices=("quantiles", "models", "standard"), default="quantiles"
    )
    parser.add_argument("--figure_width", type=float, default=7.2)
    parser.add_argument("--figure_height", type=float, default=None)
    parser.add_argument("--font_size", type=float, default=8)
    parser.add_argument("--legend_columns", type=int, default=None)
    options, remaining = parser.parse_known_args()
    if "--help" in remaining or "-h" in remaining:
        parser.print_help()
    sys.argv[1:] = remaining
    exp_config, _, _, _ = setup_exp()
    experiment_path = results_root(Path(__file__).resolve().parent)
    results = load_results(exp_config, experiment_path, options.comparison)
    plot_path = experiment_path / "compact_business_metrics_plots" / options.comparison
    if options.comparison == "quantiles":
        plot_path /= exp_config.model_names[0]
    plot_path /= Path(exp_config.failure_type) / exp_config.plot_approach
    filename = (
        f"{get_current_time()}_{exp_config.failure_type}_{exp_config.plot_approach}"
    )
    fig = plot_business_metrics_compact(
        exp_config,
        results,
        plot_path,
        filename,
        figsize=(
            options.figure_width,
            options.figure_height
            if options.figure_height is not None
            else (2.25 if len(results) > 5 else 2.0),
        ),
        font_size=options.font_size,
        legend_columns=options.legend_columns,
    )
    if exp_config.show_plot:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
