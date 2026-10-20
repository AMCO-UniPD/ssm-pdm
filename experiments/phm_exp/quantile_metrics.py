"""Aggregate saved per-life quantile errors and print a model comparison table.

Each life is first averaged across the selected runs/folds. Life_mean and
Life_std are then the mean and sample SD across those test lives, independently
for each quantile. Existing summary rows in input pickles are ignored.

Use --model_name S4 for one model, or --model_names S4 S5 for a comparison.
Without --exp_name, each model uses its own default experiment name. An explicit
--exp_name is used unchanged under each model directory. PHM_RESULTS_DIR keeps
its existing meaning. No predictions or training are performed here.

For legacy CMAPSS outputs use --data_name CMAPSS and CMAPSS_OUTPUTS_DIR.
This mode uses experiments/chronos_exp/config/quantile_metrics.yaml by default;
--report_dir writes the Markdown table and source provenance locally.
"""

import argparse
from copy import copy
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from metrics_summary import add_life_statistics, life_rows


def aggregate_metrics(metrics_dfs):
    """Average matching life/quantile cells across runs, then summarize lives.

    Life_std describes differences between the run-averaged test-life RMSEs,
    not variation between runs. A single test life has undefined sample SD.
    """
    if not metrics_dfs:
        raise ValueError("No runs/folds selected")
    frames = [life_rows(frame) for frame in metrics_dfs]
    reference = frames[0]
    if (
        reference.empty
        or not reference.index.is_unique
        or not reference.columns.is_unique
    ):
        raise ValueError(
            "Expected nonempty metrics with unique life and quantile labels"
        )
    aligned = []
    for frame in frames:
        if (
            not frame.index.is_unique
            or not frame.columns.is_unique
            or set(frame.index) != set(reference.index)
            or set(frame.columns) != set(reference.columns)
        ):
            raise ValueError(
                "All runs/folds must contain the same test lives and quantiles"
            )
        frame = frame.reindex(index=reference.index, columns=reference.columns)
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("Individual test-life metrics must be finite")
        aligned.append(frame)
    mean_metrics_df = (sum(aligned) / len(aligned)).round(2)
    # Use the stored life values so printing summaries remains idempotent.
    add_life_statistics(mean_metrics_df)
    mean_metrics_df.index.name = "Lifes"
    return mean_metrics_df


def load_model_metrics(config, metrics_path):
    """Read the latest saved per-quantile metrics for each selected run/fold."""
    if config.cv:
        run_ids = range(config.start_fold_id, min(config.stop_fold_id, config.n_folds))
        prefix = "fold"
    else:
        run_ids = range(config.start_run_id, config.n_runs)
        prefix = "run"
    frames = []
    for run in run_ids:
        columns = {}
        for quantile in config.quantiles:
            folder = Path(metrics_path) / f"{prefix}_{run + 1}" / f"quantile_{quantile}"
            files = list(folder.glob("*.pickle"))
            if not files:
                raise FileNotFoundError(f"No saved metric pickle in {folder}")
            latest = max(files, key=lambda path: path.stat().st_mtime)
            print(f"Reading {latest}")
            frame = life_rows(pd.read_pickle(latest))
            columns[f"quantile_{quantile}"] = frame["Eval Loss"]
        frames.append(pd.DataFrame(columns))
    return aggregate_metrics(frames)


def comparison_table(model_metrics):
    """Create model rows and quantile columns containing Life_mean +- Life_std."""
    rows = {}
    for model_name, frame in model_metrics.items():
        rows[model_name] = {
            quantile: f"{frame.at['Life_mean', quantile]:.2f} +- "
            f"{frame.at['Life_std', quantile]:.2f}"
            for quantile in frame.columns
        }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Models"
    return table


def main():
    # Read these existing options separately to distinguish an explicit list
    # from setup_exp's default model_names=["S4"]. Keep setup_exp's CLI intact.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--model_names", nargs="+", default=None)
    parser.add_argument("--exp_name", default="exp")
    parser.add_argument("--data_name")
    selected, _ = parser.parse_known_args()
    if selected.data_name == "CMAPSS":
        from cmapss_quantile_metrics import main as cmapss_main

        return cmapss_main()

    from exp_config import setup_exp, set_exp_name
    from perf import print_summary_metrics
    from phm_paths import results_root
    from utils import save_element

    config, _, _, exp_name = setup_exp()
    root = results_root(Path(__file__).resolve().parent)
    models = selected.model_names or [config.model_name]
    model_metrics = {}
    for model_name in dict.fromkeys(models):
        model_config = copy(config)
        model_config.model_name = model_name
        name = set_exp_name(model_config) if selected.exp_name == "exp" else exp_name
        metrics_path = (
            root / "metrics" / model_name / config.failure_type / config.approach / name
        )
        frame = load_model_metrics(model_config, metrics_path)
        if config.print_summary_metrics:
            print_summary_metrics(frame, model_name=model_name)
        if config.print_mean_metrics_df:
            print(f"{model_name} mean metrics df:")
            print(frame.to_markdown())
        if config.save_mean_metrics_df:
            dataset = f"_{config.cmapss_models}" if config.data_name == "CMAPSS" else ""
            filename = f"{model_name}{dataset}_{config.approach}_quantile_reg_global_metrics_df"
            save_element(
                frame, dirpath=str(metrics_path), filename=filename, filetype="pickle"
            )
        model_metrics[model_name] = frame

    print(
        "Life_mean +- Life_std: sample SD across test lives after averaging runs/folds;"
    )
    print("quantiles remain separate; nan means fewer than two test lives.")
    print(comparison_table(model_metrics).to_markdown())


if __name__ == "__main__":
    main()
