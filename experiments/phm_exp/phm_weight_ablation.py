"""Train the PHM quantile model for each window-weight ratio and collect accuracy and uncertainty metrics.

Run through ``run_phm_weight_ablation`` from this directory. The underlying
SQR trainer samples quantiles during training; one model is trained per fold
and ratio, then its quantile predictions are evaluated on the PHM test lives.
"""

import argparse
import csv
import math
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR.parent.parent / "src"))

from ablation_metrics import (
    interval_by_rul_region,
    pinball_by_rul_region,
    rmse_by_rul_region,
)
from phm_monitor import emit
from cv_training import train_k_fold
from exp_config import ExperimentConfig, ModelConfig, check_arguments
from utils import get_most_recent_file, load_yaml_to_dict, open_element


EVALUATION_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
REGIONS = ("overall", "constant", "decreasing")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/ssm_exp_config.yaml")
    parser.add_argument("--model-name", default="S4D")
    parser.add_argument("--failure-type", default="flow_low")
    parser.add_argument("--train-phm-tools", nargs="+", required=True)
    parser.add_argument("--test-phm-tools", nargs="+", required=True)
    parser.add_argument("--device-num", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ratios",
        nargs="+",
        default=["unweighted", "0.25", "0.5", "1", "2", "4"],
        help="Total decreasing/constant loss ratios; 'unweighted' is the baseline",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable W&B (disabled by default)")
    parser.add_argument("--results-dir", default="ablation_results/window_weights")
    return parser.parse_args()


def parse_ratios(values):
    ratios = []
    for value in values:
        if value == "unweighted":
            ratio = None
        else:
            ratio = float(value)
            if not math.isfinite(ratio) or ratio <= 0:
                raise ValueError(f"Invalid window-weight ratio: {value}")
        if ratio not in ratios:
            ratios.append(ratio)
    return ratios


def label_for(ratio):
    return "unweighted" if ratio is None else f"ratio_{ratio:g}"


def save_csv(path, fieldnames, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_plot(path, summaries, region, metric="rmse", nominal_coverage=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = list(range(len(summaries)))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(
        x,
        np.asarray([row[f"mean_{region}_{metric}"] for row in summaries], dtype=float),
        yerr=np.asarray(
            [row[f"std_{region}_{metric}"] for row in summaries], dtype=float
        ),
        fmt="o-",
        capsize=4,
    )
    ax.set_xticks(x, [row["setting"] for row in summaries])
    units = "fraction" if metric in ("coverage", "crossing_rate") else "RUL units"
    ax.set_ylabel(
        f"{region.capitalize()} {metric.replace('_', ' ')} ({units}; mean ± SD)"
    )
    if nominal_coverage is not None:
        ax.axhline(
            nominal_coverage, linestyle="--", color="gray", label="Nominal coverage"
        )
        ax.legend()

    ax.set_xlabel("Total decreasing / constant loss ratio")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    temporary = path.with_name("." + path.name)
    fig.savefig(temporary, dpi=200)
    temporary.replace(path)
    plt.close(fig)


def mean_over_lives(scores):
    means = {}
    for region in REGIONS:
        values = [score[region] for score in scores if score[region] is not None]
        means[region] = statistics.mean(values) if values else None
    return means


def uncertainty_rows(outputs_by_tau, max_rul, setting, fold):
    """Require aligned lives/targets before computing intervals across quantiles."""
    reference = outputs_by_tau[0.5]["y_true"]
    if not reference:
        raise ValueError("No test lives available")
    for outputs in outputs_by_tau.values():
        if len(outputs["y_pred"]) != len(reference) or len(outputs["y_true"]) != len(
            reference
        ):
            raise ValueError("Quantile outputs must have the same test life counts")
        if any(
            not np.array_equal(target, expected)
            for target, expected in zip(outputs["y_true"], reference)
        ):
            raise ValueError("Quantile outputs must have identical target ordering")

    pinball_rows = []
    for tau, outputs in outputs_by_tau.items():
        scores = mean_over_lives(
            [
                pinball_by_rul_region(pred, target, max_rul, tau)
                for pred, target in zip(outputs["y_pred"], reference)
            ]
        )
        pinball_rows.append(
            {
                "setting": setting,
                "fold": fold,
                "quantile": tau,
                **{f"{region}_pinball": value for region, value in scores.items()},
            }
        )

    intervals = [
        interval_by_rul_region(lower, upper, target, max_rul)
        for lower, upper, target in zip(
            outputs_by_tau[0.1]["y_pred"], outputs_by_tau[0.9]["y_pred"], reference
        )
    ]
    interval_row = {
        "setting": setting,
        "fold": fold,
        "lower_quantile": 0.1,
        "upper_quantile": 0.9,
        "nominal_coverage": 0.8,
    }
    for metric in ("coverage", "width", "crossing_rate"):
        scores = mean_over_lives([item[metric] for item in intervals])
        interval_row.update(
            {f"{region}_{metric}": value for region, value in scores.items()}
        )
    return pinball_rows, interval_row


def summarize(rows, identifiers, metrics):
    summary = {**identifiers, "completed_folds": len(rows)}
    for region in REGIONS:
        for metric in metrics:
            key = f"{region}_{metric}"
            values = [row[key] for row in rows if row[key] is not None]
            summary[f"mean_{key}"] = statistics.mean(values) if values else None
            summary[f"std_{key}"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
            )
    return summary


def save_uncertainty_results(results_dir, pinball_rows, interval_rows):
    pinball_summaries = []
    interval_summaries = []
    settings = list(dict.fromkeys(row["setting"] for row in interval_rows))
    for setting in settings:
        for tau in EVALUATION_QUANTILES:
            rows = [
                row
                for row in pinball_rows
                if row["setting"] == setting and row["quantile"] == tau
            ]
            pinball_summaries.append(
                summarize(rows, {"setting": setting, "quantile": tau}, ("pinball",))
            )
        rows = [row for row in interval_rows if row["setting"] == setting]
        interval_summaries.append(
            summarize(
                rows,
                {
                    key: rows[0][key]
                    for key in (
                        "setting",
                        "lower_quantile",
                        "upper_quantile",
                        "nominal_coverage",
                    )
                },
                ("coverage", "width", "crossing_rate"),
            )
        )

    for name, rows in (
        ("fold_pinball", pinball_rows),
        ("summary_pinball", pinball_summaries),
        ("fold_intervals", interval_rows),
        ("summary_intervals", interval_summaries),
    ):
        save_csv(results_dir / f"{name}.csv", list(rows[0]), rows)
    for region in REGIONS:
        for tau in EVALUATION_QUANTILES:
            rows = [row for row in pinball_summaries if row["quantile"] == tau]
            save_plot(
                results_dir / f"{region}_pinball_quantile_{tau}_by_weight.png",
                rows,
                region,
                "pinball",
            )
        for metric in ("coverage", "width", "crossing_rate"):
            save_plot(
                results_dir / f"{region}_{metric}_80_by_weight.png",
                interval_summaries,
                region,
                metric,
                nominal_coverage=0.8 if metric == "coverage" else None,
            )


def main():
    args = parse_args()
    ratios = parse_ratios(args.ratios)
    config_path = (EXPERIMENT_DIR / args.config).resolve()
    config = ExperimentConfig.from_dict(load_yaml_to_dict(str(config_path)))
    model_path = Path(config.model_config_path)
    if not model_path.is_absolute():
        model_path = EXPERIMENT_DIR / model_path
    model_config = ModelConfig.from_dict(load_yaml_to_dict(str(model_path)))

    config.data_name = "PHM"
    config.model_name = args.model_name
    config.failure_type = args.failure_type
    config.train_phm_tools = args.train_phm_tools
    config.test_phm_tools = args.test_phm_tools
    config.quantiles = [0.5]
    config.quantile_run = 0.5
    config.use_wandb = args.wandb
    config.resume_training = False
    config.save_best_model = True
    config.save_outputs = True
    config.save_combined_outputs = False
    config.compute_metrics = True
    config.save_metrics_df = True
    config.return_outputs = False
    config.get_test_idx = False
    check_arguments(config)

    if config.loss != "window_quantile_reg" or not config.quantile_reg:
        raise ValueError("The ablation requires the windowed SQR training loss")
    if config.quantile_scale and (model_config.tau_feat or model_config.tau_mult):
        raise ValueError("Scale head requires tau_feat=false and tau_mult=false")
    if not config.quantile_scale and not (model_config.tau_feat or model_config.tau_mult):
        raise ValueError("QuantileHead requires tau_feat or tau_mult")
    if not config.cv:
        raise ValueError("The ablation requires cross validation")
    if config.start_fold_id != 0 or config.stop_fold_id != config.n_folds:
        raise ValueError("Run all configured folds for a comparable ablation")
    if config.transformer_type not in (1, 2, 5) or config.max_rul != 500:
        raise ValueError("Regional RMSE requires the PHM RUL target clipped at 500")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_device(args.device_num)
    model_config.device = device
    results_dir = (EXPERIMENT_DIR / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    pinball_rows = []
    interval_rows = []

    def persist_results():
        summaries = []
        for label in dict.fromkeys(row["setting"] for row in fold_rows):
            rows = [row for row in fold_rows if row["setting"] == label]
            summary = summarize(rows, {"setting": label, "completed_folds": len(rows)}, ("rmse",))
            summaries.append(summary)
        save_csv(results_dir / "fold_rmse.csv", list(fold_rows[0]), fold_rows)
        save_csv(results_dir / "summary_rmse.csv", list(summaries[0]), summaries)
        for region in REGIONS:
            save_plot(results_dir / f"{region}_rmse_by_weight.png", summaries, region)
        save_uncertainty_results(results_dir, pinball_rows, interval_rows)

    for ratio in ratios:
        setting = label_for(ratio)
        config.window_weight_ratio = ratio
        exp_name = f"{args.model_name}_{args.failure_type}_window_weight_{setting}"
        print(f"\nRunning {setting} on {device}", flush=True)
        emit("setting_started", setting=setting, ratio=ratio, stage="training", status="running")

        def output_dir(name):
            path = (
                Path(os.environ.get("PHM_RESULTS_DIR", EXPERIMENT_DIR))
                / name
                / args.model_name
                / args.failure_type
                / config.approach
                / exp_name
            )
            path.mkdir(parents=True, exist_ok=True)
            return str(path)

        def collect_fold(fold_idx):
            # Use the saved predictions to avoid the two-decimal rounding in
            # lifes_metrics. Match its nonzero-target mask and mean over lives.
            output_path = (
                Path(output_dir("outputs")) / f"fold_{fold_idx}" / "quantile_0.5"
            )
            outputs_by_tau = {
                tau: open_element(
                    get_most_recent_file(str(output_path.parent / f"quantile_{tau}")),
                    filetype="pickle",
                )
                for tau in EVALUATION_QUANTILES
            }
            fold_pinball, fold_interval = uncertainty_rows(
                outputs_by_tau, config.max_rul, setting, fold_idx
            )
            pinball_rows.extend(fold_pinball)
            interval_rows.append(fold_interval)
            outputs = outputs_by_tau[0.5]
            if len(outputs["y_pred"]) != len(outputs["y_true"]):
                raise RuntimeError(
                    f"Prediction/target life count differs in {output_path}"
                )
            life_metrics = []
            for prediction, target in zip(outputs["y_pred"], outputs["y_true"]):
                life_metrics.append(
                    rmse_by_rul_region(prediction, target, config.max_rul)
                )
            fold_result = {"setting": setting, "fold": fold_idx}
            for region in ("overall", "constant", "decreasing"):
                values = [
                    item[region] for item in life_metrics if item[region] is not None
                ]
                if region == "decreasing" and not values:
                    raise ValueError(f"No decreasing RUL samples in {output_path}")
                fold_result[f"{region}_rmse"] = (
                    statistics.mean(values) if values else None
                )
            fold_rows.append(fold_result)

            emit("ablation_fold_evaluated", setting=setting, fold=fold_idx,
                 rmse=fold_result, intervals=fold_interval, pinball=fold_pinball)
            persist_results()

        metrics, _ = train_k_fold(
            exp_config=config, model_config=model_config, device=device,
            best_model_path=output_dir("best_models"), outputs_path=output_dir("outputs"),
            combined_outputs_path=output_dir("combined_outputs"), metrics_path=output_dir("metrics"),
            seed=args.seed, evaluation_quantiles=list(EVALUATION_QUANTILES),
            fold_complete_callback=collect_fold,
        )
        if len(metrics) != config.n_folds:
            raise RuntimeError(f"Expected {config.n_folds} fold results for {setting}; got {len(metrics)}")
        emit("setting_completed", setting=setting, completed_folds=config.n_folds)

    print(f"Saved ablation results to {results_dir}")


if __name__ == "__main__":
    main()
