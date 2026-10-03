"""Train the PHM quantile model for each window-weight ratio and collect RMSE.

Run through ``run_phm_weight_ablation`` from this directory. The underlying
SQR trainer samples quantiles during training; one model is trained per fold
and ratio, then its median prediction is evaluated on the PHM test lives.
"""

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR.parent.parent / "src"))

from ablation_metrics import rmse_by_rul_region
from cv_training import train_k_fold
from exp_config import ExperimentConfig, ModelConfig, check_arguments
from utils import get_most_recent_file, load_yaml_to_dict, open_element


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
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path, summaries, region):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = list(range(len(summaries)))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(
        x,
        [row[f"mean_{region}_rmse"] for row in summaries],
        yerr=[row[f"std_{region}_rmse"] for row in summaries],
        fmt="o-",
        capsize=4,
    )
    ax.set_xticks(x, [row["setting"] for row in summaries])
    ax.set_ylabel(f"{region.capitalize()} RUL RMSE (mean ± SD across folds)")
    ax.set_xlabel("Total decreasing / constant loss ratio")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


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
    config.use_wandb = False
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
    summary_rows = []
    for ratio in ratios:
        setting = label_for(ratio)
        config.window_weight_ratio = ratio
        exp_name = f"{args.model_name}_{args.failure_type}_window_weight_{setting}"
        print(f"\nRunning {setting} on {device}", flush=True)

        def output_dir(name):
            return str(
                EXPERIMENT_DIR / name / args.model_name / args.failure_type
                / config.approach / exp_name
            )

        metrics, _ = train_k_fold(
            exp_config=config,
            model_config=model_config,
            device=device,
            best_model_path=output_dir("best_models"),
            outputs_path=output_dir("outputs"),
            combined_outputs_path=output_dir("combined_outputs"),
            metrics_path=output_dir("metrics"),
            seed=args.seed,
        )
        if len(metrics) != config.n_folds:
            raise RuntimeError(
                f"Expected {config.n_folds} fold results for {setting}; got {len(metrics)}"
            )

        fold_metrics = []
        for fold_idx, _ in enumerate(metrics, start=1):
            # Use the saved predictions to avoid the two-decimal rounding in
            # lifes_metrics. Match its nonzero-target mask and mean over lives.
            output_path = (
                Path(output_dir("outputs")) / f"fold_{fold_idx}" / "quantile_0.5"
            )
            outputs = open_element(get_most_recent_file(str(output_path)), filetype="pickle")
            if len(outputs["y_pred"]) != len(outputs["y_true"]):
                raise RuntimeError(f"Prediction/target life count differs in {output_path}")
            life_metrics = []
            for prediction, target in zip(outputs["y_pred"], outputs["y_true"]):
                life_metrics.append(
                    rmse_by_rul_region(prediction, target, config.max_rul)
                )
            fold_result = {"setting": setting, "fold": fold_idx}
            for region in ("overall", "constant", "decreasing"):
                values = [item[region] for item in life_metrics if item[region] is not None]
                if region == "decreasing" and not values:
                    raise ValueError(f"No decreasing RUL samples in {output_path}")
                fold_result[f"{region}_rmse"] = statistics.mean(values) if values else None
            fold_metrics.append(fold_result)
            fold_rows.append(fold_result)

        summary = {"setting": setting}
        for region in ("overall", "constant", "decreasing"):
            values = [
                row[f"{region}_rmse"]
                for row in fold_metrics
                if row[f"{region}_rmse"] is not None
            ]
            summary[f"mean_{region}_rmse"] = statistics.mean(values) if values else None
            if len(values) > 1:
                summary[f"std_{region}_rmse"] = statistics.stdev(values)
            else:
                summary[f"std_{region}_rmse"] = 0.0 if values else None
        summary_rows.append(summary)
        save_csv(
            results_dir / "fold_rmse.csv",
            ["setting", "fold", "overall_rmse", "constant_rmse", "decreasing_rmse"],
            fold_rows,
        )
        save_csv(
            results_dir / "summary_rmse.csv",
            ["setting", "mean_overall_rmse", "std_overall_rmse",
             "mean_constant_rmse", "std_constant_rmse",
             "mean_decreasing_rmse", "std_decreasing_rmse"],
            summary_rows,
        )
        save_plot(results_dir / "overall_rmse_by_weight.png", summary_rows, "overall")
        save_plot(results_dir / "decreasing_rmse_by_weight.png", summary_rows, "decreasing")

    print(f"Saved ablation results to {results_dir}")


if __name__ == "__main__":
    main()
