"""Create a PHM benchmark blob plot from a THOP profiling JSON artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px

EXPERIMENT_PATH = Path(__file__).resolve().parent
SRC_PATH = EXPERIMENT_PATH.parent.parent / "src"
sys.path.insert(0, str(SRC_PATH))

from exp_config import ExperimentConfig
from utils import get_current_time, load_yaml_to_dict, open_element


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PHM model complexity versus quantile loss."
    )
    parser.add_argument(
        "--profile-json",
        type=Path,
        default=EXPERIMENT_PATH / "profile_outputs" / "benchmark_model_profile.json",
    )
    parser.add_argument(
        "--model-experiment",
        action="append",
        required=True,
        metavar="MODEL=EXPERIMENT_NAME",
        help="Repeat once for every profiled model.",
    )
    parser.add_argument(
        "--exp-config",
        type=Path,
        default=EXPERIMENT_PATH / "config" / "ssm_exp_config.yaml",
    )
    parser.add_argument(
        "--failure-type",
        help="Defaults to the failure type stored in the profile JSON.",
    )
    parser.add_argument(
        "--approach",
        help="Defaults to the PHM experiment config approach.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        help="Defaults to the PHM experiment config quantile_run.",
    )
    parser.add_argument(
        "--eval-loss",
        help="Defaults to the PHM experiment config life_eval_loss.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to plots/blob_plot/<failure_type>/<approach>.",
    )
    return parser.parse_args()


def parse_model_experiments(
    values: list[str], expected_models: set[str]
) -> dict[str, str]:
    mappings: dict[str, str] = {}
    for value in values:
        if value.count("=") != 1:
            raise ValueError(f"Invalid mapping '{value}'; use MODEL=EXPERIMENT_NAME")
        model_name, experiment_name = (item.strip() for item in value.split("="))
        if not model_name or not experiment_name:
            raise ValueError(f"Invalid mapping '{value}'; both sides are required")
        if model_name in mappings:
            raise ValueError(f"Duplicate mapping for model '{model_name}'")
        mappings[model_name] = experiment_name
    if set(mappings) != expected_models:
        missing = sorted(expected_models - set(mappings))
        unknown = sorted(set(mappings) - expected_models)
        messages = []
        if missing:
            messages.append(f"missing mappings for {missing}")
        if unknown:
            messages.append(f"unknown profiled models {unknown}")
        raise ValueError("; ".join(messages))
    return mappings


def find_metrics_file(metrics_dir: Path) -> Path:
    files = sorted(metrics_dir.glob("*_quantile_reg_global_metrics_df.pickle"))
    if len(files) != 1:
        raise FileNotFoundError(
            "Expected exactly one aggregate quantile metrics file in "
            f"{metrics_dir}, found {len(files)}"
        )
    return files[0]


def load_life_mean_loss(metrics_file: Path, quantile: float) -> float:
    metrics_df = open_element(str(metrics_file), filetype="pickle")
    column = f"quantile_{quantile}"
    if "Life_mean" not in metrics_df.index:
        raise KeyError(f"'Life_mean' is missing from {metrics_file}")
    if column not in metrics_df.columns:
        raise KeyError(f"'{column}' is missing from {metrics_file}")
    return float(metrics_df.loc["Life_mean", column])


def main() -> None:
    args = parse_args()
    with args.profile_json.open(encoding="utf-8") as profile_file:
        profile_data = json.load(profile_file)
    if profile_data.get("schema_version") != 1 or not isinstance(
        profile_data.get("models"), dict
    ):
        raise ValueError("Unsupported profile JSON schema")

    exp_config = ExperimentConfig.from_dict(load_yaml_to_dict(str(args.exp_config)))
    failure_type = args.failure_type or profile_data.get("profiling", {}).get(
        "failure_type"
    )
    if not failure_type:
        raise ValueError("Failure type is absent; pass --failure-type explicitly")
    approach = args.approach or exp_config.approach
    quantile = args.quantile if args.quantile is not None else exp_config.quantile_run
    if not 0 <= quantile <= 1:
        raise ValueError("--quantile must be between 0 and 1")
    eval_loss = args.eval_loss or exp_config.life_eval_loss
    mappings = parse_model_experiments(
        args.model_experiment, set(profile_data["models"])
    )

    rows = []
    for model_name, profile in profile_data["models"].items():
        metrics_dir = (
            EXPERIMENT_PATH
            / "metrics"
            / model_name
            / failure_type
            / approach
            / mappings[model_name]
        )
        metric = load_life_mean_loss(find_metrics_file(metrics_dir), quantile)
        rows.append(
            {
                "Model": model_name,
                "Parameters": profile["parameters"],
                "Life_mean evaluation loss": metric,
                "GFLOPs": profile["gflops"],
                "MACs": profile["macs"],
            }
        )
    dataframe = pd.DataFrame(rows)

    figure = px.scatter(
        dataframe,
        x="Parameters",
        y="Life_mean evaluation loss",
        size="GFLOPs",
        hover_name="Model",
        hover_data={"MACs": ":,", "GFLOPs": ".4f", "Parameters": ":,"},
        text="Model",
        log_x=True,
        size_max=60,
        labels={
            "Parameters": "Number of parameters",
            "Life_mean evaluation loss": (
                f"Life_mean {eval_loss} (quantile {quantile})"
            ),
            "GFLOPs": "GFLOPs per input window",
        },
        title=f"PHM benchmark: complexity vs. predictive loss ({failure_type})",
    )
    figure.update_traces(
        marker={"color": "#2563eb", "opacity": 0.8},
        textposition="bottom center",
    )
    figure.update_layout(template="plotly_white")

    output_dir = args.output_dir or (
        EXPERIMENT_PATH / "plots" / "blob_plot" / failure_type / approach
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{get_current_time()}_phm_blob_plot_quantile_{quantile}_{eval_loss}.png"
    )
    output_path = output_dir / filename
    figure.write_image(output_path, scale=3)
    print(dataframe.to_markdown(index=False))
    print(f"Blob plot saved to: {output_path}")


if __name__ == "__main__":
    main()
