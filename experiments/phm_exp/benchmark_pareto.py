"""Shared data loading and plotting for PHM benchmark Pareto fronts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
import plotly.graph_objects as go

EXPERIMENT_PATH = Path(__file__).resolve().parent
SRC_PATH = EXPERIMENT_PATH.parent.parent / "src"
sys.path.insert(0, str(SRC_PATH))

from exp_config import ExperimentConfig  # noqa: E402
from utils import get_current_time, load_yaml_to_dict, open_element  # noqa: E402

ComplexityMetric = Literal["parameters", "gflops"]

METRIC_CONFIG = {
    "parameters": {
        "column": "Parameters",
        "axis_title": "Number of parameters",
        "directory": "parameters",
        "filename": "parameters",
    },
    "gflops": {
        "column": "GFLOPs",
        "axis_title": "GFLOPs per input window",
        "directory": "gflops",
        "filename": "gflops",
    },
}


def parse_args(metric: ComplexityMetric) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the PHM benchmark RMSE Pareto front against "
            f"{METRIC_CONFIG[metric]['axis_title'].lower()}."
        )
    )
    parser.add_argument(
        "--profile-json",
        type=Path,
        default=EXPERIMENT_PATH / "profile_outputs" / "benchmark_model_profile.json",
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        required=True,
        help="Names of the profiled models to include in the plot.",
    )
    parser.add_argument(
        "--exp_names",
        nargs="+",
        required=True,
        help="Experiment names corresponding positionally to --model_names.",
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
        help="Defaults to life_eval_loss and must resolve to RMSE.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Defaults to pareto_front/"
            f"{METRIC_CONFIG[metric]['directory']}/<failure_type>/<approach>."
        ),
    )
    return parser.parse_args()


def map_model_experiments(
    model_names: list[str], exp_names: list[str], expected_models: set[str]
) -> dict[str, str]:
    if len(model_names) != len(exp_names):
        raise ValueError(
            "--model_names and --exp_names must contain the same number of values"
        )
    if len(set(model_names)) != len(model_names):
        raise ValueError("--model_names must not contain duplicates")
    if any(not exp_name.strip() for exp_name in exp_names):
        raise ValueError("--exp_names must not contain empty values")

    mappings = dict(zip(model_names, exp_names))
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


def load_life_mean_rmse(metrics_file: Path, quantile: float) -> float:
    metrics_df = open_element(str(metrics_file), filetype="pickle")
    column = f"quantile_{quantile}"
    if "Life_mean" not in metrics_df.index:
        raise KeyError(f"'Life_mean' is missing from {metrics_file}")
    if column not in metrics_df.columns:
        raise KeyError(f"'{column}' is missing from {metrics_file}")
    rmse = float(metrics_df.loc["Life_mean", column])
    if not math.isfinite(rmse):
        raise ValueError(f"Life_mean RMSE in {metrics_file} must be finite")
    return rmse


def classify_pareto(
    dataframe: pd.DataFrame, complexity_column: str, error_column: str = "RMSE"
) -> pd.Series:
    """Return True for points not strictly dominated in both minimized objectives."""
    complexities = dataframe[complexity_column].to_numpy(dtype=float)
    errors = dataframe[error_column].to_numpy(dtype=float)
    is_pareto = []
    for index, (complexity, error) in enumerate(zip(complexities, errors)):
        dominated = any(
            other_complexity <= complexity
            and other_error <= error
            and (other_complexity < complexity or other_error < error)
            for other_index, (other_complexity, other_error) in enumerate(
                zip(complexities, errors)
            )
            if other_index != index
        )
        is_pareto.append(not dominated)
    return pd.Series(is_pareto, index=dataframe.index, dtype=bool)


def load_profile(profile_json: Path) -> dict:
    with profile_json.open(encoding="utf-8") as profile_file:
        profile_data = json.load(profile_file)
    if profile_data.get("schema_version") != 1 or not isinstance(
        profile_data.get("models"), dict
    ):
        raise ValueError("Unsupported profile JSON schema")
    if not profile_data["models"]:
        raise ValueError("Profile JSON contains no models")
    return profile_data


def build_benchmark_dataframe(
    profile_data: dict,
    mappings: dict[str, str],
    failure_type: str,
    approach: str,
    quantile: float,
    metrics_root: Path = EXPERIMENT_PATH / "metrics",
) -> pd.DataFrame:
    rows = []
    for model_name, profile in profile_data["models"].items():
        try:
            parameters = int(profile["parameters"])
            gflops = float(profile["gflops"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid parameters or GFLOPs for model {model_name!r}"
            ) from exc
        if parameters <= 0 or not math.isfinite(gflops) or gflops <= 0:
            raise ValueError(
                f"Parameters and GFLOPs for model {model_name!r} must be positive"
            )

        metrics_dir = (
            metrics_root / model_name / failure_type / approach / mappings[model_name]
        )
        rows.append(
            {
                "Model": model_name,
                "Parameters": parameters,
                "GFLOPs": gflops,
                "RMSE": load_life_mean_rmse(find_metrics_file(metrics_dir), quantile),
            }
        )
    return pd.DataFrame(rows)


def build_pareto_figure(
    dataframe: pd.DataFrame,
    metric: ComplexityMetric,
    failure_type: str,
    approach: str,
    quantile: float,
) -> go.Figure:
    config = METRIC_CONFIG[metric]
    complexity_column = config["column"]
    plotted = dataframe.copy()
    plotted["Pareto optimal"] = classify_pareto(plotted, complexity_column)

    figure = go.Figure()
    dominated = plotted.loc[~plotted["Pareto optimal"]]
    pareto = plotted.loc[plotted["Pareto optimal"]].sort_values(
        [complexity_column, "RMSE", "Model"]
    )
    hover_template = (
        "<b>%{customdata[0]}</b><br>"
        "Parameters: %{customdata[1]:,}<br>"
        "GFLOPs: %{customdata[2]:.5g}<br>"
        "RMSE: %{y:.5g}<extra>%{fullData.name}</extra>"
    )

    if metric == "gflops":
        # Separate model traces give the external legend a key for every point.
        # Shapes distinguish models even when their GFLOPs are identical.
        symbols = [
            "circle",
            "square",
            "diamond",
            "cross",
            "x",
            "triangle-up",
            "triangle-down",
            "triangle-left",
            "triangle-right",
            "pentagon",
            "hexagon",
            "star",
            "hourglass",
            "bowtie",
        ]
        figure.add_trace(
            go.Scatter(
                x=pareto[complexity_column],
                y=pareto["RMSE"],
                mode="lines",
                name="Pareto optimal",
                line={"color": "#16a34a", "width": 2},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        for index, (_, row) in enumerate(plotted.iterrows()):
            figure.add_trace(
                go.Scatter(
                    x=[row[complexity_column]],
                    y=[row["RMSE"]],
                    mode="markers",
                    name=row["Model"],
                    marker={
                        "color": "#16a34a" if row["Pareto optimal"] else "#dc2626",
                        "size": 12,
                        "symbol": symbols[index % len(symbols)],
                    },
                    customdata=[[row["Model"], row["Parameters"], row["GFLOPs"]]],
                    hovertemplate=hover_template,
                )
            )
    elif not dominated.empty:
        figure.add_trace(
            go.Scatter(
                x=dominated[complexity_column],
                y=dominated["RMSE"],
                mode="markers+text",
                name="Dominated",
                marker={"color": "#dc2626", "size": 11},
                text=dominated["Model"],
                textposition="bottom center",
                cliponaxis=False,
                customdata=dominated[["Model", "Parameters", "GFLOPs"]],
                hovertemplate=hover_template,
            )
        )
    if metric == "parameters":
        figure.add_trace(
            go.Scatter(
                x=pareto[complexity_column],
                y=pareto["RMSE"],
                mode="lines+markers+text",
                name="Pareto optimal",
                line={"color": "#16a34a", "width": 2},
                marker={"color": "#16a34a", "size": 12},
                text=pareto["Model"],
                textposition="bottom center",
                cliponaxis=False,
                customdata=pareto[["Model", "Parameters", "GFLOPs"]],
                hovertemplate=hover_template,
            )
        )
    figure.update_layout(
        template="plotly_white",
        # Equal canvas sizes keep text the same size when LaTeX scales both
        # figures to the same printed width.
        width=1000,
        height=600,
        font={"size": 16},
        title={
            "text": f"PHM benchmark Pareto front: RMSE vs. {config['axis_title']}",
            "font": {"size": 22},
        },
        xaxis={"title": config["axis_title"], "type": "log"},
        yaxis={"title": f"Life_mean RMSE (quantile {quantile})"},
        legend={"title": "Pareto status"},
    )
    if metric == "gflops":
        figure.update_layout(
            xaxis={"dtick": "D2", "tickformat": ".3~g"},
            legend={
                "title": "Model<br>Green: Pareto optimal<br>Red: dominated",
                "x": 1.02,
                "xanchor": "left",
                "y": 1,
                "yanchor": "top",
            },
        )
    return figure


def run(metric: ComplexityMetric) -> None:
    args = parse_args(metric)
    profile_data = load_profile(args.profile_json)
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
    if eval_loss.casefold() != "rmse":
        raise ValueError("Pareto accuracy must use RMSE; set --eval-loss rmse")

    mappings = map_model_experiments(
        args.model_names, args.exp_names, set(profile_data["models"])
    )
    dataframe = build_benchmark_dataframe(
        profile_data, mappings, failure_type, approach, quantile
    )
    complexity_column = METRIC_CONFIG[metric]["column"]
    dataframe["Pareto optimal"] = classify_pareto(dataframe, complexity_column)
    figure = build_pareto_figure(dataframe, metric, failure_type, approach, quantile)

    output_dir = args.output_dir or (
        EXPERIMENT_PATH
        / "pareto_front"
        / METRIC_CONFIG[metric]["directory"]
        / failure_type
        / approach
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{get_current_time()}_phm_pareto_"
        f"{METRIC_CONFIG[metric]['filename']}_quantile_{quantile}_rmse.png"
    )
    output_path = output_dir / filename
    figure.write_image(output_path, scale=3)
    print(dataframe.to_markdown(index=False))
    print(f"Pareto plot saved to: {output_path}")
