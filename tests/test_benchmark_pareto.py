# ruff: noqa: E402

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PHM_EXPERIMENT_PATH = Path(__file__).resolve().parents[1] / "experiments" / "phm_exp"
sys.path.insert(0, str(PHM_EXPERIMENT_PATH))

from benchmark_pareto import (
    build_benchmark_dataframe,
    build_pareto_figure,
    classify_pareto,
    load_profile,
    map_model_experiments,
)


def test_classify_pareto_handles_dominance_ties_and_duplicates():
    dataframe = pd.DataFrame(
        {
            "Parameters": [1, 2, 2, 3, 1, 1],
            "RMSE": [5, 4, 6, 3, 5, 7],
        }
    )

    result = classify_pareto(dataframe, "Parameters")

    assert result.tolist() == [True, True, False, True, True, False]


def test_classify_pareto_marks_single_and_monotonic_tradeoff_optimal():
    single = pd.DataFrame({"GFLOPs": [1.0], "RMSE": [2.0]})
    tradeoff = pd.DataFrame({"GFLOPs": [0.1, 0.2, 0.3], "RMSE": [3.0, 2.0, 1.0]})

    assert classify_pareto(single, "GFLOPs").tolist() == [True]
    assert classify_pareto(tradeoff, "GFLOPs").tolist() == [True, True, True]


def test_mapping_requires_exact_profile_models():
    with pytest.raises(ValueError, match="missing mappings"):
        map_model_experiments(["S4"], ["experiment"], {"S4", "S5"})


def test_load_profile_rejects_unsupported_schema(tmp_path):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({"schema_version": 2, "models": {}}))

    with pytest.raises(ValueError, match="Unsupported profile JSON schema"):
        load_profile(profile_path)


def test_build_dataframe_reads_profile_and_rmse(tmp_path):
    metrics_root = tmp_path / "metrics"
    metrics_dir = metrics_root / "S4" / "flow_low" / "padding" / "experiment"
    metrics_dir.mkdir(parents=True)
    pd.DataFrame({"quantile_0.5": [1.25]}, index=["Life_mean"]).to_pickle(
        metrics_dir / "S4_quantile_reg_global_metrics_df.pickle"
    )
    profile = {"models": {"S4": {"parameters": 100, "gflops": 0.25}}}

    dataframe = build_benchmark_dataframe(
        profile,
        {"S4": "experiment"},
        "flow_low",
        "padding",
        0.5,
        metrics_root,
    )

    assert dataframe.to_dict("records") == [
        {"Model": "S4", "Parameters": 100, "GFLOPs": 0.25, "RMSE": 1.25}
    ]


@pytest.mark.parametrize(
    ("metric", "x_title"),
    [("parameters", "Number of parameters"), ("gflops", "GFLOPs per input window")],
)
def test_figure_colors_and_connects_the_pareto_front(metric, x_title):
    dataframe = pd.DataFrame(
        {
            "Model": ["A", "B", "C"],
            "Parameters": [100, 200, 300],
            "GFLOPs": [0.1, 0.3, 0.2],
            "RMSE": [3.0, 4.0, 2.0],
        }
    )

    figure = build_pareto_figure(dataframe, metric, "flow_low", "padding", 0.5)

    assert figure.layout.xaxis.type == "log"
    assert figure.layout.xaxis.title.text == x_title
    assert figure.data[-1].name == "Pareto optimal"
    assert figure.data[-1].mode == "lines+markers+text"
    assert figure.data[-1].marker.color == "#16a34a"
    assert figure.data[0].name == "Dominated"
    assert figure.data[0].marker.color == "#dc2626"
