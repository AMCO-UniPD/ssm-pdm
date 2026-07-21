"""Profile the PHM benchmark RUL models with THOP.

The resulting JSON file is the input artifact for ``plot_benchmark_blob.py``.
Complexity is measured for one input window (batch size one), so results are
comparable across model architectures.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

EXPERIMENT_PATH = Path(__file__).resolve().parent
SRC_PATH = EXPERIMENT_PATH.parent.parent / "src"
sys.path.insert(0, str(SRC_PATH))

from exp_config import ExperimentConfig, ModelConfig
from models import load_ssm_model
from utils import get_phm_feature_names, load_yaml_to_dict


MODEL_ALIASES = {
    "RULTransformer": "Transformer",
    "RULInformer": "Informer",
}
DEFAULT_MODELS = ["S4", "S5", "S4D", "Linear", "MLP", "LSTM", "RNN", "GRU"]
# DEFAULT_MODELS = ["S4", "S5", "S4D", "Linear", "MLP", "LSTM", "RNN", "GRU", "RULTransformer"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile PHM benchmark model complexity."
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--exp-config",
        type=Path,
        default=EXPERIMENT_PATH / "config" / "ssm_exp_config.yaml",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=EXPERIMENT_PATH / "config" / "ssm_config.yaml",
    )
    parser.add_argument("--failure-type", default="flow_low")
    parser.add_argument("--test-phm-tools", nargs="+", default=["01M02", "02M02", "03M01", "04M01", "06M01"])
    parser.add_argument(
        "--input-features",
        type=int,
        help="Override PHM feature discovery; useful when the PHM dataset is unavailable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_PATH / "profile_outputs" / "benchmark_model_profile.json",
    )
    return parser.parse_args()


def load_configs(args: argparse.Namespace) -> tuple[ExperimentConfig, ModelConfig]:
    exp_config = ExperimentConfig.from_dict(load_yaml_to_dict(str(args.exp_config)))
    model_config = ModelConfig.from_dict(load_yaml_to_dict(str(args.model_config)))
    # ``data_name`` normally comes from the shared experiment CLI and is added
    # dynamically by ``setup_exp``.  This standalone profiler has its own CLI,
    # so set the dataset explicitly before calling the shared PHM utilities.
    exp_config.data_name = "PHM"
    exp_config.failure_type = args.failure_type
    exp_config.test_phm_tools = args.test_phm_tools
    model_config.device = "cpu"
    model_config.quantile_reg = exp_config.quantile_reg
    return exp_config, model_config


def resolve_feature_count(
    config: ExperimentConfig, override: int | None
) -> tuple[int, str]:
    if override is not None:
        if override <= 0:
            raise ValueError("--input-features must be a positive integer")
        return override, "--input-features"
    return len(get_phm_feature_names(config)), "get_phm_feature_names"


def profile_model(
    display_name: str,
    config: ExperimentConfig,
    model_config: ModelConfig,
    input_shape: tuple[int, int, int],
) -> dict[str, Any]:
    try:
        from thop import profile
    except ImportError as exc:
        raise RuntimeError(
            "THOP is required. Install it in the environment used for profiling."
        ) from exc

    constructor_name = MODEL_ALIASES.get(display_name, display_name)
    config = copy.deepcopy(config)
    model_config = copy.deepcopy(model_config)
    config.model_name = constructor_name
    torch.manual_seed(0)
    model, _, _ = load_ssm_model(
        exp_config=config,
        model_config=model_config,
        model_name=constructor_name,
        output_size=config.sequence_length,
        tau=config.quantile_run,
    )
    model.eval()
    dummy_input = torch.zeros(input_shape, dtype=torch.float32)

    # Projector uses LazyLinear, so initialize it before THOP inspects parameters.
    with torch.no_grad():
        model(dummy_input)
        macs, thop_parameters = profile(model, inputs=(dummy_input,), verbose=False)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    flops = 2 * macs
    return {
        "display_name": display_name,
        "constructor_name": constructor_name,
        "parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "thop_parameters": int(thop_parameters),
        "macs": int(macs),
        "flops": int(flops),
        "gflops": flops / 1_000_000_000,
    }


def profile_to_dataframe(profile_data: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Model": model_name,
                "Parameters": values["parameters"],
                "Trainable parameters": values["trainable_parameters"],
                "MACs": values["macs"],
                "FLOPs": values["flops"],
                "GFLOPs": values["gflops"],
            }
            for model_name, values in profile_data["models"].items()
        ]
    )


def main() -> None:
    args = parse_args()
    exp_config, model_config = load_configs(args)
    feature_count, feature_source = resolve_feature_count(exp_config, args.input_features)
    input_shape = (1, exp_config.sequence_length, feature_count)

    profile_data: dict[str, Any] = {
        "schema_version": 1,
        "profiling": {
            "profiler": "thop.profile",
            "batch_size": 1,
            "input_shape": list(input_shape),
            "failure_type": exp_config.failure_type,
            "test_phm_tools": exp_config.test_phm_tools,
            "evaluation_quantile": exp_config.quantile_run,
            "feature_count_source": feature_source,
            "random_seed": 0,
            "custom_thop_operations": False,
            "mac_to_flop_conversion": "1 MAC = 2 FLOPs",
        },
        "models": {},
    }
    for display_name in args.models:
        if display_name in profile_data["models"]:
            raise ValueError(f"Duplicate model name: {display_name}")
        print(f"Profiling {display_name}...")
        profile_data["models"][display_name] = profile_model(
            display_name, exp_config, model_config, input_shape
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(profile_data, output_file, indent=2)
        output_file.write("\n")

    dataframe = profile_to_dataframe(profile_data)
    markdown_path = args.output.with_suffix(".md")
    markdown = dataframe.to_markdown(index=False)
    markdown_path.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    print(f"Profile JSON saved to: {args.output}")
    print(f"Profile table saved to: {markdown_path}")


if __name__ == "__main__":
    main()
