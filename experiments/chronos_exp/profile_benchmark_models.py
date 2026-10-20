"""Print a Markdown parameter-count and model-size table for the CMAPSS benchmark.

Uses the current model implementations and the legacy CMAPSS YAML settings;
no dataset, training, checkpoint, or W&B connection is required.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import fields
from pathlib import Path
import sys


EXPERIMENT_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_PATH.parents[1] / "src"))

MODEL_ALIASES = {"RULTransformer": "Transformer", "RULInformer": "Informer"}
SUPPORTED_MODELS = {"S4", "S5", "S4D", "LSTM", "GRU", "RNN", "Linear", "MLP",
                    "Transformer", "Informer", *MODEL_ALIASES}


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-config", type=Path,
                        default=EXPERIMENT_PATH / "config/ssm_exp_config.yaml")
    parser.add_argument("--model-config", type=Path,
                        help="Override model_config_path in the experiment YAML.")
    parser.add_argument("--models", nargs="+", choices=sorted(SUPPORTED_MODELS),
                        help="Defaults to model_names in the experiment YAML.")
    parser.add_argument("--sequence-length", type=positive_int)
    parser.add_argument("--input-features", type=positive_int,
                        help="Override the 14 selected CMAPSS sensor features (before tau).")
    parser.add_argument("--tau-feat", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tau-mult", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--legacy-s5", action="store_true",
                        help="Use the pre-refactor S5 architecture from commit 85969ec.")
    parser.add_argument("--output", type=Path, help="Also save the Markdown report here.")
    return parser.parse_args()


def load_configs(args):
    import yaml
    from exp_config import ExperimentConfig, ModelConfig

    raw = yaml.safe_load(args.exp_config.read_text(encoding="utf-8"))
    if raw.get("data_name") != "CMAPSS":
        raise ValueError("The experiment config must use data_name: CMAPSS")
    if raw.get("approach") != "windowed":
        raise ValueError("This profiler requires the benchmark's windowed approach")
    # The older CMAPSS YAML also contains dataset and plotting fields which
    # predate the current ExperimentConfig dataclass. Preserve those attributes.
    valid_keys = {field.name for field in fields(ExperimentConfig)}
    config = ExperimentConfig.from_dict({k: v for k, v in raw.items() if k in valid_keys})
    config.add_params({k: v for k, v in raw.items() if k not in valid_keys})
    if args.sequence_length is not None:
        config.sequence_length = args.sequence_length
    if config.sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    model_path = args.model_config
    if model_path is None:
        model_path = Path(config.model_config_path)
        if not model_path.is_absolute():
            model_path = args.exp_config.resolve().parent.parent / model_path
    model_config = ModelConfig.from_dict(yaml.safe_load(model_path.read_text(encoding="utf-8")))
    model_config.device = "cpu"
    model_config.quantile_reg = config.quantile_reg
    for key in ("tau_feat", "tau_mult"):
        if getattr(args, key) is not None:
            setattr(model_config, key, getattr(args, key))

    if args.input_features is None:
        if config.transformer_type not in (0, 1, 2, 3, 4):
            raise ValueError("Use --input-features for custom preprocessing")
        from ceruleo.dataset.catalog.CMAPSS import sensor_indices
        feature_count = len(sensor_indices)
    else:
        feature_count = args.input_features
    names = args.models or raw.get("model_names", [raw["model_name"]])
    constructors = [MODEL_ALIASES.get(name, name) for name in names]
    if not names or any(name not in SUPPORTED_MODELS for name in names):
        raise ValueError(f"Unsupported model_names: {names}")
    if len(set(constructors)) != len(constructors):
        raise ValueError("Duplicate models (including aliases) are not allowed")
    return config, model_config, names, feature_count, model_path


def build_legacy_s5(config, model_config, feature_count):
    """Reproduce S5Model from src/ssm_models.py at commit 85969ec.

    Unlike the current extractor, blocks operate on sensors (+ optional tau),
    use d_model as state_dim, and precede the input projection. No extra norms.
    """
    import torch
    from torch import nn
    from s5 import S5Block

    if config.quantile_scale or config.monotonic:
        raise ValueError("Legacy S5 does not support quantile_scale or monotonic models")

    class LegacyS5Model(nn.Module):
        def __init__(self):
            super().__init__()
            d_input = feature_count + int(config.quantile_reg and model_config.tau_feat)
            self.encoder = nn.Linear(d_input, model_config.d_model)
            self.s5_layers = nn.ModuleList([
                S5Block(dim=d_input, state_dim=model_config.d_model,
                        bidir=model_config.bidir)
                for _ in range(model_config.n_layers)
            ])
            d_output = config.sequence_length if model_config.gap else 1
            self.decoder = nn.Linear(model_config.d_model, d_output)

        def forward(self, x):
            tau = config.quantile_run
            if config.quantile_reg and model_config.tau_feat:
                x = torch.cat([x, x.new_full((*x.shape[:2], 1), tau)], dim=-1)
            for layer in self.s5_layers:
                x = layer(x)
            x = self.encoder(x)
            if model_config.gap:
                x = x.mean(dim=1)
            x = self.decoder(x)
            if model_config.tau_mult:
                x = x * tau
            return x if model_config.gap else x.squeeze(-1)

    return LegacyS5Model()


def torchinfo_parameters(model, input_shape):
    """Use the legacy torchinfo branch, preserving its manual-count fallback."""
    try:
        from torchinfo import summary
        result = summary(model=model, input_size=input_shape, device="cpu",
                         verbose=0, cache_forward_pass=False)
        return result.total_params, None
    except Exception as exc:
        # The old load_ssm_model caught profiler failures and called
        # model_summary_manual. Mark this explicitly instead of presenting the
        # fallback value as a successful torchinfo measurement.
        reason = f"{type(exc).__name__}: {exc}"
        cause = exc.__cause__
        while cause is not None:
            reason += f"; caused by {type(cause).__name__}: {cause}"
            cause = cause.__cause__
        print(f"torchinfo failed; using manual parameter count: {reason}", file=sys.stderr)
        return sum(p.numel() for p in model.parameters()), reason


def profile_model(name, config, model_config, feature_count, legacy_s5=False):
    import copy
    import torch
    from io import BytesIO
    from models import load_ssm_model

    torch.manual_seed(0)
    constructor = MODEL_ALIASES.get(name, name)
    config, model_config = copy.deepcopy(config), copy.deepcopy(model_config)
    config.model_name = constructor
    if legacy_s5 and constructor == "S5":
        model = build_legacy_s5(config, model_config, feature_count)
    else:
        model, _, _ = load_ssm_model(
            exp_config=config, model_config=model_config, model_name=constructor,
            output_size=config.sequence_length, tau=config.quantile_run,
        )
    model.eval()
    dummy = torch.zeros(1, config.sequence_length, feature_count)
    with torch.no_grad():
        # Materialize LazyLinear parameters and persistent buffers before
        # counting and serializing the complete prediction model.
        result = model(dummy)
        if result.shape != (1, config.sequence_length):
            raise ValueError(f"Unexpected {name} output shape: {tuple(result.shape)}")
        if not torch.isfinite(result).all():
            raise ValueError(f"Non-finite {name} output")
        parameters = sum(p.numel() for p in model.parameters())
        # Serialize exactly the state_dict a weights-only checkpoint contains.
        # Native tensor dtypes (including complex SSM weights) are preserved.
        with BytesIO() as buffer:
            torch.save(model.state_dict(), buffer)
            size_bytes = buffer.tell()
        info_parameters, info_error = torchinfo_parameters(model, tuple(dummy.shape))
    return name, parameters, info_parameters, size_bytes, info_error


def render_markdown(rows, config, model_config, feature_count, exp_path, model_path,
                    legacy_s5=False):
    from importlib.metadata import version

    lines = [
        "| Model | Parameters (manual) | Parameters (torchinfo) | Model size (MiB) |",
        "| --- | ---: | ---: | ---: |",
        *[f"| {name} | {parameters:,} | {info_parameters:,}"
          f"{' (manual fallback)' if info_error else ''} | {size_bytes / 1024**2:.6f} |"
          for name, parameters, info_parameters, size_bytes, info_error in rows],
        "",
        f"Dataset: CMAPSS {config.cmapss_models}. Input: `(1, {config.sequence_length}, "
        f"{feature_count})` (batch, time, sensors), before any tau feature.",
        f"Quantile regression: `{config.quantile_reg}`; tau: `{config.quantile_run}`; "
        f"tau_feat: `{model_config.tau_feat}`; tau_mult: `{model_config.tau_mult}`.",
        f"Architecture: d_model={model_config.d_model}, n_layers={model_config.n_layers}, "
        f"d_state={model_config.d_state}, d_ff={model_config.d_ff}, n_heads={model_config.n_heads}.",
        f"Experiment config: `{exp_path}`. Model config: `{model_path}`.",
        f"S5 implementation: {'legacy S5Model (commit 85969ec)' if legacy_s5 else 'current S5Extractor'}. "
        + (f"Legacy blocks use sensor width (+ optional tau), state_dim={model_config.d_model}; "
           "d_state is unused for legacy S5." if legacy_s5 else ""),
        f"Versions: torch `{version('torch')}`; torchinfo `{version('torchinfo')}`.",
        "",
        "Parameters count all registered parameter elements after lazy initialization "
        "(a complex element counts as one). Counts describe the current code and these "
        "configs; historical checkpoints are not loaded.",
        "The torchinfo column uses `summary(model, input_size=(1, sequence_length, "
        "sensor_features), device='cpu').total_params`, matching the old "
        "`model_summary` / `summary_func: torchinfo` branch. Tau is appended inside "
        "the model. On profiler failure, the old manual-count fallback is used "
        "and labeled explicitly. Both parameter columns use raw counts, without K/M scaling.",
        "Model size is the byte length of `torch.save(model.state_dict(), buffer)` "
        "after initialization, divided by 1,048,576 (MiB). It includes parameters, "
        "persistent buffers, and serialization overhead, preserving native dtypes "
        "(normally float32 and complex64). It excludes optimizer state, gradients, "
        "and activations; it does not measure runtime memory. Serialization overhead "
        "can vary with the PyTorch version.",
    ]
    for name, _, _, _, error in rows:
        if error:
            lines.extend(["", f"torchinfo fallback for {name}: " + " ".join(error.split())])
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    # Keep stdout valid Markdown, even when model imports/constructors print.
    with redirect_stdout(sys.stderr):
        config, model_config, names, feature_count, model_path = load_configs(args)
        import torch
        torch.set_num_threads(1)
        rows = []
        for name in names:
            print(f"Profiling {name}...", file=sys.stderr)
            rows.append(profile_model(name, config, model_config, feature_count,
                                      legacy_s5=args.legacy_s5))
    markdown = render_markdown(rows, config, model_config, feature_count,
                               args.exp_config, model_path, legacy_s5=args.legacy_s5)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    print(markdown, end="")


if __name__ == "__main__":
    main()
