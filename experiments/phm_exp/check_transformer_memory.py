"""Find Transformer configurations that fit a real CUDA training step."""

import copy
import gc
import itertools
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.nn.parameter import UninitializedParameter
from torch.utils.data import DataLoader


SRC_PATH = Path(__file__).resolve().parents[2] / "src"
sys.path.append(str(SRC_PATH))

from exp_config import setup_exp  # noqa: E402
from models import load_ssm_model, wandb_data  # noqa: E402


EXPERIMENT_PATH = Path(__file__).resolve().parent
PROBE_CONFIG_PATH = EXPERIMENT_PATH / "config" / "transformer_memory_config.yaml"


def release_cuda() -> None:
    """Drop references and release unused cached CUDA allocations."""
    gc.collect()
    torch.cuda.empty_cache()


def count_initialized_parameters(model):
    """Count parameters used by the initialized model execution path."""
    uninitialized = [
        name
        for name, parameter in model.named_parameters()
        if isinstance(parameter, UninitializedParameter)
    ]
    count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if not isinstance(parameter, UninitializedParameter)
    )
    return count, uninitialized


def training_step(model, optimizer, criterion, batch, config, device) -> float:
    """Run the same forward/loss/backward/update operations as QuantileTrainer."""
    life, rul, mask = batch
    life = life.to(device) if "padding" in config.approach else life.to(device).squeeze(-1)
    rul = rul.to(device).squeeze(-1)
    mask = mask.to(device) if "padding" in config.approach else mask.to(device).squeeze(-1)

    tau = 0.5
    model.train()
    model.tau = tau
    optimizer.zero_grad(set_to_none=True)
    output = model(life)

    if "windowed" in config.approach and config.approach != "windowed_quantile_standard":
        loss = criterion(
            output,
            rul,
            mask,
            config.train_wins["constant"],
            config.train_wins["decreasing"],
            tau,
        )
    elif config.quantile_reg:
        loss = criterion(output, rul, mask, tau)
    else:
        loss = criterion(output, rul, mask)

    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return loss.item()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; the memory probe requires a CUDA GPU")

    with PROBE_CONFIG_PATH.open() as stream:
        probe_config = yaml.safe_load(stream)

    exp_config, base_model_config, device, _ = setup_exp()
    if exp_config.model_name != "Transformer":
        raise ValueError(f"Expected model_name=Transformer, got {exp_config.model_name}")

    # Load and transform PHM only once. The returned dataset can then be wrapped
    # in loaders with different batch sizes without repeating preprocessing.
    (
        initial_loader,
        _,
        _,
        initial_model,
        initial_optimizer,
        _,
        criterion,
        _,
        exp_config,
    ) = wandb_data(config=exp_config, model_config=base_model_config)
    train_dataset = initial_loader.dataset
    initial_model = initial_optimizer = initial_loader = None
    release_cuda()

    keys = ("d_model", "n_layers", "d_ff", "n_heads", "batch_size")
    value_lists = [probe_config[f"{key}_vals"] for key in keys]
    candidates = [dict(zip(keys, values)) for values in itertools.product(*value_lists)]
    candidates = [c for c in candidates if c["d_model"] % c["n_heads"] == 0]
    candidates.sort(key=lambda c: (c["d_model"], c["n_layers"], c["d_ff"], c["batch_size"], c["n_heads"]))

    results = []
    total = len(candidates)
    print(f"Testing {total} valid Transformer configurations on {torch.cuda.get_device_name()}")

    for index, candidate in enumerate(candidates, start=1):
        model = optimizer = scheduler = loader = iterator = batch = None
        result = dict(candidate)
        try:
            model_config = copy.copy(base_model_config)
            for key in ("d_model", "n_layers", "d_ff", "n_heads"):
                setattr(model_config, key, candidate[key])

            run_config = copy.copy(exp_config)
            run_config.batch_size = candidate["batch_size"]
            loader = DataLoader(train_dataset, batch_size=run_config.batch_size, shuffle=False)
            iterator = iter(loader)

            model, optimizer, scheduler = load_ssm_model(
                exp_config=run_config,
                model_config=model_config,
                model_name=run_config.model_name,
                output_size=run_config.sequence_length,
                mono_mask=run_config.mono_mask,
            )
            model = model.to(device)

            torch.cuda.reset_peak_memory_stats()
            losses = []
            for _ in range(probe_config["train_steps"]):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                losses.append(training_step(model, optimizer, criterion, batch, run_config, device))

            # The real forward initializes the projector and regression head.
            # TransformerExtractor also inherits unused LazyLinear layers from
            # Extractor; those cannot be initialized because its overridden
            # forward never calls them, so exclude them from the useful count.
            parameter_count, uninitialized_parameters = count_initialized_parameters(model)

            result.update(
                fits=True,
                parameter_count=parameter_count,
                uninitialized_parameters=uninitialized_parameters,
                peak_memory_mb=round(torch.cuda.max_memory_allocated() / (1024**2), 1),
                loss=losses[-1],
            )
            print(f"[{index}/{total}] FIT  {candidate} peak={result['peak_memory_mb']} MiB")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if not isinstance(error, torch.cuda.OutOfMemoryError) and "out of memory" not in str(error).lower():
                raise
            result.update(fits=False, error="CUDA out of memory")
            print(f"[{index}/{total}] OOM  {candidate}")
        finally:
            model = optimizer = scheduler = loader = iterator = batch = None
            release_cuda()

        results.append(result)

    feasible = [result for result in results if result["fits"]]
    output_path = EXPERIMENT_PATH / probe_config["output_path"]
    output_path.write_text(json.dumps({"results": results}, indent=2) + "\n")

    if not feasible:
        print("No tested configuration fits in CUDA memory.")
        print(f"Full results: {output_path}")
        raise SystemExit(1)

    # Prefer the largest parameterized model; for equal-sized models, prefer
    # the larger batch and then fewer heads (smaller attention workspace).
    best = max(
        feasible,
        key=lambda result: (
            result["parameter_count"],
            result["batch_size"],
            -result["n_heads"],
        ),
    )
    print("\nLargest feasible configuration:")
    print(json.dumps(best, indent=2))
    print(f"Full results: {output_path}")


if __name__ == "__main__":
    main()
