"""Read legacy CMAPSS predictions without PHM's training configuration loader."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "chronos_exp/config/quantile_metrics.yaml"
)


def latest_output(folder, file_pos=0):
    files = sorted(
        Path(folder).glob("*.pickle"),
        key=lambda p: (p.stat().st_mtime, p.name),
        reverse=True,
    )
    if file_pos < 0 or file_pos >= len(files):
        raise FileNotFoundError(f"No output pickle at file_pos={file_pos} in {folder}")
    return files[file_pos]


def output_life_metrics(outputs, first_life_id=1):
    """Per-life RMSE across nonzero targets; padding matches perf.lifes_metrics."""
    predictions, targets = outputs["y_pred"], outputs["y_true"]
    if not len(predictions) or len(predictions) != len(targets):
        raise ValueError(
            "Predictions and targets must contain the same nonzero number of lives"
        )
    errors = []
    for prediction, target in zip(predictions, targets):
        prediction = np.asarray(prediction, dtype=float).squeeze()
        target = np.asarray(target, dtype=float).squeeze()
        if prediction.shape != target.shape:
            raise ValueError("Prediction and target shapes differ")
        valid = target != 0
        if (
            not valid.any()
            or not np.isfinite(target).all()
            or not np.isfinite(prediction[valid]).all()
        ):
            raise ValueError(
                "Each life needs finite predictions and nonzero finite targets"
            )
        errors.append(np.sqrt(np.mean((prediction[valid] - target[valid]) ** 2)))
    return pd.Series(
        errors, index=[f"Life_{first_life_id + i}" for i in range(len(errors))]
    )


def load_output_metrics(
    experiment_path, run_ids, quantiles, first_life_id=1, file_pos=0
):
    """Return per-run tables and provenance, checking that life order is stable."""
    frames, sources = [], []
    reference_targets = None
    for run in run_ids:
        columns = {}
        for quantile in quantiles:
            path = latest_output(
                Path(experiment_path) / f"run_{run + 1}" / f"quantile_{quantile}",
                file_pos,
            )
            outputs = pd.read_pickle(path)
            targets = [np.asarray(target).squeeze() for target in outputs["y_true"]]
            if reference_targets is None:
                reference_targets = targets
            elif len(targets) != len(reference_targets) or any(
                not np.array_equal(a, b) for a, b in zip(reference_targets, targets)
            ):
                raise ValueError(f"Test targets or life order differ in {path}")
            columns[f"quantile_{quantile}"] = output_life_metrics(
                outputs, first_life_id
            )
            sources.append(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        frames.append(pd.DataFrame(columns))
    return frames, sources, reference_targets


def main(argv=None):
    # Imports remain lightweight and the PHM command's behavior is unchanged.
    from quantile_metrics import aggregate_metrics, comparison_table

    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--exp_config_path", default=str(DEFAULT_CONFIG))
    selected, _ = pre.parse_known_args(argv)
    with open(selected.exp_config_path) as stream:
        config = yaml.safe_load(stream)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_name", choices=["CMAPSS"], default="CMAPSS")
    parser.add_argument("--exp_config_path", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model_names", nargs="+")
    parser.add_argument("--model_name")
    parser.add_argument("--exp_names", nargs="+")
    parser.add_argument("--exp_name")
    parser.add_argument("--cmapss_models", default=config.get("cmapss_models", "FD001"))
    parser.add_argument("--approach", default=config.get("approach", "windowed"))
    parser.add_argument(
        "--quantile_approach", default=config.get("quantile_approach", "quantile_reg")
    )
    parser.add_argument(
        "--quantiles",
        nargs="+",
        type=float,
        default=config.get("quantiles", [0.1, 0.25, 0.5, 0.75, 0.9]),
    )
    parser.add_argument("--n_runs", type=int, default=config.get("n_runs", 15))
    parser.add_argument(
        "--start_run_id", type=int, default=config.get("start_run_id", 0)
    )
    parser.add_argument("--file_pos", type=int, default=config.get("file_pos", 0))
    parser.add_argument(
        "--report_dir",
        type=Path,
        help="Write table.md, provenance.json and per-model mean-life pickles here",
    )
    parser.add_argument("--print_summary_metrics", action="store_true")
    args = parser.parse_args(argv)
    if args.n_runs <= args.start_run_id or args.start_run_id < 0:
        parser.error("Require 0 <= start_run_id < n_runs")
    if (
        not args.quantiles
        or len(set(args.quantiles)) != len(args.quantiles)
        or any(not 0 < q < 1 for q in args.quantiles)
    ):
        parser.error("Quantiles must be unique levels strictly between zero and one")
    if args.model_name and args.model_names:
        parser.error("Choose --model_name or --model_names")
    models = args.model_names or (
        [args.model_name] if args.model_name else config["model_names"]
    )
    if args.exp_name and args.exp_names:
        parser.error("Choose --exp_name or --exp_names")
    if args.exp_name:
        experiments = [args.exp_name] * len(models)
    elif args.exp_names:
        experiments = args.exp_names
    else:
        mapping = dict(zip(config.get("model_names", []), config.get("exp_names", [])))
        try:
            experiments = [mapping[model] for model in models]
        except KeyError as error:
            parser.error(
                f"No configured experiment for {error.args[0]}; supply --exp_names"
            )
    if len(experiments) != len(models) or len(set(models)) != len(models):
        parser.error("Supply one experiment per model, with unique model names")
    outputs_root = (
        Path(
            os.environ.get(
                "CMAPSS_OUTPUTS_DIR", DEFAULT_CONFIG.parent.parent / "outputs"
            )
        )
        .expanduser()
        .resolve()
    )
    model_metrics, provenance = {}, []
    reference_targets = None
    for model, experiment in zip(models, experiments):
        path = (
            outputs_root
            / model
            / args.cmapss_models
            / args.approach
            / args.quantile_approach
            / experiment
        )
        frames, sources, targets = load_output_metrics(
            path,
            range(args.start_run_id, args.n_runs),
            args.quantiles,
            first_life_id=config.get("test_idx", [0])[0] + 1,
            file_pos=args.file_pos,
        )
        if reference_targets is None:
            reference_targets = targets
        elif len(targets) != len(reference_targets) or any(
            not np.array_equal(a, b) for a, b in zip(targets, reference_targets)
        ):
            raise ValueError(f"Test targets or life order differ for model {model}")
        frame = aggregate_metrics(frames)
        model_metrics[model] = frame
        print(
            f"{model}: {experiment}; {len(frames)} runs, {len(targets)} test lives, {len(args.quantiles)} quantiles"
        )
        if args.print_summary_metrics:
            print(frame.loc[["Life_mean", "Life_median", "Life_std"]].to_markdown())
        provenance.append(
            {
                "model": model,
                "experiment": experiment,
                "runs": len(frames),
                "test_lives": len(targets),
                "sources": sources,
            }
        )
    markdown = comparison_table(model_metrics).to_markdown()
    definition = "Life_mean +- Life_std: mean and sample SD across test lives after averaging each life's RMSE across runs. Zero targets excluded; quantiles kept separate."
    if args.report_dir:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        (args.report_dir / "table.md").write_text(
            f"# {args.cmapss_models} quantile RMSE\n\n{definition}\n\n{markdown}\n"
        )
        metadata = {
            "outputs_root": str(outputs_root),
            "dataset": args.cmapss_models,
            "approach": args.approach,
            "quantile_approach": args.quantile_approach,
            "quantiles": args.quantiles,
            "run_ids": list(range(args.start_run_id + 1, args.n_runs + 1)),
            "definition": definition,
            "file_pos": args.file_pos,
            "first_life_id": config.get("test_idx", [0])[0] + 1,
            "rounding": "Run-averaged per-life RMSE rounded to 2 decimals before life summaries, matching the PHM table.",
            "models": provenance,
        }
        (args.report_dir / "provenance.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        for model, frame in model_metrics.items():
            frame.to_pickle(args.report_dir / f"{model}_mean_metrics.pickle")
        print(f"Saved report to {args.report_dir.resolve()}")
    print(definition)
    print(markdown)


if __name__ == "__main__":
    main()
