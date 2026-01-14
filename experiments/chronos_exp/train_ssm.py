"""
Training script for the `SSM` models in the `chronos-pdm` project
"""

# general imports
import os
import sys
import ipdb
import torch
import argparse
import setproctitle

chronos_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "chronos-rul", "src"
)
src_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "src",
)
sys.path.append(src_path)
sys.path.append(chronos_path)

from utils import (
    ExperimentConfig,
    generate_path,
    get_most_recent_file,
    load_yaml_to_dict,
    open_element,
    get_current_time,
    save_element,
    set_seed,
)

from models import wandb_run, best_model_perf

from ssm_models import (
    load_ssm_model,
    ModelConfig,
)

from loss import load_loss_functions
from perf import (
    lifes_metrics,
    sub_lifes_metrics,
    df_with_index_to_obsidian_table,
)
from plots import plot_predictions_grid

experiment_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "chronos_exp"
)

parser = argparse.ArgumentParser(
    description="Training script for SSM models in chronos-pdm project"
)
parser.add_argument(
    "--exp_config_path",
    type=str,
    default="config/exp_config.yaml",
    help="Path to the experiment config file",
)
args = parser.parse_args()

exp_config = load_yaml_to_dict(args.exp_config_path)
exp_config = ExperimentConfig(exp_config)
model_config = load_yaml_to_dict(exp_config.model_config_path)
model_config = ModelConfig(model_config)
model_config.quantile_reg = exp_config.quantile_reg

device = torch.device(f"cuda:{exp_config.device_num}" if torch.cuda.is_available() else "cpu")
model_config.device = device

print("#" * 50)
print(f"Using device: {device}")
print("#" * 50)

best_model_path = generate_path(
    basepath=experiment_path,
    folders=[
        "best_models",
        exp_config.model_name,
        exp_config.cmapss_models,
        exp_config.approach,
    ],
)

outputs_path = generate_path(
    basepath=experiment_path,
    folders=[
        "outputs",
        exp_config.model_name,
        exp_config.cmapss_models,
        exp_config.approach,
    ],
)

metrics_path = generate_path(
    basepath=experiment_path,
    folders=[
        "metrics",
        exp_config.model_name,
        exp_config.cmapss_models,
        exp_config.approach,
    ],
)


plot_path = generate_path(
    basepath=experiment_path,
    folders=[
        "plots",
        exp_config.model_name,
        exp_config.cmapss_models,
        exp_config.approach,
    ],
)


if exp_config.test_script:
    print("#" * 50)
    print("Running best model performance test")
    print("#" * 50)

    setproctitle.setproctitle(f"{exp_config.model_name}-test-script")

    if exp_config.save_outputs or exp_config.return_outputs:
        if exp_config.quantile_reg:
            if exp_config.save_outputs_quantile:
                print("#" * 50)
                print("Saving outputs for each quantile level separately")
                print("#" * 50)

                for run in range(exp_config.n_runs):
                    print("#" * 50)
                    print(f"Saving outputs for run: {run+1}")
                    print("#" * 50)

                    if exp_config.save_summary_dict:
                        exp_config.quantiles = [exp_config.quantile_run]

                    for quantile in exp_config.quantiles:
                        print("#" * 50)
                        print(f"Saving outputs for quantile level: {quantile}")
                        print("#" * 50)

                        quantile_reg_folders = [
                            "quantile_reg",
                            exp_config.exp_name,
                            f"run_{run+1}",
                            f"quantile_{quantile}",
                        ]
                        quantile_outputs_path = generate_path(
                            basepath=outputs_path, folders=quantile_reg_folders
                        )
                        quantile_best_model_path = generate_path(
                            basepath=best_model_path, folders=quantile_reg_folders
                        )

                        best_model_perf(
                            config=exp_config,
                            model_config=model_config,
                            device=device,
                            best_model_path=quantile_best_model_path,
                            outputs_path=quantile_outputs_path,
                            tau=quantile,
                        )

            if exp_config.save_outputs_run:
                print("#" * 50)
                print(
                    "Save the prediction on the different quantiles of the same run together"
                )
                print("#" * 50)

                for run in range(exp_config.n_runs):
                    print("#" * 50)
                    print(f"Saving outputs for run: {run+1}")
                    print("#" * 50)

                    run_folders = [
                        "quantile_reg",
                        exp_config.exp_name,
                        f"run_{run+1}",
                        f"quantile_{exp_config.quantile_run}",
                    ]
                    run_outputs_path = generate_path(
                        basepath=outputs_path, folders=run_folders
                    )
                    best_model_run_path = generate_path(
                        basepath=best_model_path, folders=run_folders
                    )

                    run_outputs_dict = {}

                    for i, quantile in enumerate(exp_config.quantiles):
                        # Compute the predictions of the i-th quantile with the best model
                        outputs_dict = best_model_perf(
                            config=exp_config,
                            model_config=model_config,
                            device=device,
                            best_model_path=best_model_run_path,
                            outputs_path=run_outputs_path,
                            tau=quantile,
                        )

                        y_pred, y_true = outputs_dict["y_pred"], outputs_dict["y_true"]

                        if i == 0:
                            run_outputs_dict["y_true"] = y_true

                        run_outputs_dict[f"quantile_{quantile}"] = y_pred

                    run_outputs_dict_path = generate_path(
                        basepath=run_outputs_path,
                        folders=[f"outputs_quantile_{exp_config.quantile_run}"],
                    )

                    save_element(
                        element=run_outputs_dict,
                        dirpath=run_outputs_dict_path,
                        filename=f"{get_current_time()}_{exp_config.model_name}_{exp_config.cmapss_models}_{exp_config.approach}_run_{run+1}_outputs",
                        filetype="pickle",
                    )

        else:
            print("#" * 50)
            print("Saving outputs")
            print("#" * 50)

            best_model_perf(
                config=exp_config,
                model_config=model_config,
                device=device,
                best_model_path=best_model_path,
                outputs_path=outputs_path,
            )

    if exp_config.compute_metrics:
        if exp_config.quantile_reg:
            for quantile in exp_config.quantiles:
                print("#" * 50)
                print(f"Computing metrics for quantile level: {quantile}")
                print("#" * 50)

                quantile_reg_folders = [
                    "quantile_reg",
                    args.exp_name,
                    f"quantile_{quantile}",
                ]
                quantile_metrics_path = generate_path(
                    basepath=metrics_path, folders=quantile_reg_folders
                )
                quantile_outputs_path = generate_path(
                    basepath=outputs_path, folders=quantile_reg_folders
                )

                metrics_df = lifes_metrics(
                    config=exp_config,
                    outputs_path=quantile_outputs_path,
                    metrics_path=quantile_metrics_path,
                )
        else:
            print("#" * 50)
            print("Computing metrics for each life and for each sensor in the test set")
            print("#" * 50)

            metrics_df = lifes_metrics(
                config=exp_config, outputs_path=outputs_path, metrics_path=metrics_path
            )

    if exp_config.obsidian_table:
        if exp_config.quantile_reg:
            for quantile in exp_config.quantiles:
                print("#" * 50)
                print(f"Producing the obsidian table for quantile level: {quantile}")
                print("#" * 50)

                quantile_reg_folders = [
                    "quantile_reg",
                    exp_config.exp_name,
                    f"quantile_{quantile}",
                ]
                quantile_metrics_path = generate_path(
                    basepath=metrics_path, folders=quantile_reg_folders
                )
                metrics_df_path = get_most_recent_file(
                    quantile_metrics_path, file_pos=exp_config.file_pos
                )
                metrics_df = open_element(metrics_df_path)
                print("#" * 50)
                print(f"Obsidian table for quantile level: {quantile}")
                print("#" * 50)
                if exp_config.sub_lifes_metrics:
                    sub_metrics_df = sub_lifes_metrics(
                        config=exp_config, metrics_df=metrics_df
                    )
                    obsidian_table = df_with_index_to_obsidian_table(sub_metrics_df)
                else:
                    obsidian_table = df_with_index_to_obsidian_table(metrics_df)

                print(obsidian_table)
        else:
            print("#" * 50)
            print("Producing the obsidian table")
            print("#" * 50)

            metrics_path = get_most_recent_file(
                metrics_path, file_pos=exp_config.file_pos
            )
            metrics_df = open_element(metrics_path)
            if exp_config.sub_lifes_metrics:
                sub_metrics_df = sub_lifes_metrics(
                    config=exp_config, metrics_df=metrics_df
                )
                obsidian_table = df_with_index_to_obsidian_table(sub_metrics_df)
            else:
                obsidian_table = df_with_index_to_obsidian_table(metrics_df)

            print(obsidian_table)

    if exp_config.plot_preds:
        if exp_config.quantile_reg:
            for quantile in exp_config.quantiles:
                print("#" * 50)
                print(
                    f"Producing grid plot of the predictions for quantile level: {quantile}"
                )
                print("#" * 50)

                quantile_reg_folders = [
                    "quantile_reg",
                    args.exp_name,
                    f"quantile_{quantile}",
                ]
                quantile_outputs_path = generate_path(
                    basepath=outputs_path, folders=quantile_reg_folders
                )
                quantile_plot_path = generate_path(
                    basepath=plot_path, folders=quantile_reg_folders
                )

                plot_predictions_grid(
                    config=exp_config,
                    outputs_path=quantile_outputs_path,
                    plot_path=quantile_plot_path,
                )
        else:
            print("#" * 50)
            print("Producing grid plot of the predictions")
            print("#" * 50)

            plot_predictions_grid(
                config=exp_config, outputs_path=outputs_path, plot_path=plot_path
            )

else:
    print("#" * 50)
    print("Model training started")
    print("Experiment configuration")
    print("#" * 50)
    print(f"Model name: {exp_config.model_name}")
    print(f"CMAPSS model: {exp_config.cmapss_models}")
    print(f"Val idx: {exp_config.val_idx}")
    print(f"Test idx: {exp_config.test_idx}")
    print(f"Transformer type: {exp_config.transformer_type}")
    print(f"Scaler: {exp_config.scaler}")
    print(f"Epochs: {exp_config.epochs}")
    print(f"Sequence length: {exp_config.sequence_length}")
    print(f"Training loss: {exp_config.loss}")
    print(f"Eval loss: {exp_config.eval_loss}")
    print(f"Approach: {exp_config.approach}")
    print("Model configuration")
    print("#" * 50)
    print(f"Learning rate: {model_config.lr}")
    print(f"Number of fc layers: {model_config.n_layers}")
    print(f"Activation function: {model_config.activation}")
    print(f"Dropout: {model_config.dropout}")
    print("#" * 50)

    if exp_config.quantile_reg:
        if exp_config.set_exp_name:
            exp_name = exp_config.exp_name
            print("#" * 50)
            print(f"Experiment name set to: {exp_name}")
            print("#" * 50)
        else:
            exp_time = get_current_time()
            if exp_config.n_runs > 1:
                exp_name = f"multi_run_{exp_time}_{exp_config.model_name}_{exp_config.cmapss_models}_{exp_config.approach}_quantile_reg"
            else:
                exp_name = f"{exp_time}_{exp_config.model_name}_{exp_config.cmapss_models}_{exp_config.approach}_quantile_reg"

            if exp_config.transformer_type == 2:
                exp_name = f"{exp_name}_feat_extraction"

            if model_config.tau_mult:
                exp_name = f"{exp_name}_tau_mult"

            if not model_config.tau_feat:
                exp_name = f"{exp_name}_no_tau_feat"

        print("#" * 50)
        print(
            f"Starting quantile regression experiment: {exp_name}"
        ) if not exp_config.set_exp_name else print(
            f"Continuing quantile regression experiment: {exp_name}"
        )
        print(f"Quantile distribution: {exp_config.quantile_dist}")
        print(f"Distribution parameters: {exp_config.bounds}")
        print(f"Quantile levels for evaluation: {exp_config.quantiles}")
        print(f"Number of runs: {exp_config.n_runs}")
        print(f"Start run id: {exp_config.start_run_id}")
        print("#" * 50)

        assert (
            exp_config.loss == "quantile_reg"
        ), "The loss function must be quantile for quantile regression"
        assert (
            exp_config.compute_metrics == False
        ), "We do not compute the single metrics_df for each quantile and in each run"
        assert (
            exp_config.obsidian_table == False
        ), "We do not produce the obsidian table for each quantile and in each run"
        assert (
            exp_config.plot_preds == False
        ), "We do not plot the predictions for each quantile and in each run"

        for run in range(
            exp_config.start_run_id, exp_config.start_run_id + exp_config.n_runs
        ):
            print(f"Starting run: {run+1}")
            print("#" * 50)
            print(f"Setting the seed for this run to {run+1}")
            set_seed(run)

            for quantile in exp_config.quantiles:
                print("#" * 50)
                print(f"Starting experiment for quantile level: {quantile}")
                print("#" * 50)

                run_name = f"{exp_config.model_name}_{exp_config.cmapss_models}_{exp_config.approach}_run_{run+1}_quantile_{quantile}_bounds_{exp_config.bounds[0]}_{exp_config.bounds[1]}"

                if exp_config.transformer_type == 2:
                    run_name = f"{run_name}_feat_extraction"

                if model_config.tau_mult:
                    run_name = f"{run_name}_tau_mult"

                if not model_config.tau_feat:
                    run_name = f"{run_name}_no_tau_feat"

                setproctitle.setproctitle(run_name)

                quantile_reg_folders = [
                    "quantile_reg",
                    exp_name,
                    f"run_{run+1}",
                    f"quantile_{quantile}",
                ]

                quantile_best_model_path = generate_path(
                    basepath=best_model_path, folders=quantile_reg_folders
                )
                quantile_outputs_path = generate_path(
                    basepath=outputs_path, folders=quantile_reg_folders
                )

                model, model_info = wandb_run(
                    run_name=run_name,
                    config=exp_config,
                    model_config=model_config,
                    device=device,
                    best_model_path=quantile_best_model_path,
                    outputs_path=quantile_outputs_path,
                    tau=quantile,
                )

    else:
        if model_config.gap:
            run_name = f"{exp_config.model_name}_{exp_config.cmapss_models}_{exp_config.approach}_gap"
        else:
            run_name = f"{exp_config.model_name}_{exp_config.cmapss_models}_{exp_config.approach}"

        if exp_config.transformer_type == 2:
            run_name = f"{run_name}_feat_extraction"

        if exp_config.loss == "pinball":
            run_name = f"{run_name}_pinball_{exp_config.tau}"

        setproctitle.setproctitle(run_name)

        model, model_info = wandb_run(
            run_name=run_name,
            config=exp_config,
            model_config=model_config,
            device=device,
            best_model_path=best_model_path,
            outputs_path=outputs_path,
            metrics_path=metrics_path,
            plot_path=plot_path,
        )
