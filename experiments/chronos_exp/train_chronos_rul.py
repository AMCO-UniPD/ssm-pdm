"""
Training script for the `chronos-pdm` project
"""

# general imports
import os
import sys
import ipdb
import torch
import argparse
import setproctitle

chronos_path = os.path.join(os.path.dirname(__file__),"..","..","src",
                            "chronos-rul","src")
src_path = os.path.join(os.path.dirname(__file__),"..","..","src",
)
sys.path.append(src_path)
sys.path.append(chronos_path)

from utils import (
    ExperimentConfig,
    generate_path,
    get_most_recent_file,
    load_yaml_to_dict,
    open_element,
)

from models import (
    wandb_run,
    best_model_perf
)

from loss import load_loss_functions
from perf import (
        lifes_metrics,
        df_with_index_to_obsidian_table,
)
from plots import plot_predictions_grid

experiment_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"chronos_exp")

parser=argparse.ArgumentParser(description="Training script chronos-rul")
parser.add_argument("--exp_config_path",type=str,default="config/exp_config.yaml",help="Path to the experiment config file")
args=parser.parse_args()

exp_config=load_yaml_to_dict(args.exp_config_path)
exp_config=ExperimentConfig(exp_config)
model_config=load_yaml_to_dict(exp_config.model_config_path)

device = torch.device(f"cuda:{exp_config.device_num}" if torch.cuda.is_available() else "cpu")

print("#"*50)
print(f"Using device: {device}")
print("#"*50)

# Set up the paths

best_model_path = generate_path(basepath=experiment_path,
                                   folders=["best_models",
                                            exp_config.model_name,
                                            exp_config.cmapss_models])

outputs_path = generate_path(basepath=experiment_path,
                                   folders=["outputs",
                                            exp_config.model_name,
                                            exp_config.cmapss_models])

metrics_path = generate_path(basepath=experiment_path,
                             folders=["metrics",
                                      exp_config.model_name,
                                      exp_config.cmapss_models])

plot_path = generate_path(basepath=experiment_path,
                             folders=["plots",
                                      exp_config.model_name,
                                      exp_config.cmapss_models])

if exp_config.test_script:

    print("#"*50)
    print("Running best model performance test")
    print("#"*50)

    setproctitle.setproctitle("chronos-rul-test-script")

    if exp_config.save_outputs:

        print("#"*50)
        print("Saving outputs")
        print("#"*50)

        best_model_perf(
            config=exp_config,
            model_config=model_config,
            device=device,
            best_model_path=best_model_path,
            outputs_path=outputs_path,
        )

    if exp_config.compute_metrics:

        print("#" * 50)
        print("Computing metrics for each life and for each sensor in the test set")
        print("#" * 50)

        metrics_df = lifes_metrics(
            config=exp_config,
            outputs_path=outputs_path,
            metrics_path=metrics_path,
        )
        print("#" * 50)
        print(f"metrics_df shape: {metrics_df.shape}")

    if exp_config.obsidian_table:

       print("#" * 50)
       print("Producing the obsidian table")
       print("#" * 50)

       metrics_path = get_most_recent_file(metrics_path, file_pos=exp_config.file_pos)
       metrics_df = open_element(metrics_path)
       print(df_with_index_to_obsidian_table(metrics_df))

    if exp_config.plot_preds:

        print("#" * 50)
        print("Producing grid plot of the predictions")
        print("#" * 50)

        fig = plot_predictions_grid(
            config=exp_config,
            sensor_idx=exp_config.sensor_idx,
            outputs_path=outputs_path,
            plot_path=plot_path,
        )

else:

    print("#"*50)
    print("Model training started")
    print(f"Model name: {exp_config.model_name}")
    print(f"Pretrained model id: {exp_config.model_id}") if not model_config["random_init"] else print("Random initialization")
    print(f"CMAPSS model: {exp_config.cmapss_models}")
    print(f"Val idx: {exp_config.val_idx}")
    print(f"Test idx: {exp_config.test_idx}")
    print(f"Transformer type: {exp_config.transformer_type}")
    print(f"Scaler: {exp_config.scaler}")
    print(f"Epochs: {exp_config.epochs}")
    print(f"Learning rate: {exp_config.lr}")
    print(f"Sequence length: {exp_config.seq_len}")
    print(f"Training loss: {exp_config.loss}")
    print(f"Eval loss: {exp_config.eval_loss}")
    print("#"*50)

    run_name=f"{exp_config.model_name}_{exp_config.cmapss_models}"
    setproctitle.setproctitle(run_name)
    model,model_info = wandb_run(
        run_name=run_name,
        config=exp_config,
        model_config=model_config,
        device=device,
        best_model_path=best_model_path,
        metrics_path=metrics_path,
        plot_path=plot_path
    )
