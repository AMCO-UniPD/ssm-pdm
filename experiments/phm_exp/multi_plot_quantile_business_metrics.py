"""
Python script to produce the business metrics plots
for Quantile Regression model → we compare the unexpected breaks
and unexploited lifetime for a specific quantile for different models
"""

import os
import sys
import setproctitle
import ipdb

src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.append(src_path)

from config_vars import BASELINE_MODEL_NAMES
from exp_config import setup_exp
from perf import mask_outputs
from utils import (
    generate_path,
    get_most_recent_file,
    open_element,
    get_current_time,
)

from ceruleo.results.results import (
    FittedLife,
    PredictionResult,
    unexpected_breaks,
    unexploited_lifetime,
    models_cv_results
)

from plots import plot_business_metrics

experiment_path = os.path.dirname((os.path.realpath(__file__)))

exp_config, model_config, device, exp_name = setup_exp()

plot_path = generate_path(
    basepath=experiment_path,
    folders=[
        "multi_quantile_business_metrics_plots",
        exp_config.failure_type,
        exp_config.plot_approach,
    ],
)

results_dict = {}

for model_name, exp_name in zip(exp_config.model_names, exp_config.exp_names):
    print(f"model: {model_name}")
    print(f"experiment_name: {exp_name}")
    print("-"*50)

    if model_name in BASELINE_MODEL_NAMES:

        quantile_combined_outputs_path = generate_path(
            basepath=experiment_path,
            folders=[
                "baseline_outputs",
                model_name,
                ]
        )

        outputs_dict_path=get_most_recent_file(quantile_combined_outputs_path)
        outputs_dict=open_element(outputs_dict_path,filetype="pickle")
        outputs_dict = mask_outputs(outputs_dict=outputs_dict, life_idx=exp_config.life_idx)

        prediction_results = [
            PredictionResult(
                name=f"Life_{i+1}",
                true_RUL=outputs_dict["y_true"][i],
                predicted_RUL=outputs_dict["y_pred"][i],
            )
            for i in range(len(outputs_dict["y_true"]))
        ]

        results_dict[model_name] = prediction_results

    else:

        quantile_combined_outputs_path = generate_path(
            basepath=experiment_path,
            folders=[
                "combined_outputs",
                model_name,
                exp_config.failure_type,
                exp_config.plot_approach,
                exp_name,
                f"run_{exp_config.plot_run_id}" if not exp_config.cv else f"fold_{exp_config.plot_run_id}",
                f"quantile_{exp_config.quantile_run}"
            ],
        )

        outputs_dict_path=get_most_recent_file(quantile_combined_outputs_path)
        outputs_dict=open_element(outputs_dict_path,filetype="pickle")
        outputs_dict = mask_outputs(outputs_dict=outputs_dict, life_idx=exp_config.life_idx)

        prediction_results = [
            PredictionResult(
                name=f"Life_{i+1}",
                true_RUL=outputs_dict["y_true"][i],
                predicted_RUL=outputs_dict["y_pred"][i],
            )
            for i in range(len(outputs_dict["y_true"]))
        ]

        #WARN: Remove the quantile_{exp_config.quantile_run} from the model name
        # results_dict[f"{model_name}_quantile_{exp_config.quantile_run}"] = prediction_results
        results_dict[model_name] = prediction_results

base_filename = f"{get_current_time()}_{exp_config.failure_type}_{exp_config.plot_approach}_multi_plot_business_metrics_quantile"

print("-"*50)
print("Producing multi business metrics plots")
print("-"*50)

plot_business_metrics(
    exp_config = exp_config,
    results_dict = results_dict,
    plot_path = plot_path,
    base_filename = base_filename
)
