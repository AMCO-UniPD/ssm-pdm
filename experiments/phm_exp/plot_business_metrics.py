"""
Python script to produce the business metrics plots:
unexpected breaks and unexploited lifetime
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

from ceruleo.graphics.results import (
    plot_unexpected_breaks,
    plot_unexploited_lifetime,
    plot_J_Cost
)

experiment_path = os.path.dirname((os.path.realpath(__file__)))

exp_config, model_config, device, exp_name = setup_exp()

plot_path = generate_path(
    basepath=experiment_path,
    folders=[
        "business_metrics_plots",
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

        combined_outputs_path = generate_path(
            basepath=experiment_path,
            folders=[
                "baseline_outputs",
                model_name,
                ]
        )

    else:

        combined_outputs_path = generate_path(
            basepath=experiment_path,
            folders=[
                "combined_outputs",
                model_name,
                exp_config.failure_type,
                exp_config.plot_approach,
                exp_name,
                f"run_{exp_config.plot_run_id}"
            ],
        )

    outputs_dict_path=get_most_recent_file(combined_outputs_path)
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

bin_edges, model_results = models_cv_results(
    results_dict = results_dict,
    nbins = 10
)

base_filename = f"{get_current_time()}_{exp_config.failure_type}_{exp_config.plot_approach}"

if exp_config.plot_ub:

    _ = plot_unexpected_breaks(
        results_dict = results_dict,
        max_window = exp_config.max_windows,
        n = exp_config.n_maintenance_windows,
        add_shade = exp_config.add_shade,
        save_plot = True,
        filename = f"{base_filename}_unexpected_breaks.png",
        plot_path = plot_path
    )

if exp_config.plot_ul:

    _ = plot_unexploited_lifetime(
        results_dict = results_dict,
        max_window = exp_config.max_windows,
        n = exp_config.n_maintenance_windows,
        add_shade = exp_config.add_shade,
        log_scale = exp_config.log_scale,
        save_plot = True,
        filename = f"{base_filename}_unexploited_lifetime.png",
        plot_path = plot_path
    )

if exp_config.plot_J:

    _ = plot_J_Cost(
        results = results_dict,
        window = exp_config.max_windows,
        step = exp_config.n_maintenance_windows,
        save_plot = True,
        filename = f"{base_filename}_J_Cost.png",
        plot_path = plot_path
    )

