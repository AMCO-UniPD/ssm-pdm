"""
Python script to compute business metrics for a non quantile regression model
"""

# general imports
import os
import sys
import ipdb
import setproctitle

src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.append(src_path)

from exp_config import setup_exp
from utils import generate_path
from perf import get_fitted_lifes, lifes_business_metrics
from models import wandb_data

experiment_path = os.path.dirname((os.path.realpath(__file__)))

exp_config, model_config, device, exp_name = setup_exp()

is_baseline_model = exp_config.model_name in ["mean", "median"]

if is_baseline_model:

    outputs_path = generate_path(
        basepath=experiment_path,
        folders=[
            "baseline_outputs",
            exp_config.model_name,
        ],
    )

    business_metrics_path = generate_path(
        basepath=experiment_path,
        folders=[
            "baseline_business_metrics",
            exp_config.model_name,
        ],
    )

else:

    outputs_path = generate_path(
        basepath=experiment_path,
        folders=[
            "combined_outputs",
            exp_config.model_name,
            exp_config.failure_type,
            exp_config.approach,
            exp_name
        ],
    )

    business_metrics_path = generate_path(
        basepath=experiment_path,
        folders=[
            "business_metrics",
            exp_config.model_name,
            exp_config.failure_type,
            exp_config.approach,
            exp_name
        ],
    )

setproctitle.setproctitle(f"compute_business_metrics_standard_{exp_name}")

wandb_data(config=exp_config,model_config=model_config)

for life_id in exp_config.life_idx:

    print("-"*50)
    print(f"Computing business metrics for life {life_id}")
    print("-"*50)

    fitted_lifes = get_fitted_lifes(
        config=exp_config,
        outputs_path=outputs_path,
        is_baseline=is_baseline_model,
        life_id=life_id
    )

    business_metrics_path_life_id = generate_path(
        basepath=business_metrics_path,
        folders=[f"life_{life_id}"]
    )

    business_metrics_df = lifes_business_metrics(
        fitted_lifes=fitted_lifes,
        config=exp_config,
        metrics_path=business_metrics_path_life_id,
    )

