"""
Python script to compute business metrics for a quantile regression model
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
from perf import lifes_business_metrics
from models import wandb_data

experiment_path = os.path.dirname((os.path.realpath(__file__)))

exp_config, model_config, device, exp_name = setup_exp()

#TODO: Here I can use the quantile baseline models when I implement them

is_baseline_model = "quantile" in exp_config.model_name

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

setproctitle.setproctitle(f"compute_business_metrics_{exp_name}")

wandb_data(config=exp_config,model_config=model_config)

for quantile in exp_config.quantiles:

    print("-"*50)
    print(f"Computing business metrics for quantile {quantile}")
    print("-"*50)

    business_metrics_df = lifes_business_metrics(
        config=exp_config,
        outputs_path=outputs_path,
        metrics_path=business_metrics_path,
        is_baseline=is_baseline_model,
        tau=quantile
    )
