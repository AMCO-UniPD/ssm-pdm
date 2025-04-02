"""
Python script to run statistical tests on the metrics_df created
in quantile_reg_metrics.py
"""

# general imports
import os
import sys
import ipdb
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower

src_path = os.path.join(os.path.dirname(__file__),"..","..","src",
)
sys.path.append(src_path)

from utils import (
    generate_path,
    load_yaml_to_dict,
    ExperimentConfig,
    get_most_recent_file,
    open_element,
)

from perf import print_summary_metrics

experiment_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"chronos_exp")

config_path=os.path.join(experiment_path,"config","ssm_exp_config.yaml")
config=load_yaml_to_dict(config_path)
config=ExperimentConfig(config)


metrics_df_dirpath = generate_path(basepath=experiment_path,
                                folders=[
                                    "metrics",
                                    config.model_name,
                                    config.cmapss_models,
                                    config.approach,
                                    "quantile_reg",
                                    config.exp_name,
                                ])

if config.print_summary_metrics:

    ipdb.set_trace()
    metrics_df_path = get_most_recent_file(metrics_df_dirpath,file_pos=config.file_pos)
    print('#'* 50)
    print(f"Opening metrics_df from {metrics_df_path}")
    print('#'* 50)

    metrics_df = open_element(metrics_df_path)
    print_summary_metrics(metrics_df)

if config.stat_test:

    metrics_df_runs_dirpath = generate_path(basepath=metrics_df_dirpath,folders=["metrics_df_runs"])
    metrics_df_runs_path = get_most_recent_file(metrics_df_runs_dirpath,file_pos=config.file_pos)
    print('#'* 50)
    print(f"Opening metrics_df from {metrics_df_runs_path}")
    print('#'* 50)

    metrics_dfs = open_element(metrics_df_runs_path)

    print('#'* 50)
    print(f"Executing independent T test on quantiles: {config.quantile_to_test[0]} and {config.quantile_to_test[1]}")
    print('#'* 50)

    metrics_df_run=[]
    runs=[]
    for i,metrics_df in enumerate(metrics_dfs):
        metrics_df_run.append(metrics_df.mean(axis=0))
        runs.append(f"run_{i+1}")

    metrics_df_runs=pd.concat(metrics_df_run,axis=1)
    metrics_df_runs.columns=runs
    q1=metrics_df_runs.loc[f"quantile_{config.quantile_to_test[0]}"]
    q2=metrics_df_runs.loc[f"quantile_{config.quantile_to_test[1]}"]
    t_stat,p_val=stats.ttest_ind(q1,q2)
    print('#'* 50)
    print(f"Independent T-test results between quantiles {config.quantile_to_test[0]} and {config.quantile_to_test[1]}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_val:.4f}")
    print('#'* 50)

    if config.power_analysis:
        effect_size=abs(q1.mean()-q2.mean())/np.std(np.concatenate([q1,q2]))
        analysis=TTestIndPower()
        sample_size=analysis.solve_power(
                                         effect_size=effect_size,
                                         alpha=0.05,
                                         power=0.8,
                                         alternative='two-sided'
                                         )
        print('#'* 50)
        print(f"Power analysis between quantiles {config.quantile_to_test[0]} and {config.quantile_to_test[1]}")
        print(f"Suggested sample size: {sample_size}")
        print('#'* 50)
