# Quantile RMSE summaries

Run the script from the repository root for CMAPSS:

```sh
export CMAPSS_OUTPUTS_DIR=/home/davide_frizzo/chronos-pdm/experiments/chronos_exp/outputs
python experiments/phm_exp/quantile_metrics.py \
  --data_name CMAPSS \
  --report_dir reports/cmapss-fd001-quantile-metrics
```

`CMAPSS_OUTPUTS_DIR` points directly to the `outputs` directory. If unset, the
script uses this checkout's `experiments/chronos_exp/outputs` directory.
The input layout is:

```text
$CMAPSS_OUTPUTS_DIR/<model>/<dataset>/<approach>/<quantile_approach>/<experiment>/run_<n>/quantile_<q>/*.pickle
```

The default configuration is
[`../chronos_exp/config/quantile_metrics.yaml`](../chronos_exp/config/quantile_metrics.yaml).
It selects the six benchmark models, their explicit FD001 experiment names,
runs 1–15, and quantiles 0.1, 0.25, 0.5, 0.75, and 0.9. It uses the experiments
without `_tau_mult`, `_tau_mult_no_tau_feat`, or `_feat_extraction` suffixes.
The experiment names were matched to the legacy config and checked for complete
outputs; they are not inferred from directory modification times.

To change the selection, use `--exp_config_path`, or override values with
`--model_names`, `--exp_names` (one name per model, in the same order),
`--cmapss_models`, `--approach`, `--quantile_approach`, `--quantiles`,
`--start_run_id`, or `--n_runs`. `--model_name S4` selects one configured model.
`--start_run_id` is zero based, while `--n_runs` is the exclusive upper bound;
0 and 15 select run_1 through run_15. Within each run/quantile folder the newest
pickle is used, or the next newest with `--file_pos 1`.

For each life, quantile, and run, RMSE is computed from `y_pred` and `y_true`,
excluding zero targets as in the legacy evaluator. Each life's RMSE is averaged
across runs. These per-life means are rounded to two decimals, matching the PHM
summary. `Life_mean` and `Life_std` are then the equally weighted mean and sample
standard deviation (`ddof=1`) across those lives. Quantiles stay separate. A
single life has an undefined sample SD (`nan`). The FD001 outputs contain the
second half of the test set: 50 lives, labelled Life_51 through Life_100.

The command always ends with a Markdown table. `--report_dir` additionally saves
`table.md`, per-model mean-life pickle files, and `provenance.json` with exact
input paths, SHA-256 checksums, experiments, runs, and aggregation settings.
It does not modify the source outputs. `--print_summary_metrics` adds diagnostic
summary rows without changing the final table.

The PHM invocation and `PHM_RESULTS_DIR` behavior remain available, for example
from `experiments/phm_exp`:

```sh
python quantile_metrics.py --model_names S4 S5 --save_mean_metrics_df
```
