# PHM window-weight ablation

Run `./run_phm_weight_ablation` from `experiments/phm_exp` in the PHM training
environment. The launcher reads the model, failure type, and tool lists from
`phm_exp_config`, and the training settings from `config/ssm_exp_config.yaml`.

The default sweep trains one SQR model per fold for each of `unweighted`,
`0.25`, `0.5`, `1`, `2`, and `4`. The ratio is the total loss weight assigned to
decreasing windows divided by the total assigned to constant windows. For
training-fold counts \(N_c,N_d\), the weights are

\[
w_c=\frac{2}{(1+r)N_c},\qquad
w_d=\frac{2r}{(1+r)N_d}.
\]

The `unweighted` setting assigns every window weight \(2/(N_c+N_d)\). The
`ratio_1` setting reproduces the original inverse-frequency weighting. All
settings have the same total weight of two across the training fold. Window
counts are recomputed for each fold, and the same seed is used for each ratio.

The runner uses the existing cross-validation training and best-model
evaluation pipeline. It trains one model per fold because SQR samples quantile
levels during training. It evaluates the median (\(\tau=0.5\)) prediction on
the same PHM test lives and writes:

- `ablation_results/window_weights/fold_rmse.csv` with overall, constant-region,
  and decreasing-region RMSE for every fold
- `ablation_results/window_weights/summary_rmse.csv` with the mean and standard
  deviation of each RMSE across folds
- `ablation_results/window_weights/overall_rmse_by_weight.png`
- `ablation_results/window_weights/decreasing_rmse_by_weight.png`

Each life contributes one RMSE to the fold mean, matching the existing life
metric. The regional metrics use the saved median predictions and the true RUL:
the constant region is the clipped plateau at 500, and the decreasing region
contains valid target samples below 500. This splits a window that crosses the
plateau boundary by its target values. As in the existing life metric, zero
targets are excluded because saved outputs use zero for padding. A life with
no samples in a region is omitted from that region's mean.

The existing `best_models/`, `outputs/`, and `metrics/` folders also contain
the separate artifacts for each ratio. To select a shorter sweep, pass for
example `--ratios unweighted 1 2` to the launcher.

## Expected results and interpretation

The ratio refers to the **total weight of each group**, not the weight of an
individual window. At `ratio_0.25`, constant windows receive four times the
*combined* weight of decreasing windows. At `ratio_2` and `ratio_4`, decreasing
windows receive two and four times the combined weight of constant windows,
respectively. Because there are usually fewer decreasing windows, an
individual decreasing window can still have a higher weight even when the
group ratio is below one: \(w_d/w_c=rN_c/N_d\). The relevant region is the
**decreasing RUL** region; RUL does not increase there.

We expect larger ratios to make errors in decreasing windows more influential
during training. This may reduce RMSE in that region, especially when moving
from the unweighted baseline toward `ratio_1`. It is **not** guaranteed that
test RMSE improves at every step. Strong emphasis on decreasing windows can
reduce accuracy in constant windows, and gains in the decreasing region may
level off. The best ratio for whole-life RMSE therefore has to be measured.
The decreasing-region RMSE plot directly tests whether a higher ratio helps
where the application needs accurate predictions. The CSV also reports the
constant-region RMSE so we can inspect the tradeoff behind the overall curve.

