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
levels during training. It evaluates the same selected checkpoint at quantiles
`0.1`, `0.25`, `0.5`,
`0.75`, and `0.9` on the same PHM test lives. These are inference passes,
not additional training runs. Median RMSE outputs are:

- `ablation_results/window_weights/fold_rmse.csv` with overall, constant-region,
  and decreasing-region RMSE for every fold
- `ablation_results/window_weights/summary_rmse.csv` with the mean and standard
  deviation of each RMSE across folds
- `ablation_results/window_weights/overall_rmse_by_weight.png`
- `ablation_results/window_weights/decreasing_rmse_by_weight.png`

The uncertainty evaluation also writes:

- `fold_pinball.csv` and `summary_pinball.csv`: unweighted pinball loss for
  every quantile, overall and in each RUL region
- `fold_intervals.csv` and `summary_intervals.csv`: coverage, mean interval
  width, and quantile crossing rate for the nominal 80% interval formed by
  quantiles `0.1` and `0.9`, overall and in each RUL region
- `{region}_pinball_quantile_{tau}_by_weight.png` for each region and quantile
- `{region}_{coverage,width,crossing_rate}_80_by_weight.png` for each region;
  coverage plots include a reference line at `0.8`

All these files are placed in `ablation_results/window_weights/`. Summaries
contain the mean and sample standard deviation across folds. The runner
continues to disable W&B logging and saves quantile predictions locally under
`outputs/.../fold_{fold}/quantile_{tau}/`.

Pinball loss uses the requested quantile's asymmetric penalty with no training
window weights. Coverage counts targets satisfying `q_0.1 <= target <= q_0.9`,
including endpoints; width is `q_0.9 - q_0.1` in original RUL units. Coverage
and crossing rates are fractions, not percentages. Bounds are not sorted or
repaired: crossed intervals have zero coverage and signed negative width,
and the crossing rate reports how frequently this occurs. Inspect crossings
before interpreting width as sharpness.

Each life contributes one metric value to the fold mean, matching the existing
life RMSE convention. Metrics first average over valid samples within a life;
coverage therefore averages life-level coverage, rather than pooling all
samples from lives of different lengths. A life without samples in a region
is omitted from that region's mean. If an entire fold lacks a region, its CSV
values are blank and its plot values are omitted.

The regional metrics use the saved quantile predictions and the true RUL:
the constant region is the clipped plateau at 500, and the decreasing region
contains valid target samples below 500. This splits a window that crosses the
plateau boundary by its target values. As in the existing life metric, zero
targets are excluded because saved outputs use zero for padding.

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

The uncertainty metrics test whether changing the weighting affects more than
median accuracy. Compare unweighted pinball loss at each quantile and read
interval coverage together with width and crossing rate. Improved RMSE alone
does not establish improved uncertainty estimation. Regional coverage can
reveal shortcomings hidden by overall coverage; the clipped constant-RUL
plateau also contains tied targets, so coverage need not equal the nominal
level exactly even for correct quantiles.
