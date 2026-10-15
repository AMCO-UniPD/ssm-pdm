The compact exporter creates one PDF and one 300 dpi PNG containing UB, UL and
J. It uses the same Ceruleo metric functions as the existing exporters, including
their uncertainty bands, log scales, colours and line styles. Existing scripts
and exported figures remain available.

Run from any directory (with the project Python environment active):

```bash
# Figure 6: five quantiles of S4D
bash experiments/phm_exp/launch_compact_plot_business_metrics.sh S4D windowed_quantile_weighted

# Figure 3: models and preventive baselines
bash experiments/phm_exp/launch_multi_compact_plot_business_metrics.sh windowed_quantile_weighted
```

The launchers use the same model/experiment defaults as the original quantile
launchers. Append experiment options such as `--cv`, `--add_shade`, `--life_idx`,
`--quantile_run` or `--exp_names` to match the original figure's inputs. As in the
original scripts, quantile predictions come from run/fold 1 (5 for
RULTransformer); standard predictions use `--plot_run_id`. Baselines come from
`baseline_outputs`. `PHM_RESULTS_DIR` can select an existing results directory.
Set `PYTHON` to a Python executable if needed.

For standard predictions, call `compact_plot_business_metrics.py` directly with
`--comparison standard` and the arguments used for `plot_business_metrics.py`.
The other modes are `--comparison quantiles` and `--comparison models`.

Outputs go under `compact_business_metrics_plots/<comparison>/[model]/<failure>/<approach>`
in the results directory. The default width is 7.2 inches, with 8 pt text. Height
is 2.0 inches for up to five curves and 2.25 inches for larger comparisons. A
single legend uses one row for five curves or two rows for the default 13 models.
Adjust `--figure_width`, `--figure_height`, `--font_size`, and `--legend_columns`
after inspecting the plots with the real data at their printed size.

In LaTeX, include the combined PDF once at full text width and retain the overall
caption. The `(a)`, `(b)` and `(c)` headings are already in the PDF. Measure the
net height saving in the paper; these dimensions are trial defaults.
