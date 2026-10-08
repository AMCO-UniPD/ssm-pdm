Run a monitored campaign from the repository root:

```bash
experiments/phm_exp/launch_phm_campaign_tmux --campaign-id S4D_my_campaign
```

The launcher uses the project `.venv`, falling back to `~/ssm-pdm/.venv` for
worktrees. Set `PHM_PYTHON` to use another project environment. `tmux`,
`cloudflared`, an initialized `src/ceruleo` submodule, the PHM data, and W&B
credentials must be available. Use `--dry-run` to inspect the configuration
without creating files or processes. Optional arguments include `--port`,
`--artifact-root`, `--exp-config`, and `--shell-config`.

Each campaign receives a unique experiment name and snapshots its experiment,
model, and shell configurations. Existing campaign directories are rejected to
prevent results from being overwritten. The tmux session has `dashboard`,
`tunnel`, and `pipeline` windows. The launcher prints the public dashboard URL,
session name, and artifact directory. Attach using `tmux attach -t phm-CAMPAIGN`.
Stopping that session stops the campaign and dashboard.

`run_phm_exp` selects the CV launcher when CV is configured, then runs training,
output grouping, quantile metric grouping, business metrics, and final plots.
Any stage failure stops the pipeline and is reported on the dashboard. The
existing configured training procedure is preserved: one fresh model per
quantile, per fold. Five folds with five quantiles means 25 training runs.
Campaigns are grouped in W&B by their campaign ID.

The dashboard refreshes every five seconds. It shows configuration, links to
the W&B project and first run, live epoch losses and logs, and a test-life
quantile metrics table after each fold. The metric is the configured
`life_eval_loss` (currently RMSE), rather than necessarily pinball loss. An
average table appears after all folds complete.

The best fold minimizes the mean of the saved checkpoints' minimum validation
losses across quantiles. Test metrics do not influence selection. The final
plots use that fold's held-out predictions: full and trailing-sample quantile
plots, unexpected breaks, unexploited lifetime, and business cost J. PNGs appear
on the dashboard; PDFs are also saved in the campaign `plots` directory.

Telemetry is inactive unless `PHM_MONITOR_DIR` is set. Campaign artifacts under
`experiments/phm_exp/campaigns/` are ignored by Git. The dashboard only serves
telemetry and plot assets and binds to localhost; Cloudflare provides its
public URL.

The project supplies default scikit-learn estimator tags for Ceruleo's legacy
`Transformer`, which allows scikit-learn 1.8 to validate both new and cached
transformers. No package downgrade or submodule edits are required.

Validation:

```bash
PHM_PYTHON=/path/to/project/python
"$PHM_PYTHON" -m unittest discover -s tests -p test_phm_campaign_monitor.py -v
```
