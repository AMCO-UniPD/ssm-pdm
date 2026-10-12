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

Resume a stopped or crashed campaign by appending `--resume` to the original
launch command, keeping its explicit `--campaign-id` and artifact root:

```bash
experiments/phm_exp/launch_phm_campaign_tmux --campaign-id S4D_my_campaign --resume
```

Resume uses the saved configuration and preserves logs/results. It skips
completed folds and quantile runs with readable saved artifacts, including
historical campaigns with completed-fold or final-epoch telemetry. An unfinished
training run restarts from its beginning; exact epoch/optimizer recovery is not
available. Successful downstream stages recorded by this version are retained;
historical stages without completion markers are rerun. Add `--dry-run` to inspect
without launching. An exited tmux pipeline is reused, an active one is rejected,
and a worker lock excludes duplicate runners. See `CAMPAIGN_DETAILS.md` for both
campaign launchers and older artifact-root examples.

The dashboard refreshes every five seconds. It shows configuration, links to
the W&B project and first run, live epoch losses and logs, and a test-life
quantile metrics table after each fold. The metric is the configured
`life_eval_loss` (currently RMSE), rather than necessarily pinball loss. An
average table appears after all folds complete.

Final reports resolve predictions using the resumed `PHM_RESULTS_DIR`, saved
results paths relocated within a moved campaign, or its local `results/`
directory for older manifests. They do not fall back to the checkout's legacy
outputs when a migrated campaign has saved results. See `CAMPAIGN_DETAILS.md`
for a plot-only regeneration command.

The best fold minimizes the mean of the saved checkpoints' minimum validation
losses across quantiles. Test metrics do not influence selection. The final
plots use that fold's held-out predictions: full, last-1000-sample, and configured trailing-sample quantile
plots, unexpected breaks, unexploited lifetime, and business cost J. PNGs appear
on the dashboard; PDFs are also saved in the campaign `plots` directory.

All new campaign artifacts default to
`/mnt/disk1/davide_frizzo/experiments/ssm-pdm/<campaign-id>/`. Set
`PHM_ARTIFACT_ROOT` or pass `--artifact-root` to choose a different root.
The launcher sets `PHM_MONITOR_DIR` to that campaign directory and
`PHM_RESULTS_DIR` to its `results/` subdirectory. Checkpoints, predictions,
metric tables, and business metrics live under `results/`; final figures live
under `plots/`. `WANDB_DIR` directs local W&B files to the campaign directory's
`wandb/` folder. Configuration snapshots, status/events, and service logs live
at the campaign root. Existing campaigns retain their recorded locations.

**Download offline HTML** saves a self-contained snapshot of the current
dashboard. Telemetry, logs, plot images, and the favicon are embedded; it does
not poll a server or load remote assets when opened locally. Download after
completion for the final results, or export later from the saved campaign:

```bash
python3 experiments/phm_exp/phm_dashboard.py \
  --directory /path/to/campaign --export dashboard.html
```

Telemetry is inactive unless `PHM_MONITOR_DIR` is set. Direct training scripts
use their legacy output paths unless `PHM_RESULTS_DIR` is set. Campaign artifacts
under the old `experiments/phm_exp/campaigns/` path are ignored by Git.
The dashboard only serves
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


For an already completed campaign, the same `--resume` command refreshes plots
and the persisted offline dashboard from saved predictions without entering
training. This also works after killing its tmux session: monitoring is recreated.
