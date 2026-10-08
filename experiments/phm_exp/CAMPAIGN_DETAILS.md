# Window-weight ablation campaign on acquario3

Work from `/home/davide_frizzo/ssm-pdm-worktrees/quantile-tau-propagation`.
The executable Bash launcher uses this checkout's `.venv`, falling back to
`~/ssm-pdm/.venv`. Override with `PHM_PYTHON=/path/to/python` or a leading
`--python /path/to/python`. Training requires the initialized Ceruleo submodule,
PHM data, and the project's Python dependencies; monitoring requires tmux.

Inspect the plan without creating artifacts, sessions, or training processes:

```bash
experiments/phm_exp/launch_window_weight_campaign \
  --campaign-id S4D_window_weights_scale --head scale --dry-run
```

Start the campaign when ready:

```bash
experiments/phm_exp/launch_window_weight_campaign \
  --campaign-id S4D_window_weights_scale --head scale --port 8879
```

`--head scale` overrides only the campaign configuration snapshots to enable
QuantileScaleHead with `tau_feat=false` and `tau_mult=false`. `--head feature`
uses QuantileHead with `tau_feat=true`, `tau_mult=false`. The default
`--head configured` preserves the current YAML settings. Existing YAML files
and running campaigns are not modified. All three quantile models propagate
the current model tau to their head on each forward pass.

Defaults are seed 42 and ratios `unweighted 0.25 0.5 1 2 4`; override with
`--seed` and `--ratios`. Experiment and tool/GPU settings come from
`config/ssm_exp_config.yaml` and `phm_exp_config`, or `--config` and
`--shell-config`. Full resolved settings, command, revision, dirty-checkout
indicator, and source hashes are recorded in `manifest.json`. The experiment,
model, and shell configurations are copied before launching. Commit the source
changes if you need an independently reproducible repository revision.

The default artifact root is
`/mnt/disk1/davide_frizzo/experiments/ssm-pdm`; override with `--artifact-root`
or `PHM_ARTIFACT_ROOT`. Each campaign gets a new directory; existing directories
and tmux sessions are rejected. No checkpoint resume is implemented: use a new
campaign ID for a fresh run.

## Execution and evaluation

For each weight setting, train one SQR model for each configured CV fold.
Quantiles are sampled during training; evaluation at 0.1, 0.25, 0.5, 0.75,
and 0.9 uses the same validation-selected checkpoint without retraining.
The default five folds and six settings mean **30 training runs**, not 150.
The same seed is reset for every weight setting; group counts are recomputed
from the training split. Ratio means total decreasing/constant loss weight;
ratio 1 balances the groups, whereas unweighted assigns equal per-window weights.

The checkpoint minimizes the existing validation loss at tau 0.5. Held-out test
metrics do not select checkpoints, folds, or weight settings. After each fold's
quantile predictions have been saved, the runner writes RMSE, pinball loss,
80% coverage, width, and crossing rate. Regional metrics exclude zero padding;
constant means the clipped RUL plateau at 500 and decreasing means RUL below
500. Each life contributes equally to a fold, and summaries report the mean
and sample SD across evaluated folds. Partial summaries label their completed
fold count; one fold has SD zero because variability cannot yet be estimated.
RMSE/pinball/width are in original RUL units; rates are fractions. Read coverage
against 0.8 together with width and crossing, not crossing alone.

## Dashboard, services, and logs

The launcher prints a loopback dashboard URL, artifact location, and attach
command. The session `phm-ablation-CAMPAIGN_ID` contains separate `dashboard`,
`pipeline`, and `logs` windows. Exited panes remain available. Pipeline exit
codes are preserved through logging, and completed, failed, and interrupted
outcomes are recorded. Training completion and completed evaluation counts
are shown separately. A worker heartbeat distinguishes a disconnected/stale
observation from a known terminal outcome. The dashboard refreshes every five
seconds; refreshes only read saved data and do not evaluate models or make plots.

From your local machine, forward the port through your configured SSH alias:

```bash
ssh -N -L 8879:127.0.0.1:8879 acquario3
```

Then visit `http://127.0.0.1:8879`. Add `--tunnel` at launch only if you want an
explicitly public Cloudflare quick tunnel; cloudflared is otherwise unnecessary.
W&B remains disabled by default as in the original ablation. Add `--wandb`
to enable it using your existing credentials; all run links appear by setting
and fold, including the first run. Credentials are never included in the manifest.

```bash
tmux attach -t phm-ablation-S4D_window_weights_scale
# Stop the pipeline only; monitoring stays available:
tmux send-keys -t phm-ablation-S4D_window_weights_scale:pipeline C-c
# Stop the entire session; this also interrupts active training:
tmux kill-session -t phm-ablation-S4D_window_weights_scale
```

Campaign files include `manifest.json`, `status.json`, `events.jsonl`,
`heartbeat.json`, `runner.log`, `dashboard.log`, copied configurations,
`results/` (checkpoints/predictions/original metrics), and `ablation_results/`
(regional CSVs and comparison plots). Optional tunnel diagnostics are in
`cloudflared.log`; its URL is saved in `dashboard_url.txt`. The dashboard serves
only its API, favicon, exported report, and allowed ablation result files.

Restart just monitoring after stopping the original dashboard, without training:

```bash
experiments/phm_exp/launch_window_weight_campaign \
  --dashboard-only /mnt/disk1/davide_frizzo/experiments/ssm-pdm/S4D_window_weights_scale \
  --port 8879
```

This creates a separate `phm-ablation-CAMPAIGN_ID-monitor` session. Use another
port if a dashboard is already running.

## Offline report

Use the visible **Download offline HTML** link. The self-contained
`dashboard_snapshot.html` is persisted in the campaign directory every five
seconds and refreshed after the worker records its terminal outcome. It embeds
configuration, progress, W&B links, all recorded curves, CSV table data, PNGs,
favicon, styles, and local selectors. It can be opened directly after the
server and tmux session have stopped; no server or network is needed to render.
The report displays capture time, last recorded update, and partial/final state.
Regenerate it from saved artifacts without training or evaluation:

```bash
~/ssm-pdm/.venv/bin/python experiments/phm_exp/ablation_dashboard.py \
  --directory /mnt/disk1/davide_frizzo/experiments/ssm-pdm/S4D_window_weights_scale \
  --export /mnt/disk1/davide_frizzo/experiments/ssm-pdm/S4D_window_weights_scale/dashboard_snapshot.html
```

## Verification

```bash
~/ssm-pdm/.venv/bin/python -m unittest discover -s tests -p test_tau_propagation.py -v
~/ssm-pdm/.venv/bin/python -m unittest discover -s tests -p test_ablation_campaign.py -v
```

The tests use synthetic metrics and cheap workers; creating the dashboard does
not start a full experimental campaign.
