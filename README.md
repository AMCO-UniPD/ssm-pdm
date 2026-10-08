# Remaining Useful Life Estimation through State Space Models

This repository contains the codebase for using State Space Models in the
context of Predictive Maintenance and in particular to tackle the Remaining
Useful Life (RUL) estimation task.

## Publications

This codebase was used for the production of two research articles:

- [*A Quantile Regression Approach for Remaining Useful Life Estimation with State Space Models*](https://www.sciencedirect.com/science/article/pii/S2405896325027351) → published at J3C 2025 IFAC Joint Conference on Computers, Cognition and
Communication
- *Uncertainty and Business-Aware Remaining Useful Life Estimation for Semiconductor Manufacturing* → Submitted to IEEE Transactions on Semiconductor Manufacturing (currently under submission on arxiv)

## Launch a PHM experimental campaign

From the repository root, launch training and its live dashboard in a persistent
tmux session:

```bash
./experiments/phm_exp/launch_phm_campaign_tmux \
  --campaign-id "S4D_$(date +%Y%m%d_%H%M%S)" \
  --port 8879
```

Choose an unused port for each concurrent dashboard. The default is `8878`;
an existing dashboard keeps its port occupied after training finishes. Campaign
IDs must be unique and contain only letters, numbers, underscores, and hyphens.
The launcher prints the public Cloudflare dashboard URL, tmux attach command,
and campaign artifact directory. Training continues when you detach from tmux
or close your terminal.

Configure the experiment before launching:

| File | Settings |
| --- | --- |
| `experiments/phm_exp/phm_exp_config` | Model name, GPU index, failure type, train/test tools, and W&B project |
| `experiments/phm_exp/config/ssm_exp_config.yaml` | Folds, quantiles, epochs, optimizer settings, preprocessing, and plotting |
| `experiments/phm_exp/config/ssm_config.yaml` | Model architecture and quantile head settings |

Campaigns require quantile regression and a complete CV run: `cv: true`,
`start_fold_id: 0`, and `stop_fold_id` equal to `n_folds`. Each quantile is trained
independently within each fold, so five folds and five quantiles produce
**25 W&B runs**. The launcher snapshots the configuration for each campaign.

The project environment, PHM dataset, initialized `src/ceruleo` submodule,
W&B credentials, `tmux`, and `cloudflared` must be available. Use W&B SDK
`0.24.1` or newer; version `0.24.0` has an upload bug. The launcher uses
`.venv/bin/python`, falling back to `~/ssm-pdm/.venv/bin/python` for worktrees.
To select another environment:

```bash
PHM_PYTHON=/path/to/environment/bin/python \
  ./experiments/phm_exp/launch_phm_campaign_tmux \
  --campaign-id my_campaign --port 8880
```

Inspect a campaign without starting training, a dashboard, or a tunnel:

```bash
./experiments/phm_exp/launch_phm_campaign_tmux --dry-run
```

Additional options are `--exp-config`, `--shell-config`, and `--artifact-root`;
run the launcher with `--help` for details. Artifacts default to
`experiments/phm_exp/campaigns/<campaign-id>/`.

The dashboard displays the effective configuration, current/total W&B runs,
W&B links, epoch losses, live logs, and quantile metric tables after each fold.
After all folds, it displays the average table and quantile/business plots for
the fold with the lowest mean checkpoint validation loss across quantiles.

To attach or stop a campaign, substitute its ID below:

```bash
tmux attach -t phm-my_campaign
tmux kill-session -t phm-my_campaign
```

Detach with **Ctrl-b**, then **d**. Killing the session stops training, the
dashboard, and the tunnel. See [the monitoring guide](experiments/phm_exp/CAMPAIGN_MONITOR.md)
for more details.
