#!/usr/bin/env bash
# Usage: bash launch_compact_plot_business_metrics.sh [model] [approach] [extra args...]
# Example: ... S4D windowed_quantile_weighted --cv --figure_height 2.8
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
source "$script_dir/phm_exp_config"

model_name="${1:-S4D}"
plot_approach="${2:-windowed_quantile_weighted}"
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then shift; fi

exp_name="${model_name}_${failure_type}_${plot_approach}_cv_phm_in_features"

"${PYTHON:-python}" "$script_dir/compact_plot_business_metrics.py" \
    --comparison quantiles \
    --model_names "$model_name" \
    --model_name "$model_name" \
    --exp_names "$exp_name" \
    --exp_config_path "config/ssm_exp_config.yaml" \
    --data_name "$data_name" \
    --failure_type "$failure_type" \
    --plot_approach "$plot_approach" \
    --train_phm_tools "${train_phm_tools[@]}" \
    --test_phm_tools "${test_phm_tools[@]}" \
    --log_scale \
    "$@"
