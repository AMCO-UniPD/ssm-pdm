#!/usr/bin/env bash
# Usage: bash launch_multi_compact_plot_business_metrics.sh [approach] [extra args...]
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
source "$script_dir/phm_exp_config"

plot_approach="${1:-windowed_quantile_weighted}"
if (( $# > 0 )); then shift; fi
model_names=(S4D S5 S4 MLP Linear LSTM GRU RNN RULTransformer mean median quantile_0.1 quantile_0.9)
exp_names=()
for selected_model in "${model_names[@]}"; do
    exp_names+=("${selected_model}_${failure_type}_${plot_approach}_cv_phm_in_features")
done

"${PYTHON:-python}" "$script_dir/compact_plot_business_metrics.py" \
    --comparison models \
    --model_names "${model_names[@]}" \
    --model_name "$model_name" \
    --exp_names "${exp_names[@]}" \
    --exp_config_path "config/ssm_exp_config.yaml" \
    --data_name "$data_name" \
    --failure_type "$failure_type" \
    --plot_approach "$plot_approach" \
    --train_phm_tools "${train_phm_tools[@]}" \
    --test_phm_tools "${test_phm_tools[@]}" \
    --log_scale \
    "$@"
