#!/bin/bash
# ============================================================
# Canonical JAX MAPPO launcher for SMAX
# ============================================================

set -euo pipefail

config="configs/smax_configs/smax_ippo.yaml"
model="jax-ippo"
use_recurrent_policy=true # true: RNN-agent, false: FF-agent

# env settings
map_name="smacv2_20_units"  # empty string keeps the YAML value; set to override the SMAX map
allowed_target_windows=(20)
# training settings
lr_actor=0.0005
lr_critic=0.0005
seeds=(42 1 2 31 32)

device="cuda"  # recommended to use cuda if available, but default to cpu for safety
n_rollout_threads=8
run_in_background=false
run_name_postfix="lr5e-4"

# export KMP_DUPLICATE_LIB_OK=TRUE # for Window
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# main training loop over config variants and seeds
for allowed_target_window in "${allowed_target_windows[@]}"; do
    echo "[allowed_target_window=${allowed_target_window}] start"
    for seed in "${seeds[@]}"; do
        wandb_group="${map_name:-default}"
        wandb_args=(
            --use-wandb
            --wandb-group "${wandb_group}"
            --wandb-run-name-postfix "${run_name_postfix}"
        )
        echo "[seed=${seed}] start]"
        python main.py --env SMAX \
            --config ${config} \
            --model ${model} \
            --map-name "${map_name}" \
            --use-recurrent-policy ${use_recurrent_policy} \
            --allowed-target-window ${allowed_target_window} \
            --lr-actor ${lr_actor} \
            --lr-critic ${lr_critic} \
            --n-rollout-threads ${n_rollout_threads} \
            --device ${device} \
            --seed ${seed} \
            "${wandb_args[@]}"
    done
done