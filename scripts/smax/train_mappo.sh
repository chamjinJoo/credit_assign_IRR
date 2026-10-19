#!/bin/bash
# ============================================================
# JAX MAPPO launcher for SMAX
# ============================================================

set -euo pipefail

config="configs/smax_configs/smax_mappo.yaml"
model="jax-mappo"
use_recurrent_policy=true # true: RNN-agent, false: FF-agent

# env settings
map_name="smacv2_10_units"  # empty string keeps the YAML value; set to override the SMAX map
allowed_target_windows=(8)
# training settings
lr_actor=0.002
lr_critic=0.002
seeds=(42 1 2 31 32)

device="cuda"  # recommended to use cuda if available, but default to cpu for safety
n_rollout_threads=8

# reproducibility settings for JAX
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# main training loop over config variants and seeds
for allowed_target_window in "${allowed_target_windows[@]}"; do
    echo "[allowed_target_window=${allowed_target_window}] start"
    for seed in "${seeds[@]}"; do
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
            --use-wandb
    done
done