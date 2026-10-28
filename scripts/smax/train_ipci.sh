#!/bin/bash
# ============================================================
# JAX PCI-IPPO launcher for SMAX
# ============================================================

set -euo pipefail

config="configs/smax_configs/smax_pci_ippo.yaml"

# rpci settings
causal_lambda=0.1
model="jax-pci-ippo"

# env settings
map_name="3s5z"  # empty string keeps the YAML value; set to override the SMAX map
allowed_target_windows=(6)

# training settings
seeds=(42 1 2)
lr_actor=0.004
lr_critic=0.004

device="cuda"
n_rollout_threads=8

# reproducibility settings for JAX
export CUBLAS_WORKSPACE_CONFIG=:4096:8

run_training() {
    local allowed_target_window="$1"
    local seed="$2"
    echo "[allowed_target_window=${allowed_target_window} seed=${seed}] start"
    python main.py \
        --env SMAX \
        --config ${config} \
        --model ${model} \
        --map-name "${map_name}" \
        --causal-lambda ${causal_lambda} \
        --allowed-target-window ${allowed_target_window} \
        --lr-actor ${lr_actor} \
        --lr-critic ${lr_critic} \
        --n-rollout-threads ${n_rollout_threads} \
        --device ${device} \
        --seed ${seed} \
        --use-wandb
}

for allowed_target_window in "${allowed_target_windows[@]}"; do
    for seed in "${seeds[@]}"; do
        run_training "${allowed_target_window}" "${seed}"
    done
done
