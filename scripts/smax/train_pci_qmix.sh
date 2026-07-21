#!/bin/bash
# ============================================================
# Canonical JAX PCI-QMIX launcher for SMAX
# ============================================================

set -euo pipefail

config="configs/smax_configs/smax_pci_qmix.yaml"
model="jax-pci-qmix"

# env settings
map_name="smacv2_10_units"
allowed_target_windows=(10)

# QMIX settings
lr=5.0e-5

# Outcome bridge settings
causal_lambda=1.0
agent_delta_td_coef=0.1

# training settings
seeds=(42 1 2 31 32)
device="cuda"
n_rollout_threads=16
run_name_postfix=""

export CUBLAS_WORKSPACE_CONFIG=:4096:8

for allowed_target_window in "${allowed_target_windows[@]}"; do
    echo "[allowed_target_window=${allowed_target_window}] start"
    for seed in "${seeds[@]}"; do
        wandb_group="${map_name:-default}"
        wandb_args=(
            --use-wandb
            --wandb-group "${wandb_group}"
            --wandb-run-name-postfix "${run_name_postfix}"
        )
        echo "[allowed_target_window=${allowed_target_window} seed=${seed}] start"

        python main.py \
            --env SMAX \
            --config ${config} \
            --model ${model} \
            --map-name "${map_name}" \
            --allowed-target-window ${allowed_target_window} \
            --lr ${lr} \
            --causal-lambda ${causal_lambda} \
            --agent-delta-td-coef ${agent_delta_td_coef} \
            --n-rollout-threads ${n_rollout_threads} \
            --device ${device} \
            --seed ${seed} \
            "${wandb_args[@]}"
    done
done
