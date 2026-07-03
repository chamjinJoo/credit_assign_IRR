#!/bin/bash
# ============================================================
# Canonical JAX PCI-QMIX launcher for SMAX
# ============================================================

set -euo pipefail

config="configs/smax_configs/smax_pci_qmix.yaml"
model="jax-pci-qmix"

# env settings
map_name="3m"
allowed_target_windows=(2)

# QMIX settings
lr=1.0e-3
qmix_update_epochs=8

# Outcome bridge settings
lr_bridge=2.0e-3
causal_lambda=1.0
bridge_epoch=2
causal_mini_batch_size=3200
clip_delta=0.0
agent_delta_td_coef=1.0

# training settings
seeds=(42 1 2 31 32)
device="cuda"
n_rollout_threads=16
run_name_postfix="lr1e-3"

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
            --lr-bridge ${lr_bridge} \
            --qmix-update-epochs ${qmix_update_epochs} \
            --causal-lambda ${causal_lambda} \
            --bridge-epoch ${bridge_epoch} \
            --causal-mini-batch-size ${causal_mini_batch_size} \
            --clip-delta ${clip_delta} \
            --agent-delta-td-coef ${agent_delta_td_coef} \
            --n-rollout-threads ${n_rollout_threads} \
            --device ${device} \
            --seed ${seed} \
            "${wandb_args[@]}"
    done
done
