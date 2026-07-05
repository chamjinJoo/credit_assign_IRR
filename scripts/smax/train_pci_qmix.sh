#!/bin/bash
# ============================================================
# Canonical JAX PCI-QMIX launcher for SMAX
# ============================================================

set -euo pipefail

config="configs/smax_configs/smax_pci_qmix.yaml"
model="jax-pci-qmix"

# env settings
map_name="2s3z"
allowed_target_windows=(3 4 5)

# QMIX settings
lr=5.0e-5
qmix_update_epochs=8

# Outcome bridge settings
lr_bridge=2.0e-3
causal_lambda=1.0

# training settings
seeds=(42 1 2 31 32)
device="cuda"
n_rollout_threads=16
run_name_postfix="lr5e-5"

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
            --n-rollout-threads ${n_rollout_threads} \
            --device ${device} \
            --seed ${seed} \
            "${wandb_args[@]}"
    done
done
