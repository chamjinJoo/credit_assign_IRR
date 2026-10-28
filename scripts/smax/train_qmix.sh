#!/bin/bash
# ============================================================
# JAX QMIX launcher for SMAX
# ============================================================

set -euo pipefail

config="configs/smax_configs/smax_qmix.yaml"
model="jax-qmix"

# env settings
map_name="smacv2_20_units"
allowed_target_windows=(20)

# QMIX settings
lr=5.0e-5
qmix_update_epochs=8

# training settings
seeds=(42 1 2 31 32)
device="cuda"
n_rollout_threads=16

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
        --allowed-target-window ${allowed_target_window} \
        --lr ${lr} \
        --qmix-update-epochs ${qmix_update_epochs} \
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
