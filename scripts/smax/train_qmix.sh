#!/bin/bash
# ============================================================
# Canonical JAX QMIX launcher for SMAX
# ============================================================

set -euo pipefail

config="configs/smax_configs/smax_ippo.yaml"
model="jax-qmix"

# env settings
map_name="3m"
allowed_target_windows=(2 3)

# QMIX settings
lr=0.005
target_update_interval=200
epsilon_start=1.0
epsilon_finish=0.05
epsilon_anneal_time=200000
replay_buffer_size=5000
train_batch_size=32

# training settings
seeds=(42 1 2 31 32)
device="cuda"
n_rollout_threads=8
run_name_postfix="lr5e-3"

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
            --target-update-interval ${target_update_interval} \
            --epsilon-start ${epsilon_start} \
            --epsilon-finish ${epsilon_finish} \
            --epsilon-anneal-time ${epsilon_anneal_time} \
            --replay-buffer-size ${replay_buffer_size} \
            --train-batch-size ${train_batch_size} \
            --n-rollout-threads ${n_rollout_threads} \
            --device ${device} \
            --seed ${seed} \
            "${wandb_args[@]}"
    done
done
