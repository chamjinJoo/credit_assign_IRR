#!/bin/bash
# ============================================================
# Draft recurrent JAX PCI launcher for SMAX
# ============================================================

set -euo pipefail

config="configs/smax_configs/smax_pci_ippo.yaml"

# rpci settings
causal_lambda=0.1
train_naivepredictor=false
use_naive_predictor_for_delta=false
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
run_in_background=false
run_name_postfix="42/1/2_3090"

# reproducibility settings for JAX
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# export KMP_DUPLICATE_LIB_OK=TRUE # for Window

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
        echo "[allowed_target_window=${allowed_target_window} seed=${seed}] start"
        
        python main.py \
            --env SMAX \
            --config ${config} \
            --model ${model} \
            --map-name "${map_name}" \
            --causal-lambda ${causal_lambda} \
            --train-naivepredictor ${train_naivepredictor} \
            --use-naive-predictor-for-delta ${use_naive_predictor_for_delta} \
            --allowed-target-window ${allowed_target_window} \
            --lr-actor ${lr_actor} \
            --lr-critic ${lr_critic} \
            --n-rollout-threads ${n_rollout_threads} \
            --device ${device} \
            --seed ${seed} \
            "${wandb_args[@]}"
    done
done
