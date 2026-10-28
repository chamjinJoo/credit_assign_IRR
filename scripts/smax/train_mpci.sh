#!/bin/bash
# ============================================================
# JAX PCI-MAPPO launcher for SMAX
# ============================================================

set -euo pipefail

config="configs/smax_configs/smax_pci_mappo.yaml"

# rpci settings
causal_lambda=1.0
model="jax-pci-mappo"

# env settings
map_name="3m"  # empty string keeps the YAML value; set to override the SMAX map
# maps: [3m, 2s3z, 3s_vs_5z, 3s5z_vs_3s6z, 5m_vs_6m, 3s5z, 6h_vs_8z, 10m_vs_11m, smacv2_5_units, smacv2_10_units]
allowed_target_windows=(2)

# training settings
seeds=(42 1 2 31 32)
lr_actor=0.0005
lr_critic=0.0005

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
