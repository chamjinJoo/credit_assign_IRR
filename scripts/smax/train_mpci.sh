#!/bin/bash
# ============================================================
# Draft recurrent JAX PCI launcher for SMAX
# ============================================================

set -euo pipefail

config="configs/smax_configs/smax_pci_mappo.yaml"

# rpci settings
causal_lambda=1.0
model="jax-pci-mappo"

# env settings
map_name="6h_vs_8z"  # empty string keeps the YAML value; set to override the SMAX map
# maps: [3m, 2s3z, 3s_vs_5z, 3s5z_vs_3s6z, 5m_vs_6m, 3s5z, 6h_vs_8z, 10m_vs_11m, smacv2_5_units, smacv2_10_units]
allowed_target_window=(6)

# training settings
seeds=(42)
lr_actor=0.002
lr_critic=0.002

device="cuda"
n_rollout_threads=8
run_in_background=false

# reproducibility settings for JAX
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# export KMP_DUPLICATE_LIB_OK=TRUE # for Window

# main training loop over config variants and seeds
for seed in "${seeds[@]}"; do
    echo "[seed=${seed}] start]"
    python main.py --env SMAX \
      --config ${config} \
      --model ${model} \
      --map-name "${map_name}" \
      --causal-lambda ${causal_lambda} \
      --lr-actor ${lr_actor} \
      --lr-critic ${lr_critic} \
      --allowed-target-window ${allowed_target_window} \
      --n-rollout-threads ${n_rollout_threads} \
      --device ${device} \
      --seed ${seed}
done