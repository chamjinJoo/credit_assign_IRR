#!/bin/bash
# ============================================================
# Draft recurrent JAX PCI launcher for SMAX
# ============================================================

set -euo pipefail

config="configs_jax/smax_qpci.yaml"
model="jax-qpci"
use_centralized_critic=true # false: IPPO-critic / true: MAPPO-critic
causal_lambda=1.0
clip_delta=0.0

n_rollout_threads=4
max_iterations=1600
max_steps_per_iteration=6400
log_interval=2
device="cpu"
seed=42

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate pcimarl
fi

export CUBLAS_WORKSPACE_CONFIG=:4096:8
# export KMP_DUPLICATE_LIB_OK=TRUE # for Window

python main.py \
  --env SMAX \
  --config ${config} \
  --model ${model} \
  --use-centralized-critic ${use_centralized_critic} \
  --causal-lambda ${causal_lambda} \
  --n-rollout-threads ${n_rollout_threads} \
  --max-iterations ${max_iterations} \
  --max-steps ${max_steps_per_iteration} \
  --log-interval ${log_interval} \
  --device ${device} \
  --seed ${seed} \
  --use-wandb \
  --wandb-project pci-marl-smax-jax
