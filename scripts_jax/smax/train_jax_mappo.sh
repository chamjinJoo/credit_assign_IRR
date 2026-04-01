#!/bin/bash
# ============================================================
# Canonical JAX MAPPO launcher for SMAX
# ============================================================

set -euo pipefail

config="configs_jax/smax_mappo.yaml"
model="jax-mappo"
use_recurrent_policy=true # true: RNN-agent, false: FF-agent

seed=42
n_rollout_threads=4
max_iterations=1600
max_steps_per_iteration=6400

log_interval=2
device="cpu"  # recommended to use cuda if available, but default to cpu for safety
seed=42

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate pcimarl
fi

export CUBLAS_WORKSPACE_CONFIG=:4096:8

python main.py \
  --env SMAX \
  --config ${config} \
  --model ${model} \
  --use-recurrent-policy ${use_recurrent_policy} \
  --n-rollout-threads ${n_rollout_threads} \
  --max-iterations ${max_iterations} \
  --max-steps ${max_steps_per_iteration} \
  --log-interval ${log_interval} \
  --device ${device} \
  --seed ${seed} \
  --use-wandb \
  --wandb-project pci-marl-smax-jax
