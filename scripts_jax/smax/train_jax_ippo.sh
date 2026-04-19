#!/bin/bash
# ============================================================
# Canonical JAX MAPPO launcher for SMAX
# ============================================================

set -euo pipefail

config="configs_jax/smax_ippo.yaml"
model="jax-ippo"
use_recurrent_policy=true # true: RNN-agent, false: FF-agent

# training settings
n_rollout_threads=4
max_iterations=1600
max_steps_per_iteration=6400
log_interval=2
device="cpu"  # recommended to use cuda if available, but default to cpu for safety
seeds=(42)
run_in_background=true

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate pcimarl
fi

export CUBLAS_WORKSPACE_CONFIG=:4096:8
# export KMP_DUPLICATE_LIB_OK=TRUE # for Window

pids=()

# main training loop over seeds
for seed in "${seeds[@]}"; do
    echo "[seed=${seed}] start"
    if ${run_in_background}; then
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
        --wandb-project pci-marl-smax-jax &
      pid=$!
      pids+=("${pid}")
      echo "[seed=${seed}] PID: ${pid}"
    else
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
        --wandb-project pci-marl-smax-jax-v2
    fi
done

if ${run_in_background}; then
  echo "all seed launched. waiting for completion..."
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
  echo "all seed training completed!"
else
  echo "all seed training completed sequentially!"
fi