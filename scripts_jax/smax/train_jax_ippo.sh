#!/bin/bash
# ============================================================
# Canonical JAX MAPPO launcher for SMAX
# ============================================================

set -euo pipefail

config="configs_jax/smax_ippo.yaml"
model="jax-ippo"
use_recurrent_policy=true # true: RNN-agent, false: FF-agent

# env settings
map_name="2s3z"  # empty string keeps the YAML value; set to override the SMAX map
allowed_target_windows=(5)

# training settings
seeds=(42 1 2)
device="cpu"  # recommended to use cuda if available, but default to cpu for safety
n_rollout_threads=4
max_iterations=1600
max_steps_per_iteration=6400
log_interval=2
run_in_background=true

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate pcimarl
fi

export CUBLAS_WORKSPACE_CONFIG=:4096:8
# export KMP_DUPLICATE_LIB_OK=TRUE # for Window

pids=()

# main training loop over config variants and seeds
for allowed_target_window in "${allowed_target_windows[@]}"; do
  echo "[allowed_target_window=${allowed_target_window}] start"
  for seed in "${seeds[@]}"; do
      echo "[allowed_target_window=${allowed_target_window} seed=${seed}] start"
      if ${run_in_background}; then
        python main.py \
        --env SMAX \
        --config ${config} \
        --model ${model} \
        --map-name "${map_name}" \
        --use-recurrent-policy ${use_recurrent_policy} \
        --allowed-target-window ${allowed_target_window} \
        --n-rollout-threads ${n_rollout_threads} \
          --max-iterations ${max_iterations} \
          --max-steps ${max_steps_per_iteration} \
          --log-interval ${log_interval} \
          --device ${device} \
          --seed ${seed} \
          --use-wandb \
          --wandb-project pci-marl-smax-jax-v2 &
        pid=$!
        pids+=("${pid}")
        echo "[allowed_target_window=${allowed_target_window} seed=${seed}] PID: ${pid}"
      else
        python main.py \
        --env SMAX \
        --config ${config} \
        --model ${model} \
        --map-name "${map_name}" \
        --use-recurrent-policy ${use_recurrent_policy} \
        --allowed-target-window ${allowed_target_window} \
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
done

if ${run_in_background}; then
  echo "all seed launched. waiting for completion..."
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
  echo "all config/seed training completed!"
else
  echo "all config/seed training completed sequentially!"
fi