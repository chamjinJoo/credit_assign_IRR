#!/bin/bash
# ============================================================
# Canonical JAX MAPPO launcher for SMAX
# ============================================================

set -euo pipefail

config="configs_jax/smax_mappo.yaml"
model="jax-mappo"
use_recurrent_policy=true # true: RNN-agent, false: FF-agent

# env settings
map_name="smacv2_10_units"  # empty string keeps the YAML value; set to override the SMAX map
allowed_target_windows=(6 8 10)

# training settings
seeds=(42 1 2)
device="cpu"  # recommended to use cuda if available, but default to cpu for safety
n_rollout_threads=4
max_iterations=1600
max_steps_per_iteration=6400
log_interval=2
run_in_background=false
anneal_lr=true
wandb_run_name_postfix=""
# test=false
# checkpoint="./save_checkpoints/20260406_185738_seed42/smaxbatchwrapper/checkpoints/jax_checkpoint_iter_1599.pkl"
# save_replay=true
# replay_dir="./smax_replays"
# replay_episodes=10

# conda activation (optional)
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate pcimarl
fi

# reproducibility settings for JAX
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# export KMP_DUPLICATE_LIB_OK=TRUE # for Window

pids=()

# main training loop over config variants and seeds
for allowed_target_window in "${allowed_target_windows[@]}"; do
  echo "[allowed_target_window=${allowed_target_window}] start"
  for seed in "${seeds[@]}"; do
      wandb_group="${map_name:-default}"
      wandb_args=(
        --use-wandb
        --wandb-project pci-marl-smax-jax-v3
        --wandb-group "${wandb_group}"
      )
      if [[ -n "${wandb_run_name_postfix}" ]]; then
        wandb_args+=(--wandb-run-name-postfix "${wandb_run_name_postfix}")
      fi
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
          --anneal-lr ${anneal_lr} \
          "${wandb_args[@]}" &
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
          --anneal-lr ${anneal_lr} \
          "${wandb_args[@]}"
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
