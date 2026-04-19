#!/bin/bash
# ============================================================
# Draft recurrent JAX PCI launcher for SMAX
# ============================================================

set -euo pipefail

config="configs_jax/smax_rpci.yaml"

# rpci settings
use_centralized_critic=false # false: IPPO-critic / true: MAPPO-critic
causal_lambda=1.0
clip_delta=0.0

if ${use_centralized_critic}; then
  model="jax-pci-mappo"
else
  model="jax-pci-ippo"
fi

# training settings
n_rollout_threads=4
max_iterations=1600
max_steps_per_iteration=6400 # follow the smax paper: 128 env * 64 steps
log_interval=2
device="cuda"
seeds=(42)
run_in_background=false

#checkpoint="save_checkpoints/20260412_134339_seed42/smaxbatchwrapper/checkpoints/jax_checkpoint_iter_162.pkl"

# conda activation (optional)
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate pcimarl
fi

# reproducibility settings for JAX
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
        --use-centralized-critic ${use_centralized_critic} \
        --causal-lambda ${causal_lambda} \
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
        --use-centralized-critic ${use_centralized_critic} \
        --causal-lambda ${causal_lambda} \
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
