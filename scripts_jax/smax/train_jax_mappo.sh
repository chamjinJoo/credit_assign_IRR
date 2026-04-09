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
device="cuda"  # recommended to use cuda if available, but default to cpu for safety
seed=42

test=false
checkpoint="./save_checkpoints/20260406_185738_seed42/smaxbatchwrapper/checkpoints/jax_checkpoint_iter_1599.pkl"
save_replay=true
replay_dir="./smax_replays"
replay_episodes=10

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate pcimarl
fi

export CUBLAS_WORKSPACE_CONFIG=:4096:8
# export KMP_DUPLICATE_LIB_OK=TRUE # for Window

if ${test}; then
  echo "Running in test with checkpoint: ${checkpoint}"
  replay_flag=""
  if ${save_replay}; then
    replay_flag="--save-replay --replay-dir ${replay_dir} --replay-episodes ${replay_episodes}"
  fi

  python main.py \
    --env SMAX \
    --config ${config} \
    --model ${model} \
    --use-recurrent-policy ${use_recurrent_policy} \
    --n-rollout-threads ${n_rollout_threads} \
    --log-interval ${log_interval} \
    --device ${device} \
    --seed ${seed} \
    --test \
    --checkpoint ${checkpoint} \
    ${replay_flag}
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
    --wandb-project pci-marl-smax-jax
fi
