#!/bin/bash
set -euo pipefail

# env
env="JaxLBF"
config="configs/lbf_configs/lbf_mappo.yaml"

# model setting
model="jax-mappo"
use_centralized_critic=true

# training and logging 
n_rollout_threads=8 # the number of parallel environments in JAX env.
log_interval=2
device="cuda"
seeds=(42 1 2 31 32)

run_training() {
    local seed="$1"
    echo "[seed=${seed}] start"
    python main.py --env ${env} --config ${config} \
        --model ${model} \
        --use-centralized-critic ${use_centralized_critic} \
        --n-rollout-threads ${n_rollout_threads} \
        --log-interval ${log_interval} \
        --device ${device} \
        --seed ${seed} \
        --use-wandb
}

for seed in "${seeds[@]}"; do
    run_training "${seed}"
done
