#!/bin/bash
set -euo pipefail

# env
env="JaxLBF"
config="configs/lbf_configs/lbf_ippo.yaml"

# model setting
model="jax-ippo"
use_centralized_critic=false

# training and logging 
n_rollout_threads=16 # the number of parallel environments in JAX env.
log_interval=2
device="cuda"
seeds=(42)
run_in_background=false

# Conda 초기화 및 환경 활성화
eval "$(conda shell.bash hook)"
conda activate pcimarl

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
        --use-wandb \
        --wandb-project pci-marl-lbf
}

for seed in "${seeds[@]}"; do
    if [ "${run_in_background}" = "true" ]; then
        run_training "${seed}" &
    else
        run_training "${seed}"
    fi
done

if [ "${run_in_background}" = "true" ]; then
    wait
fi
