#!/bin/bash
set -euo pipefail

# env setting
env="StagHunt"
num_agents=3
config="configs/staghunt_configs/staghunt_pci_ippo.yaml"
 
# model setting
model="r-pci" 
use_centralized_critic=false # false: i-pci / true m-pci
recurrent_N=1

causal_lambda=1.0

# training and logging 
n_rollout_threads=8
device="cpu"
seeds=(42 1 2 31 32)

# CuBLAS workspace config for deterministic behavior
export CUBLAS_WORKSPACE_CONFIG=:4096:8

run_training() {
    local seed="$1"
    echo "[seed=${seed}] start]"
    python main.py --env ${env} --num-agents ${num_agents} \
        --config ${config} --model ${model} \
        --use-centralized-critic ${use_centralized_critic} \
        --recurrent-N ${recurrent_N} \
        --n-rollout-threads ${n_rollout_threads} \
        --causal-lambda ${causal_lambda} \
        --device ${device} \
        --seed ${seed} \
        --use-wandb
}

for seed in "${seeds[@]}"; do
    run_training "${seed}"
done
