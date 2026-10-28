#!/bin/bash
set -euo pipefail

# env setting
env="StagHunt"
num_agents=3
config="configs/staghunt_configs/staghunt_mappo.yaml"

# model setting
model="r-mappo" # [pci-rnn, pci-mappo, mappo, ippo, ippo-rnn, mappo-rnn]
recurrent_N=1

# training and logging
n_rollout_threads=16
device="cuda"
seeds=(42 1 2 31 32)

# CuBLAS workspace config for deterministic behavior
export CUBLAS_WORKSPACE_CONFIG=:4096:8

run_training() {
    local seed="$1"
    echo "[seed=${seed}] start]"
    python main.py --env ${env} --num-agents ${num_agents} \
        --config ${config} --model ${model} \
        --recurrent-N ${recurrent_N} \
        --n-rollout-threads ${n_rollout_threads} \
        --device ${device} \
        --seed ${seed} \
        --use-wandb
}

for seed in "${seeds[@]}"; do
    run_training "${seed}"
done
