#!/bin/bash
set -euo pipefail

# env setting
env="StagHunt"
num_agents=3
config="configs/staghunt_configs/staghunt_qmix.yaml"

# model setting
model="qmix"

# QMIX setting
lr=0.0005

# training and logging
n_rollout_threads=8
device="cuda"
seeds=(42 1 2 31 32)

export CUBLAS_WORKSPACE_CONFIG=:4096:8

run_training() {
    local seed="$1"
    echo "[seed=${seed}] start"
    python main.py --env ${env} --num-agents ${num_agents} \
        --config ${config} --model ${model} \
        --lr ${lr} \
        --n-rollout-threads ${n_rollout_threads} \
        --device ${device} \
        --seed ${seed} \
        --use-wandb
}

for seed in "${seeds[@]}"; do
    run_training "${seed}"
done
