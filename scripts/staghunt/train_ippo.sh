#!/bin/bash
# env setting
env="StagHunt"
num_agents=3
config="configs/staghunt_configs/staghunt_ippo.yaml"
 
# model setting
model="r-ippo"
recurrent_N=1

# training and logging 
n_rollout_threads=16
device="cpu"
seeds=(42 1 2 31 32)

# CuBLAS workspace config for deterministic behavior
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for seed in ${seeds[@]}; do
    echo "[seed=${seed}] start]"
    python main.py --env ${env} --num-agents ${num_agents} \
        --config ${config} --model ${model} \
        --recurrent-N ${recurrent_N} \
        --n-rollout-threads ${n_rollout_threads} \
        --device ${device} \
        --seed ${seed} \
        --use-wandb
done
