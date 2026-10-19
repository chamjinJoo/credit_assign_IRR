#!/bin/bash
# env setting
env="StagHunt"
num_agents=3
config="configs/staghunt_configs/staghunt_pci_ippo.yaml"
 
# model setting
model="r-pci" 
use_centralized_critic=false # false: i-pci / true m-pci
train_naivepredictor=false
recurrent_N=1

causal_lambda=1.0

# Optional diagnostic proxy shuffling. The paper configuration leaves it off.
proxy_shuffle_mode="none"
proxy_shuffle_prob=0.0
proxy_shuffle_strategy="episode_axis"

# training and logging 
n_rollout_threads=8
device="cpu"
seeds=(42 1 2 31 32)

# CuBLAS workspace config for deterministic behavior
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for seed in ${seeds[@]}; do
    echo "[seed=${seed}] start]"
    python main.py --env ${env} --num-agents ${num_agents} \
        --config ${config} --model ${model} \
        --use-centralized-critic ${use_centralized_critic} \
        --recurrent-N ${recurrent_N} \
        --n-rollout-threads ${n_rollout_threads} \
        --causal-lambda ${causal_lambda} \
        --train-naivepredictor ${train_naivepredictor} \
        --proxy-shuffle-mode ${proxy_shuffle_mode} \
        --proxy-shuffle-prob ${proxy_shuffle_prob} \
        --proxy-shuffle-strategy ${proxy_shuffle_strategy} \
        --device ${device} \
        --seed ${seed} \
        --use-wandb
done
