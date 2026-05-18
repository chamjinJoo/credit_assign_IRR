#!/bin/bash
# env setting
env="StagHunt"
num_agents=3
forage_quantity=3 # default: the same as num_agents
grid_observation=true
config="configs/stag_hunt.yaml"

# model setting
model="r-ippo" # [pci-rnn, pci-mappo, mappo, ippo, ippo-rnn, mappo-rnn]
use_value_parameter_sharing=true
recurrent_N=1

# training and logging
n_rollout_threads=4
device="cpu"
seeds=(42 1 2 31 32)

# CuBLAS workspace config for deterministic behavior
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for seed in ${seeds[@]}; do
    echo "[seed=${seed}] start]"
    python main.py --env ${env} --num-agents ${num_agents} --forage-quantity ${forage_quantity} \
        --grid-observation ${grid_observation} \
        --config ${config} --model ${model} \
        --use-value-parameter-sharing ${use_value_parameter_sharing} \
        --recurrent-N ${recurrent_N} \
        --n-rollout-threads ${n_rollout_threads} \
        --device ${device} \
        --seed ${seed}
done