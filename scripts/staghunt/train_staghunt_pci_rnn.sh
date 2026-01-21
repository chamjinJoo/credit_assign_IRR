#!/bin/bash
# env setting
env="StagHunt"
num_agents=3
forage_quantity=3 # default: the same as num_agents
grid_observation=true
config="configs/stag_hunt.yaml"

# model setting
model="pci-rnn" # [pci-rnn, pci-mappo, mappo, ippo]
causal_lambda=1.0
use_value_parameter_sharing=true
causal_lambda_adaptive_per_agent=false
recurrent_N=1

# training and logging 
max_iterations=1500
max_steps_per_iteration=2000
n_rollout_threads=8
log_interval=2
device="cpu"
seed=3

# Conda 초기화 및 환경 활성화
eval "$(conda shell.bash hook)"
conda activate pcimarl

# CuBLAS workspace config for deterministic behavior
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python main.py --env ${env} --num-agents ${num_agents} --forage-quantity ${forage_quantity} --grid-observation ${grid_observation} \
    --config ${config} --model ${model} --use-value-parameter-sharing ${use_value_parameter_sharing} \
    --recurrent-N ${recurrent_N} --n-rollout-threads ${n_rollout_threads} --max-steps ${max_steps_per_iteration} --max-iterations ${max_iterations} \
    --causal-lambda ${causal_lambda} --causal-lambda-adaptive-per-agent ${causal_lambda_adaptive_per_agent} \
    --log-interval ${log_interval} --device ${device} --seed ${seed} --use-wandb --wandb-project pci-marl-staghunt
