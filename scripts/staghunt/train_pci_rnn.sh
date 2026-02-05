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
clip_delta=0.01
use_value_parameter_sharing=true
recurrent_N=1

# training and logging 
max_iterations=1500
max_steps_per_iteration=2000
episodes_per_iteration=32 # if it is set, use EpiosdeBuffer automatically 
n_rollout_threads=8
log_interval=2
device="cpu"
seed=2

# Conda 초기화 및 환경 활성화
eval "$(conda shell.bash hook)"
conda activate pcimarl

# CuBLAS workspace config for deterministic behavior
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python main.py --env ${env} --num-agents ${num_agents} --forage-quantity ${forage_quantity} --grid-observation ${grid_observation} \
    --config ${config} --model ${model} --use-value-parameter-sharing ${use_value_parameter_sharing} \
    --recurrent-N ${recurrent_N} \
    --n-rollout-threads ${n_rollout_threads} \
    --max-steps ${max_steps_per_iteration} \
    --episodes-per-iteration ${episodes_per_iteration} \
    --max-iterations ${max_iterations} \
    --clip-delta ${clip_delta} \
    --causal-lambda ${causal_lambda} \
    --log-interval ${log_interval} --device ${device} --seed ${seed} \
    --use-wandb --wandb-project pci-marl-staghunt
