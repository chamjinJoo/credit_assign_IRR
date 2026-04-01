#!/bin/bash
# env setting
env="StagHunt"
num_agents=3
forage_quantity=3 # default: the same as num_agents
grid_observation=true
config="configs/stag_hunt.yaml"

# model setting
model="r-mappo" # [pci-rnn, pci-mappo, mappo, ippo, ippo-rnn, mappo-rnn]
use_recurrent_policy=false
use_value_parameter_sharing=true
recurrent_N=1
global_reward=true

# training and logging
max_iterations=1500
max_steps_per_iteration=2000
n_rollout_threads=4
log_interval=2
device="cpu"
seeds=(42)

# Conda 초기화 및 환경 활성화
eval "$(conda shell.bash hook)"
conda activate pcimarl

# CuBLAS workspace config for deterministic behavior
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for seed in ${seeds[@]}; do
    echo "[seed=${seed}] start]"
    python main.py --env ${env} --num-agents ${num_agents} \
        --forage-quantity ${forage_quantity} \
        --grid-observation ${grid_observation} \
        --config ${config} --model ${model} \
        --use-recurrent-policy ${use_recurrent_policy} \
        --global-reward ${global_reward} \
        --use-value-parameter-sharing ${use_value_parameter_sharing} \
        --recurrent-N ${recurrent_N} \
        --n-rollout-threads ${n_rollout_threads} \
        --max-steps ${max_steps_per_iteration} \
        --max-iterations ${max_iterations} \
        --log-interval ${log_interval} \
        --device ${device} \
        --seed ${seed} \
        --use-wandb --wandb-project pci-marl-staghunt \
        # & # deactivate for sequential training
    echo "[seed=${seed}] PID: $!"
done
