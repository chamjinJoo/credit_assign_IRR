#!/bin/bash
# env setting
env="StagHunt"
num_agents=3
config="configs/stag_hunt.yaml"
 
# model setting
model="r-pci" 
use_centralized_critic=false # false: i-pci / true m-pci
use_value_parameter_sharing=true
train_naivepredictor=true
recurrent_N=1

causal_lambda=1.0

# training and logging 
max_iterations=4000
max_steps_per_iteration=2000
episodes_per_iteration=0 # if it is positive, use EpiosdeBuffer automatically 
n_rollout_threads=8
log_interval=2
device="cpu"
seeds=(1 2 31 32)

# Conda 초기화 및 환경 활성화
eval "$(conda shell.bash hook)"
conda activate pcimarl

# CuBLAS workspace config for deterministic behavior
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for seed in ${seeds[@]}; do
    echo "[seed=${seed}] start]"
    python main.py --env ${env} --num-agents ${num_agents} \
        --config ${config} --model ${model} \
        --use-centralized-critic ${use_centralized_critic} \
        --use-value-parameter-sharing ${use_value_parameter_sharing} \
        --recurrent-N ${recurrent_N} \
        --train-naivepredictor ${train_naivepredictor} \
        --n-rollout-threads ${n_rollout_threads} \
        --max-steps ${max_steps_per_iteration} \
        --episodes-per-iteration ${episodes_per_iteration} \
        --max-iterations ${max_iterations} \
        --causal-lambda ${causal_lambda} \
        --log-interval ${log_interval} --device ${device} --seed ${seed} \
        --use-wandb \
        --wandb-project pci-marl-staghunt-vanilla \
        # & # deactivate for sequential training
    echo "[seed=${seed}] PID: $!"
done