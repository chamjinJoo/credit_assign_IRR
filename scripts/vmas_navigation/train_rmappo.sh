#!/bin/bash

env="VMASNavigation"
config="configs/vmas_navigation.yaml"
num_agents=4
model="r-mappo"
use_value_parameter_sharing=true
global_reward=true

max_iterations=1500
max_steps_per_iteration=2000
n_rollout_threads=4
log_interval=2
device="cpu"
seed=42

eval "$(conda shell.bash hook)"
conda activate pcimarl

export CUBLAS_WORKSPACE_CONFIG=:4096:8

python main.py --env ${env} --num-agents ${num_agents} \
    --config ${config} --model ${model} --global-reward ${global_reward} \
    --use-value-parameter-sharing ${use_value_parameter_sharing} \
    --n-rollout-threads ${n_rollout_threads} \
    --max-steps ${max_steps_per_iteration} --max-iterations ${max_iterations} \
    --log-interval ${log_interval} --device ${device} --seed ${seed} \
    --use-wandb --wandb-project pci-marl-vmas-navigation
