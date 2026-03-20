#!/bin/bash
# env setting 
env="StagHunt"
num_agents=3
config="configs/stag_hunt.yaml"

# model setting
model="qmix"

# training and logging
n_rollout_threads=4
max_iterations=3000
max_steps_per_iteration=2000
episodes_per_iteration=40  # Match 2000 transitions/iter: 40 eps → 10 eps/env × 50 steps × 4 envs = 2000 transitions
log_interval=2
device="cpu"
seed=42

# Conda 초기화 및 환경 활성화
eval "$(conda shell.bash hook)"
conda activate pcimarl

# CuBLAS workspace config for deterministic behavior
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python main.py --env ${env} --num-agents ${num_agents} \
    --config ${config} --model ${model} \
    --max-steps ${max_steps_per_iteration} \
    --max-iterations ${max_iterations} \
    --episodes-per-iteration ${episodes_per_iteration} \
    --log-interval ${log_interval} \
    --device ${device} \
    --seed ${seed} \
    --use-wandb --wandb-project pci-marl-staghunt
