#!/bin/bash

# env setting
env="StagHunt"
num_agents=3
forage_quantity=3 # default: None
grid_observation=True
config="configs/stag_hunt.yaml"

# model setting
model="ippo" # [pci-mappo, mappo, ippo]

# training and logging
seed=42
max_iterations=3000
max_steps_per_iteration=1024
log_interval=2
device="cpu"

# Conda 초기화 및 환경 활성화
eval "$(conda shell.bash hook)"
conda activate pcimarl

python main.py --env ${env} --num-agents ${num_agents} --forage-quantity ${forage_quantity} --grid-observation ${grid_observation} \
    --config ${config} --model ${model} --max-steps ${max_steps_per_iteration} --max-iterations ${max_iterations} \
    --log-interval ${log_interval} --use-wandb --device ${device} --seed ${seed} --use-wandb --wandb-project pci-marl-staghunt