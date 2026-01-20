#!/bin/bash
# env setting 
env="StagHunt"
forage_quantity=3   
num_agents=3
grid_observation=true
config="configs/stag_hunt.yaml"

# model setting
model="mappo" # [pci-mappo, mappo, ippo]

# training and logging
max_iterations=1500
max_steps_per_iteration=2000
log_interval=2
device="cpu"
seed=42

# Conda 초기화 및 환경 활성화
eval "$(conda shell.bash hook)"
conda activate pcimarl

python main.py --env ${env} --num-agents ${num_agents} --forage-quantity ${forage_quantity} --grid-observation ${grid_observation} \
    --config ${config} --model ${model} --max-steps ${max_steps_per_iteration} \
    --max-iterations ${max_iterations} --log-interval ${log_interval} --device ${device} --seed ${seed} \
    --use-wandb --wandb-project pci-marl-staghunt