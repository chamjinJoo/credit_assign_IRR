#!/bin/bash
env="LBForaging"
model="coma" # [pci-mappo, mappo, ippo]
num_agents=3
seed=42
max_iterations=1500
max_steps_per_iteration=2000
log_interval=2
device="cpu"
config="configs/lbforaging.yaml"

# Conda 초기화 및 환경 활성화
eval "$(conda shell.bash hook)"
conda activate pcimarl

python main.py --env ${env} --num-agents ${num_agents} \
    --config ${config} --model ${model} --max-steps ${max_steps_per_iteration} \
    --max-iterations ${max_iterations} --log-interval ${log_interval} --device ${device} --seed ${seed} \
    --use-wandb 
