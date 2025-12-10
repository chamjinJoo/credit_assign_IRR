#!/bin/bash
env="StagHunt"
model="pci-mappo" # [pci-mappo, mappo, ippo]
numagents=3
max_iterations=3000
max_steps_per_iteration=1024
log_interval=2
device="cpu"
config="configs/stag_hunt.yaml"
seed=331

# Conda 초기화 및 환경 활성화
eval "$(conda shell.bash hook)"
conda activate jaxmarl

python main.py --env ${env} --num-agents ${numagents} --config ${config} --model ${model} \
--max-steps ${max_steps_per_iteration} --max-iterations ${max_iterations} --causal-lambda 1.0 \
--log-interval ${log_interval} --use-wandb --device ${device} --seed ${seed}