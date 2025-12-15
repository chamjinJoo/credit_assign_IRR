#!/bin/bash
env="StagHunt"
model="pci-mappo" # [pci-mappo, mappo, ippo]
numagents=3
causal_lambda=1.0
use_value_parameter_sharing=false
causal_lambda_adaptive_per_agent=true
max_iterations=3000
max_steps_per_iteration=1024
log_interval=2
device="cpu"
config="configs/stag_hunt.yaml"
seed=1

# Conda 초기화 및 환경 활성화
eval "$(conda shell.bash hook)"
conda activate jaxmarl

python main.py --env ${env} --num-agents ${numagents} --config ${config} \
--model ${model} --use-value-parameter-sharing ${use_value_parameter_sharing} \
--max-steps ${max_steps_per_iteration} --max-iterations ${max_iterations} \
--causal-lambda ${causal_lambda} --causal-lambda-adaptive-per-agent ${causal_lambda_adaptive_per_agent} \
--log-interval ${log_interval} --use-wandb \
--device ${device} --seed ${seed}