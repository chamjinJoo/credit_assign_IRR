#!/bin/bash
# env setting
env="SMACv2"
map_name="10gen_protoss" 
# scenarios: 10gen_protoss, 10gen_terran, 10gen_zerg
config="configs/smacv2.yaml" 

# model setting
model="pci-rnn" # [pci-rnn, pci-mappo, mappo, ippo]
causal_lambda=1.0
use_value_parameter_sharing=true
causal_lambda_adaptive_per_agent=false
recurrent_N=1

# training and logging 
max_iterations=3000
max_steps_per_iteration=2000
log_interval=2
device="cpu"  # recommended to use cuda if available
seed=42

# Conda 초기화 및 환경 활성화
eval "$(conda shell.bash hook)"
conda activate pcimarl

# CuBLAS workspace config for deterministic behavior
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python main.py --env ${env} --map-name ${map_name} --config ${config} --model ${model} \
    --use-value-parameter-sharing ${use_value_parameter_sharing} --max-steps ${max_steps_per_iteration} --max-iterations ${max_iterations} \
    --causal-lambda ${causal_lambda} --causal-lambda-adaptive-per-agent ${causal_lambda_adaptive_per_agent} \
    --log-interval ${log_interval} --device ${device} --seed ${seed} --use-wandb --wandb-project pci-marl-smacv2 
