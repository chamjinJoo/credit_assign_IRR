#!/bin/bash
# env setting
env="SMAC"
map_name="2s3z"
# easy: 3m, 2s3z, 10m_vs_11m (for scalability)
# hard: Corridor, 5m_vs_6m, 3s_vs_5z
config="configs/smac.yaml" # using basic config as template

# model setting
model="ippo-rnn" # [pci-rnn, pci-mappo, mappo, ippo]
use_value_parameter_sharing=true
recurrent_N=1

# training and logging 
max_iterations=1500
max_steps_per_iteration=2000
log_interval=2
device="cpu"  # recommended to use cuda if available, but default to cpu for safety
seed=42

# Conda 초기화 및 환경 활성화
eval "$(conda shell.bash hook)"
conda activate pcimarl

# CuBLAS workspace config for deterministic behavior
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python main.py --env ${env} --map-name ${map_name} --config ${config} --model ${model} \
    --use-value-parameter-sharing ${use_value_parameter_sharing} \
    --recurrent-N ${recurrent_N} --max-steps ${max_steps_per_iteration} --max-iterations ${max_iterations} \
    --log-interval ${log_interval} --device ${device} --seed ${seed} --use-wandb --wandb-project pci-marl-smac \