#!/bin/bash
# env setting
env="SMACv2"
# scenarios: 10gen_protoss, 10gen_terran, 10gen_zerg
config="configs/smacv2.yaml"

# model setting
model="ippo-rnn" # [pci-rnn, pci-mappo, mappo, ippo]
use_value_parameter_sharing=true
recurrent_N=1

# training and logging 
max_iterations=1500
max_steps_per_iteration=1000
log_interval=2
device="cpu"  # recommended to use cuda if available, but default to cpu for safety
seed=42
n_rollout_threads=8

# Conda 초기화 및 환경 활성화
eval "$(conda shell.bash hook)"
conda activate pcimarl

# CuBLAS workspace config for deterministic behavior
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python main.py --env ${env} --config ${config} --model ${model} \
    --use-value-parameter-sharing ${use_value_parameter_sharing} \
    --recurrent-N ${recurrent_N} --max-steps ${max_steps_per_iteration} --max-iterations ${max_iterations} \
    --log-interval ${log_interval} --device ${device} --seed ${seed} --use-wandb --wandb-project pci-marl-smacv2 \
    --n-rollout-threads ${n_rollout_threads}
