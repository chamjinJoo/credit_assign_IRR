#!/bin/bash
# env setting
env="SMACv2"
# scenarios: 10gen_protoss, 10gen_terran, 10gen_zerg
config="configs/smacv2.yaml" 

# model setting
model="r-mappo" # [pci-rnn, pci-mappo, mappo, ippo]
use_value_parameter_sharing=true
recurrent_N=1

# training and logging 
n_rollout_threads=4
max_iterations=1600
max_steps_per_iteration=6400 # 8 episode for 10gen scenarios (episode limit is 200 steps)
log_interval=2
device="cuda"  # recommended to use cuda if available, but default to cpu for safety
seed=42

export SC2PATH="$HOME/StarCraftII"

# when using Slurm
# SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# export SC2PATH="${SC2PATH:-$SCRIPT_ROOT/3rdparty/StarCraftII}"

# Conda 초기화 및 환경 활성화
eval "$(conda shell.bash hook)"
conda activate pcimarl

# CuBLAS workspace config for deterministic behavior
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python main.py --env ${env} --config ${config} --model ${model} \
    --use-value-parameter-sharing ${use_value_parameter_sharing} \
    --n-rollout-threads ${n_rollout_threads} \
    --recurrent-N ${recurrent_N} \
    --max-steps ${max_steps_per_iteration} \
    --max-iterations ${max_iterations} \
    --log-interval ${log_interval} \
    --device ${device} --seed ${seed} \
    --use-wandb \
    --wandb-project pci-marl-smacv2 