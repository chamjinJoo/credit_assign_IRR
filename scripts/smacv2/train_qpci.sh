#!/bin/bash
# Q-PCI Training Script for SMACv2 Environment
# Value Bridge Network predicts action values instead of immediate rewards

env="SMACv2"
config="configs/smacv2.yaml" 

# model setting
model="q-pci"
causal_lambda=0.5
clip_delta=0.0
use_centralized_critic=true
use_value_parameter_sharing=true

# training and logging 
n_rollout_threads=4
max_iterations=1000
max_steps_per_iteration=6400
log_interval=2
device="cpu"
seed=42

export SC2PATH="$HOME/StarCraftII"

# Conda initialization
eval "$(conda shell.bash hook)"
conda activate pcimarl

# CuBLAS workspace config for deterministic behavior
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Headless mode (no rendering)
python main.py --env ${env} --config ${config} --model ${model} \
    --use-value-parameter-sharing ${use_value_parameter_sharing} \
    --use-centralized-critic ${use_centralized_critic} \
    --n-rollout-threads ${n_rollout_threads} \
    --max-steps ${max_steps_per_iteration} \
    --max-iterations ${max_iterations} \
    --clip-delta ${clip_delta} \
    --causal-lambda ${causal_lambda} \
    --log-interval ${log_interval} \
    --device ${device} --seed ${seed} \
    --use-wandb \
    --wandb-project pci-marl-smacv2
