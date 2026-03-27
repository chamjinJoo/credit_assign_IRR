#!/bin/bash
# env setting
env="SMACv2"
config="configs/smacv2.yaml" 

# model setting
model="qmix"

# training and logging 
n_rollout_threads=2
max_grad_norm=10
max_iterations=1500
max_steps_per_iteration=6400
log_interval=2
device="cuda"
seed=42

export SC2PATH="$HOME/StarCraftII"
# Conda 초기화 및 환경 활성화
eval "$(conda shell.bash hook)"
conda activate pcimarl

# CuBLAS workspace config for deterministic behavior
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python main.py --env ${env} --config ${config} --model ${model} \
    --max-grad-norm ${max_grad_norm} \
    --n-rollout-threads ${n_rollout_threads} \
    --max-steps ${max_steps_per_iteration} \
    --max-iterations ${max_iterations} \
    --log-interval ${log_interval} \
    --device ${device} \
    --seed ${seed} \
    --use-wandb --wandb-project pci-marl-smacv2
