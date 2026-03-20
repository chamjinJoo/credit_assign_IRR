#!/bin/bash
# env setting
env="SMACv2"
config="configs/smacv2.yaml" 

# model setting
model="qmix"

# training and logging 
n_rollout_threads=4
max_iterations=3000
max_steps_per_iteration=3200
log_interval=2
device="cpu"
seed=42

export SC2PATH="$HOME/StarCraftII"
# Conda 초기화 및 환경 활성화
eval "$(conda shell.bash hook)"
conda activate pcimarl

# CuBLAS workspace config for deterministic behavior
export CUBLAS_WORKSPACE_CONFIG=:4096:8

xvfb-run -a python main.py --env ${env} --config ${config} --model ${model} \
    --n-rollout-threads ${n_rollout_threads} \
    --max-steps ${max_steps_per_iteration} \
    --max-iterations ${max_iterations} \
    --log-interval ${log_interval} \
    --device ${device} \
    --seed ${seed} \
    --use-wandb --wandb-project pci-marl-smacv2
