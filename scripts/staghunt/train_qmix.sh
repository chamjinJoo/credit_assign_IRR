#!/bin/bash
# env setting 
env="StagHunt"
num_agents=3
config="configs/stag_hunt.yaml"

# model setting
model="qmix"

# training and logging
n_rollout_threads=2
max_grad_norm=10
max_iterations=1500
max_steps_per_iteration=2000
# episodes_per_iteration=0  # Match 2000 transitions/iter: 40 eps → 10 eps/env × 50 steps × 4 envs = 2000 transitions
log_interval=2
device="cpu"
seed=31
# checkpoint="save_checkpoints/20260320_215805_seed42/staghuntenv/checkpoints/checkpoint_300.pt"
# Conda 초기화 및 환경 활성화
eval "$(conda shell.bash hook)"
conda activate pcimarl

# CuBLAS workspace config for deterministic behavior
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python main.py --env ${env} --num-agents ${num_agents} \
    --config ${config} --model ${model} \
    --max-grad-norm ${max_grad_norm} \
    --n-rollout-threads ${n_rollout_threads} \
    --max-steps ${max_steps_per_iteration} \
    --max-iterations ${max_iterations} \
    --log-interval ${log_interval} \
    --device ${device} \
    --seed ${seed} \
    --use-wandb --wandb-project pci-marl-staghunt # --checkpoint ${checkpoint}
