#!/bin/bash
# env setting
env="StagHunt"
num_agents=3
config="configs/stag_hunt.yaml"

# model setting
model="pci-qmix"
causal_lambda=1.0
clip_delta=0.01
bridge_update_interval=2
bridge_epoch=1
train_naivepredictor=false

# training and logging
max_iterations=1500
max_steps_per_iteration=2000
#episodes_per_iteration=40 # QMIX uses EpisodeBuffer by default
n_rollout_threads=2
log_interval=2
device="cpu"
seed=42

# Conda 초기화 및 환경 활성화
eval "$(conda shell.bash hook)"
conda activate pcimarl

# CuBLAS workspace config for deterministic behavior
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python main.py --env ${env} --num-agents ${num_agents} \
        --config ${config} --model ${model} \
        --causal-lambda ${causal_lambda} \
        --clip-delta ${clip_delta} \
        --train-naivepredictor ${train_naivepredictor} \
        --n-rollout-threads ${n_rollout_threads} \
        --max-steps ${max_steps_per_iteration} \
        --max-iterations ${max_iterations} \
        --log-interval ${log_interval} --device ${device} --seed ${seed} \
        --use-wandb --wandb-project pci-marl-staghunt