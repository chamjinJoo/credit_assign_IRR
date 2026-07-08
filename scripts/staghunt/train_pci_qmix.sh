#!/bin/bash
# env setting
env="StagHunt"
num_agents=3
config="configs/staghunt_configs/staghunt_pci_qmix.yaml"

# model setting
model="pci-qmix"

# QMIX setting
lr=0.0005

# PCI-QMIX outcome bridge setting
lr_bridge=0.001
causal_lambda=1.0
bridge_epoch=4
causal_mini_batch_size=2000
standardize_delta=true
agent_delta_td_coef=1.0

# training and logging
n_rollout_threads=8
device="cuda"
seeds=(42 1 2 31 32)
run_name_postfix=""

export CUBLAS_WORKSPACE_CONFIG=:4096:8

for seed in "${seeds[@]}"; do
    echo "[seed=${seed}] start"
    python main.py --env ${env} --num-agents ${num_agents} \
        --config ${config} --model ${model} \
        --lr ${lr} \
        --lr-bridge ${lr_bridge} \
        --causal-lambda ${causal_lambda} \
        --bridge-epoch ${bridge_epoch} \
        --causal-mini-batch-size ${causal_mini_batch_size} \
        --agent-delta-td-coef ${agent_delta_td_coef} \
        --standardize-delta ${standardize_delta} \
        --n-rollout-threads ${n_rollout_threads} \
        --device ${device} \
        --seed ${seed} \
        --use-wandb \
        --wandb-run-name-postfix "${run_name_postfix}"
done
