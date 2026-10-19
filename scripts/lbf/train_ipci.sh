#!/bin/bash
set -euo pipefail

# env
env="JaxLBF"
config="configs/lbf_configs/lbf_pci_ippo.yaml"

# model setting
model="jax-pci-ippo"
use_centralized_critic=false
train_naivepredictor=false
use_naive_predictor_for_delta=false
causal_lambda=0.1

# training and logging
n_rollout_threads=8
log_interval=2
device="cuda"
seeds=(42 1 2 31 32)

# Conda 초기화 및 환경 활성화
eval "$(conda shell.bash hook)"
conda activate pcimarl

run_training() {
    local seed="$1"
    echo "[seed=${seed}] start"
    
    python main.py --env ${env} --config ${config} \
        --model ${model} \
        --use-centralized-critic ${use_centralized_critic} \
        --n-rollout-threads ${n_rollout_threads} \
        --causal-lambda ${causal_lambda} \
        --train-naivepredictor ${train_naivepredictor} \
        --use-naive-predictor-for-delta ${use_naive_predictor_for_delta} \
        --log-interval ${log_interval} \
        --device ${device} \
        --seed ${seed} \
        --use-wandb
}

for seed in "${seeds[@]}"; do
    run_training "${seed}"
done
