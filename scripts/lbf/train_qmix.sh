#!/bin/bash
set -euo pipefail

# env
env="JaxLBF"
config="configs/lbf_configs/lbf_qmix.yaml"

# model setting
model="jax-qmix"

# training and logging
n_rollout_threads=8
log_interval=2
device="cuda"
seeds=(42 1 2 31 32)

eval "$(conda shell.bash hook)"
conda activate pcimarl

run_training() {
    local seed="$1"
    echo "[seed=${seed}] start"
    python main.py --env ${env} --config ${config} \
        --model ${model} \
        --n-rollout-threads ${n_rollout_threads} \
        --log-interval ${log_interval} \
        --device ${device} \
        --seed ${seed} \
        --use-wandb
}

for seed in "${seeds[@]}"; do
    run_training "${seed}"
done
