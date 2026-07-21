#!/bin/bash
set -euo pipefail

# env
env="JaxLBF"
config="configs/lbf_configs/lbf_pci_qmix.yaml"

# model setting
model="jax-pci-qmix"
causal_lambda=1.0

# training and logging
n_rollout_threads=16
log_interval=2
device="cuda"
seeds=(1)
run_in_background=false

eval "$(conda shell.bash hook)"
conda activate pcimarl

run_training() {
    local seed="$1"
    echo "[seed=${seed}] start"
    python main.py --env ${env} --config ${config} \
        --model ${model} \
        --n-rollout-threads ${n_rollout_threads} \
        --causal-lambda ${causal_lambda} \
        --log-interval ${log_interval} \
        --device ${device} \
        --seed ${seed} \
        --use-wandb \
        --wandb-project pci-marl-lbf
}

for seed in "${seeds[@]}"; do
    if [ "${run_in_background}" = "true" ]; then
        run_training "${seed}" &
    else
        run_training "${seed}"
    fi
done

if [ "${run_in_background}" = "true" ]; then
    wait
fi
