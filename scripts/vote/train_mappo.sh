#!/bin/bash
set -euo pipefail

# env
env="Vote"
config="configs/vote_configs/vote_mappo.yaml"

# model setting
model="r-mappo"
recurrent_N=1

# training and logging
n_rollout_threads=1
device="cpu"
seeds=(42 1 2 31 32)
run_in_background=false
max_iterations=2000
max_steps=512

eval "$(conda shell.bash hook)"
conda activate pcimarl

run_training() {
    local seed="$1"
    echo "[seed=${seed}] start"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python main.py \
        --env ${env} --config ${config} \
        --model ${model} \
        --recurrent-N ${recurrent_N} \
        --n-rollout-threads ${n_rollout_threads} \
        --device ${device} \
        --seed ${seed} \
        --max-iterations ${max_iterations} \
        --max-steps ${max_steps} \
        --use-wandb \
        --wandb-project pci-marl-vote
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
