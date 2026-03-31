#!/bin/bash
# Q-PCI Training Script for StagHunt Environment
# Value Bridge Network predicts action values instead of immediate rewards

env="StagHunt"
num_agents=3
config="configs/stag_hunt.yaml"
 
# model setting
model="q-pci" 
causal_lambda=0.5
clip_delta=0.0
recurrent_N=1
train_naivepredictor=true

# training and logging 
max_iterations=1500
max_steps_per_iteration=2000
n_rollout_threads=4
log_interval=2
device="cpu"
seeds=(42)

# Conda initialization
eval "$(conda shell.bash hook)"
conda activate pcimarl

# CuBLAS workspace config for deterministic behavior
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for seed in ${seeds[@]}; do
    echo "[seed=${seed}] start]"
    python main.py --env ${env} --num-agents ${num_agents} \
        --config ${config} --model ${model} \
        --recurrent-N ${recurrent_N} \
        --train-naivepredictor ${train_naivepredictor} \
        --n-rollout-threads ${n_rollout_threads} \
        --max-steps ${max_steps_per_iteration} \
        --max-iterations ${max_iterations} \
        --clip-delta ${clip_delta} \
        --causal-lambda ${causal_lambda} \
        --log-interval ${log_interval} --device ${device} --seed ${seed} \
        --use-wandb --wandb-project pci-marl-staghunt \
        # & # run in parallel
    echo "[seed=${seed}] PID: $!"
done

echo "Q-PCI training started... (${#seeds[@]} parallel runs)"
