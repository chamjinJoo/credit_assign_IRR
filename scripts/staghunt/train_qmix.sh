#!/bin/bash
# env setting
env="StagHunt"
num_agents=3
config="configs/stag_hunt.yaml"

# model setting
model="qmix"

# QMIX setting
lr=0.0005
target_update_interval=200
epsilon_start=1.0
epsilon_finish=0.05
epsilon_anneal_time=100000
replay_buffer_size=5000
train_batch_size=32
# Match PPO-style rollouts in configs/stag_hunt.yaml:
# 40 episodes * 50 max_steps = 2000 env steps per iteration.
episodes_per_iteration=40

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
        --target-update-interval ${target_update_interval} \
        --epsilon-start ${epsilon_start} \
        --epsilon-finish ${epsilon_finish} \
        --epsilon-anneal-time ${epsilon_anneal_time} \
        --replay-buffer-size ${replay_buffer_size} \
        --train-batch-size ${train_batch_size} \
        --episodes-per-iteration ${episodes_per_iteration} \
        --n-rollout-threads ${n_rollout_threads} \
        --device ${device} \
        --seed ${seed} \
        --use-wandb \
        --wandb-run-name-postfix "${run_name_postfix}"
done
