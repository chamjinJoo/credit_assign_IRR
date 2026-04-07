#!/bin/bash
# env setting
env="StagHunt"
num_agents=3
config="configs/stag_hunt.yaml"
 
# model setting
model="r-pci" 
use_recurrent_policy=true
use_centralized_critic=false
use_value_parameter_sharing=true
train_naivepredictor=false
recurrent_N=1

causal_lambda=1.0
clip_delta=0.0

# training and logging 
max_iterations=1500
max_steps_per_iteration=2000
episodes_per_iteration=0 # if it is positive, use EpiosdeBuffer automatically 
n_rollout_threads=4
log_interval=2
device="cpu"
seeds=(42)

# Conda 초기화 및 환경 활성화
eval "$(conda shell.bash hook)"
conda activate pcimarl

# CuBLAS workspace config for deterministic behavior
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for seed in ${seeds[@]}; do
    echo "[seed=${seed}] start]"
    python main.py --env ${env} --num-agents ${num_agents} \
        --config ${config} --model ${model} \
        --use-recurrent-policy ${use_recurrent_policy} \
        --use-centralized-critic ${use_centralized_critic} \
        --use-value-parameter-sharing ${use_value_parameter_sharing} \
        --recurrent-N ${recurrent_N} \
        --train-naivepredictor ${train_naivepredictor} \
        --n-rollout-threads ${n_rollout_threads} \
        --max-steps ${max_steps_per_iteration} \
        --episodes-per-iteration ${episodes_per_iteration} \
        --max-iterations ${max_iterations} \
        --clip-delta ${clip_delta} \
        --causal-lambda ${causal_lambda} \
        --log-interval ${log_interval} --device ${device} --seed ${seed} \
        --use-wandb \
        --wandb-project pci-marl-staghunt \
        # & # deactivate for sequential training
    echo "[seed=${seed}] PID: $!"
done

echo "all seed is running... (${#seeds[@]} parallel processes)"
wait
echo "all seed training completed!"