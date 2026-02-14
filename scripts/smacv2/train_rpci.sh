#!/bin/bash
# env setting
env="SMACv2"
# scenarios: 10gen_protoss, 10gen_terran, 10gen_zerg
config="configs/smacv2.yaml" 

# model setting
model="r-pci" # [pci-rnn, pci-mappo, mappo, ippo]
causal_lambda=1.0
clip_delta=0.01
use_value_parameter_sharing=true
causal_lambda_adaptive_per_agent=false
recurrent_N=1

# training and logging 
n_rollout_threads=8
max_iterations=3000
max_steps_per_iteration=3200 # 8 episode for 10gen scenarios (episode limit is 200 steps)
log_interval=2
device="cpu"  # recommended to use cuda if available
seed=42

export SC2PATH="$HOME/StarCraftII"
# Conda 초기화 및 환경 활성화
eval "$(conda shell.bash hook)"
conda activate pcimarl

# CuBLAS workspace config for deterministic behavior
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# headless mode (no rendering)
xvfb-run -a python main.py --env ${env} --config ${config} --model ${model} \
    --use-value-parameter-sharing ${use_value_parameter_sharing} \
    --n-rollout-threads ${n_rollout_threads} \
    --max-steps ${max_steps_per_iteration} \
    --max-iterations ${max_iterations} \
    --clip-delta ${clip_delta} \
    --causal-lambda ${causal_lambda} \
    --causal-lambda-adaptive-per-agent ${causal_lambda_adaptive_per_agent} \
    --log-interval ${log_interval} \
    --device ${device} --seed ${seed} \
    --use-wandb \
    --wandb-project pci-marl-smacv2