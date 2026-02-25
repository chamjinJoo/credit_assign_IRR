#!/bin/bash
# env setting
env="SMACv2"
# scenarios: 10gen_protoss, 10gen_terran, 10gen_zerg
config="configs/smacv2.yaml" 

# model setting
model="r-pci" # [pci-rnn, pci-mappo, mappo, ippo]
causal_lambda=0.1 # 0.1 for SMAC, SMACv2
clip_delta=0.01
use_centralized_critic=false  # true: MAPPO-style centralized critic, false: IPPO-style individual critic

use_value_parameter_sharing=true
causal_lambda_adaptive_per_agent=false

# training and logging 
n_rollout_threads=8
max_iterations=1500
max_steps_per_iteration=6400 # 200 steps = 1 episode limit for 10gen scenarios
episodes_per_iteration=0 # if it is positive, use EpiosdeBuffer automatically 
log_interval=2
device="cpu"
seed=42

# load model if necessary
# checkpoint="save_checkpoints/20260222_210018/smacv2wrapper/checkpoints/checkpoint_900.pt"

export SC2PATH="$HOME/StarCraftII"
# Conda 초기화 및 환경 활성화
eval "$(conda shell.bash hook)"
conda activate pcimarl

# CuBLAS workspace config for deterministic behavior
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# headless mode (no rendering)
xvfb-run -a python main.py --env ${env} --config ${config} --model ${model} \
    --use-value-parameter-sharing ${use_value_parameter_sharing} \
    --use-centralized-critic ${use_centralized_critic} \
    --n-rollout-threads ${n_rollout_threads} \
    --max-steps ${max_steps_per_iteration} \
    --episodes-per-iteration ${episodes_per_iteration} \
    --max-iterations ${max_iterations} \
    --clip-delta ${clip_delta} \
    --causal-lambda ${causal_lambda} \
    --causal-lambda-adaptive-per-agent ${causal_lambda_adaptive_per_agent} \
    --log-interval ${log_interval} \
    --device ${device} --seed ${seed} \
    --use-wandb \
    --wandb-project pci-marl-smacv2 \
    # --checkpoint ${checkpoint}