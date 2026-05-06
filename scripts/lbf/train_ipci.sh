#!/bin/bash
# env
env="LBForaging"
config="configs/lbforaging.yaml"

# model setting
model="r-pci" 
lr_bridge=0.001 # 1e-3 default
use_recurrent_policy=true
use_centralized_critic=false
use_value_parameter_sharing=true
train_naivepredictor=false
recurrent_N=1

causal_lambda=0.1

# training and logging 
n_rollout_threads=4
log_interval=2
device="cpu"
seed=42

# Conda 초기화 및 환경 활성화
eval "$(conda shell.bash hook)"
conda activate pcimarl

python main.py --env ${env} --config ${config} \
    --model ${model} \
    --use-recurrent-policy ${use_recurrent_policy} \
    --use-centralized-critic ${use_centralized_critic} \
    --use-value-parameter-sharing ${use_value_parameter_sharing} \
    --lr-bridge ${lr_bridge} \
    --recurrent-N ${recurrent_N} \
    --n-rollout-threads ${n_rollout_threads} \
    --causal-lambda ${causal_lambda} \
    --log-interval ${log_interval} \
    --device ${device} \
    --seed ${seed} \
    --use-wandb \
    --wandb-project pci-marl-lbf