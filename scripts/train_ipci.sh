#!/bin/bash
# env
env="JaxLBF"
config="configs/lbforaging.yaml"

# model setting
model="jax_pci_ippo"
use_centralized_critic=false
causal_lambda=1.0

# training and logging 
n_rollout_threads=16 # the number of parallel environments in JAX env.
log_interval=2
device="cuda"
seed=42

# Conda 초기화 및 환경 활성화
eval "$(conda shell.bash hook)"
conda activate pcimarl
export KMP_DUPLICATE_LIB_OK=TRUE

python main.py --env ${env} --config ${config} \
    --model ${model} \
    --use-centralized-critic ${use_centralized_critic} \
    --n-rollout-threads ${n_rollout_threads} \
    --causal-lambda ${causal_lambda} \
    --log-interval ${log_interval} \
    --device ${device} \
    --seed ${seed} \
    --use-wandb \
    --wandb-project pci-marl-lbf
