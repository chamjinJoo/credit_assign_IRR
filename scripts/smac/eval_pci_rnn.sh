#!/bin/bash
# env setting
env="SMAC"
map_name="2s3z" 
config="configs/smac.yaml" 

# model setting
checkpoint="./save_checkpoints/20260202_145519/smacwrapper/checkpoints/checkpoint_300.pt"
model="pci-rnn" # [pci-rnn, pci-mappo, mappo, ippo]
use_value_parameter_sharing=true
recurrent_N=1
device="cpu"  # recommended to use cuda if available
seed=42

# Conda 초기화 및 환경 활성화
eval "$(conda shell.bash hook)"
conda activate pcimarl

# CuBLAS workspace config for deterministic behavior
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python main.py --env ${env} --map-name ${map_name} --config ${config} --model ${model} --recurrent-N ${recurrent_N} --test \
    --checkpoint ${checkpoint} --use-value-parameter-sharing ${use_value_parameter_sharing} --device ${device} --seed ${seed}