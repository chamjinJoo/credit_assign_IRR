#!/bin/bash
# env setting
env="StagHunt"
num_agents=3
config="configs/stag_hunt.yaml"

# model setting
model="pci-qmix"
causal_lambda=1.0
clip_delta=0.01
bridge_update_interval=1
bridge_epoch=1
train_naivepredictor=false

# training and logging
max_iterations=1500
max_steps_per_iteration=2000
episodes_per_iteration=40 # QMIX uses EpisodeBuffer by default
n_rollout_threads=4
log_interval=2
device="cuda"
seeds=(42)

# Conda 초기화 및 환경 활성화
eval "$(conda shell.bash hook)"
conda activate pcimarl

# CuBLAS workspace config for deterministic behavior
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for seed in ${seeds[@]}; do
    echo "[seed=${seed}] start PCI-QMIX training"
    python main.py --env ${env} --num-agents ${num_agents} \
        --config ${config} --model ${model} \
        --causal-lambda ${causal_lambda} \
        --clip-delta ${clip_delta} \
        --train-naivepredictor ${train_naivepredictor} \
        --n-rollout-threads ${n_rollout_threads} \
        --max-steps ${max_steps_per_iteration} \
        --episodes-per-iteration ${episodes_per_iteration} \
        --max-iterations ${max_iterations} \
        --log-interval ${log_interval} --device ${device} --seed ${seed} \
        --use-wandb --wandb-project pci-marl-staghunt \
        & # deactivate for sequential training
    echo "[seed=${seed}] PID: $!"
done

echo "모든 seed 실행 중... (${#seeds[@]}개 병렬 실행)"
wait
echo "모든 seed 학습 완료!"
