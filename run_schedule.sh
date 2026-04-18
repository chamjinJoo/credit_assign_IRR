#!/bin/bash
set -e

cd /absolute/path/to/your/project

# conda 안정적으로 로드
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pcimarl

mkdir -p logs

# echo "[START MAPPO] $(date)"
# bash scripts_jax/smax/train_jax_mappo.sh 

echo "[START IPPO] $(date)"
bash scripts_jax/smax/train_jax_ippo.sh

echo "[DONE] $(date)"