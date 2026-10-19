#!/usr/bin/env bash
set -euo pipefail

# Submit outcome-bridge cost jobs through the repo's sync-aware Slurm wrapper.
# Each Slurm job runs one env x backbone x variant condition for the requested seeds.

ENVS=(${COST_ENVS:-staghunt smax lbf})
BACKBONES=(${COST_BACKBONES:-ippo mappo})
VARIANTS=(${COST_VARIANTS:-vanilla pci})
SEEDS=${COST_SEEDS:-42}
MAX_ITERATIONS=${COST_MAX_ITERATIONS:-}
MAX_STEPS=${COST_MAX_STEPS:-}
N_ROLLOUT_THREADS=${COST_N_ROLLOUT_THREADS:-8}
LOG_INTERVAL=${COST_LOG_INTERVAL:-2}
DEVICE=${COST_DEVICE:-cuda}
WANDB_PROJECT=${COST_WANDB_PROJECT:-pci-marl-bridge-cost}
USE_WANDB=${COST_USE_WANDB:-true}
SBATCH_TEMPLATE=${COST_SBATCH_TEMPLATE:-slurms/templates/slurm_train_staghunt.sbatch}
WRAPPER=${COST_SBATCH_WRAPPER:-scripts/sbatch_after_sync_job_outputs.sh}
SBATCH_EXCLUSIVE=${COST_SBATCH_EXCLUSIVE:-true}
SBATCH_EXTRA_ARGS=(${COST_SBATCH_EXTRA_ARGS:-})

for env_key in "${ENVS[@]}"; do
    for backbone in "${BACKBONES[@]}"; do
        for variant in "${VARIANTS[@]}"; do
            for seed in ${SEEDS}; do
                job_name="cost_${env_key}_${variant}_${backbone}_s${seed}"
                export_vars=(
                    "COST_ENVS=${env_key}"
                    "COST_BACKBONES=${backbone}"
                    "COST_VARIANTS=${variant}"
                    "COST_SEEDS=${seed}"
                    "COST_N_ROLLOUT_THREADS=${N_ROLLOUT_THREADS}"
                    "COST_LOG_INTERVAL=${LOG_INTERVAL}"
                    "COST_DEVICE=${DEVICE}"
                    "COST_USE_WANDB=${USE_WANDB}"
                    "COST_WANDB_PROJECT=${WANDB_PROJECT}"
                )
                if [[ -n "${MAX_ITERATIONS}" ]]; then
                    export_vars+=("COST_MAX_ITERATIONS=${MAX_ITERATIONS}")
                fi
                if [[ -n "${MAX_STEPS}" ]]; then
                    export_vars+=("COST_MAX_STEPS=${MAX_STEPS}")
                fi
                export_arg="ALL"
                for item in "${export_vars[@]}"; do
                    export_arg+=",${item}"
                done

                sbatch_args=(--job-name "${job_name}" --export "${export_arg}")
                if [[ ${#SBATCH_EXTRA_ARGS[@]} -gt 0 ]]; then
                    sbatch_args+=("${SBATCH_EXTRA_ARGS[@]}")
                fi
                if [[ "${SBATCH_EXCLUSIVE}" == "true" ]]; then
                    sbatch_args+=(--exclusive)
                fi

                echo "[submit] ${job_name} seed=${seed} exclusive=${SBATCH_EXCLUSIVE}"
                "${WRAPPER}" \
                    -- \
                    "${sbatch_args[@]}" \
                    "${SBATCH_TEMPLATE}" \
                    "${job_name}" \
                    "bash scripts/cost/run_bridge_cost_sweep.sh"
            done
        done
    done
done
