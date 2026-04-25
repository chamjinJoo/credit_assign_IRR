#!/bin/bash
# ============================================================
# Draft recurrent JAX PCI launcher for SMAX
# ============================================================

set -euo pipefail

config="configs_jax/smax_pci_ippo.yaml"

# rpci settings
causal_lambda=1.0
clip_delta=0.0
model="jax-pci-ippo"

# env settings
map_name="3m"  # empty string keeps the YAML value; set to override the SMAX map
# maps: [3m, 2s3z, 3s_vs_5z, 3s5z_vs_3s6z, 5m_vs_6m, 3s5z, 6h_vs_8z, 10m_vs_11m, smacv2_5_units, smacv2_10_units]
allowed_target_windows=(3)

# training settings
seeds=(42)
device="cuda"
n_rollout_threads=16
max_iterations=1600
max_steps_per_iteration=6400 # follow the smax paper: 128 env * 64 steps
log_interval=2
run_in_background=false
wandb_run_name_postfix=""

#checkpoint="save_checkpoints/20260412_134339_seed42/smaxbatchwrapper/checkpoints/jax_checkpoint_iter_162.pkl"

# conda activation (optional)
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate pcimarl
fi

# reproducibility settings for JAX
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# export KMP_DUPLICATE_LIB_OK=TRUE # for Window

pids=()

# main training loop over config variants and seeds
for allowed_target_window in "${allowed_target_windows[@]}"; do
  echo "[allowed_target_window=${allowed_target_window}] start"
  for seed in "${seeds[@]}"; do
      wandb_group="smax_${map_name:-default}"
      wandb_args=(
        --use-wandb
        --wandb-project pci-marl-smax-jax-v3
        --wandb-group "${wandb_group}"
      )
      if [[ -n "${wandb_run_name_postfix}" ]]; then
        wandb_args+=(--wandb-run-name-postfix "${wandb_run_name_postfix}")
      fi
      echo "[allowed_target_window=${allowed_target_window} seed=${seed}] start"
      if ${run_in_background}; then
        python main.py \
        --env SMAX \
        --config ${config} \
        --model ${model} \
        --map-name "${map_name}" \
        --causal-lambda ${causal_lambda} \
        --allowed-target-window ${allowed_target_window} \
        --n-rollout-threads ${n_rollout_threads} \
        --max-iterations ${max_iterations} \
        --max-steps ${max_steps_per_iteration} \
        --log-interval ${log_interval} \
        --device ${device} \
        --seed ${seed} \
        "${wandb_args[@]}" &
        pid=$!
        pids+=("${pid}")
        echo "[allowed_target_window=${allowed_target_window} seed=${seed}] PID: ${pid}"
      else
        python main.py \
        --env SMAX \
        --config ${config} \
        --model ${model} \
        --map-name "${map_name}" \
        --causal-lambda ${causal_lambda} \
        --allowed-target-window ${allowed_target_window} \
        --n-rollout-threads ${n_rollout_threads} \
        --max-iterations ${max_iterations} \
        --max-steps ${max_steps_per_iteration} \
        --log-interval ${log_interval} \
        --device ${device} \
        --seed ${seed} \
        "${wandb_args[@]}"
    fi
  done
done

if ${run_in_background}; then
  echo "all seed launched. waiting for completion..."
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
  echo "all config/seed training completed!"
else
  echo "all config/seed training completed sequentially!"
fi
