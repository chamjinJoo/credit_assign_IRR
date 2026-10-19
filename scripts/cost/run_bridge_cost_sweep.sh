#!/usr/bin/env bash
set -euo pipefail

# Outcome bridge computational-cost sweep.
# Runs StagHunt, SMAX smacv2_5_units, and a 3-agent JaxLBF setting on GPU.
# Conditions: env x backbone(ippo,mappo) x variant(vanilla,pci).

ENVS=(${COST_ENVS:-staghunt smax lbf})
BACKBONES=(${COST_BACKBONES:-ippo mappo})
VARIANTS=(${COST_VARIANTS:-vanilla pci})
SEEDS=(${COST_SEEDS:-42})

DEVICE=${COST_DEVICE:-cuda}
N_ROLLOUT_THREADS=${COST_N_ROLLOUT_THREADS:-8}
MAX_ITERATIONS=${COST_MAX_ITERATIONS:-}
MAX_STEPS=${COST_MAX_STEPS:-}
LOG_INTERVAL=${COST_LOG_INTERVAL:-2}
WANDB_PROJECT=${COST_WANDB_PROJECT:-pci-marl-bridge-cost}
USE_WANDB=${COST_USE_WANDB:-true}

export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}

common_args=()
if [[ -n "${MAX_ITERATIONS}" ]]; then
    common_args+=(--max-iterations "${MAX_ITERATIONS}")
fi
if [[ -n "${MAX_STEPS}" ]]; then
    common_args+=(--max-steps "${MAX_STEPS}")
fi
if [[ -n "${LOG_INTERVAL}" ]]; then
    common_args+=(--log-interval "${LOG_INTERVAL}")
fi
if [[ "${USE_WANDB}" == "true" ]]; then
    common_args+=(--use-wandb --wandb-project "${WANDB_PROJECT}")
fi

run_one() {
    local env_key="$1"
    local backbone="$2"
    local variant="$3"
    local seed="$4"

    local env_name config model centralized
    local extra_args=()

    case "${env_key}:${variant}:${backbone}" in
        staghunt:vanilla:ippo)
            env_name="StagHunt"
            config="configs/staghunt_configs/staghunt_ippo.yaml"
            model="r-ippo"
            centralized="false"
            extra_args+=(--num-agents 3 --recurrent-N 1)
            ;;
        staghunt:vanilla:mappo)
            env_name="StagHunt"
            config="configs/staghunt_configs/staghunt_mappo.yaml"
            model="r-mappo"
            centralized="true"
            extra_args+=(--num-agents 3 --recurrent-N 1)
            ;;
        staghunt:pci:ippo)
            env_name="StagHunt"
            config="configs/staghunt_configs/staghunt_pci_ippo.yaml"
            model="r-pci"
            centralized="false"
            extra_args+=(--num-agents 3 --recurrent-N 1 --causal-lambda 1.0)
            ;;
        staghunt:pci:mappo)
            env_name="StagHunt"
            config="configs/staghunt_configs/staghunt_pci_mappo.yaml"
            model="r-pci"
            centralized="true"
            extra_args+=(--num-agents 3 --recurrent-N 1 --causal-lambda 1.0)
            ;;
        smax:vanilla:ippo)
            env_name="SMAX"
            config="configs/smax_configs/smax_ippo.yaml"
            model="jax-ippo"
            centralized="false"
            extra_args+=(--map-name smacv2_5_units --allowed-target-window 3 --use-recurrent-policy true)
            ;;
        smax:vanilla:mappo)
            env_name="SMAX"
            config="configs/smax_configs/smax_mappo.yaml"
            model="jax-mappo"
            centralized="true"
            extra_args+=(--map-name smacv2_5_units --allowed-target-window 3 --use-recurrent-policy true)
            ;;
        smax:pci:ippo)
            env_name="SMAX"
            config="configs/smax_configs/smax_pci_ippo.yaml"
            model="jax-pci-ippo"
            centralized="false"
            extra_args+=(--map-name smacv2_5_units --allowed-target-window 3 --causal-lambda 1.0)
            ;;
        smax:pci:mappo)
            env_name="SMAX"
            config="configs/smax_configs/smax_pci_mappo.yaml"
            model="jax-pci-mappo"
            centralized="true"
            extra_args+=(--map-name smacv2_5_units --allowed-target-window 3 --causal-lambda 1.0)
            ;;
        lbf:vanilla:ippo)
            env_name="JaxLBF"
            config="configs/lbf_configs/lbf_ippo.yaml"
            model="jax-ippo"
            centralized="false"
            extra_args+=(--num-agents 3 --field-size 10 --max-food 3 --sight 2 --max-episode-steps 50 --force-coop false --use-recurrent-policy true)
            ;;
        lbf:vanilla:mappo)
            env_name="JaxLBF"
            config="configs/lbf_configs/lbf_mappo.yaml"
            model="jax-mappo"
            centralized="true"
            extra_args+=(--num-agents 3 --field-size 10 --max-food 3 --sight 2 --max-episode-steps 50 --force-coop false --use-recurrent-policy true)
            ;;
        lbf:pci:ippo)
            env_name="JaxLBF"
            config="configs/lbf_configs/lbf_pci_ippo.yaml"
            model="jax-pci-ippo"
            centralized="false"
            extra_args+=(--num-agents 3 --field-size 10 --max-food 3 --sight 2 --max-episode-steps 50 --force-coop false --causal-lambda 1.0)
            ;;
        lbf:pci:mappo)
            env_name="JaxLBF"
            config="configs/lbf_configs/lbf_pci_mappo.yaml"
            model="jax-pci-mappo"
            centralized="true"
            extra_args+=(--num-agents 3 --field-size 10 --max-food 3 --sight 2 --max-episode-steps 50 --force-coop false --causal-lambda 1.0)
            ;;
        *)
            echo "Unsupported condition: env=${env_key} variant=${variant} backbone=${backbone}" >&2
            return 2
            ;;
    esac

    echo "[cost] env=${env_key} variant=${variant} backbone=${backbone} seed=${seed}"
    echo "[cost_meta] env=${env_key}"
    echo "[cost_meta] variant=${variant}"
    echo "[cost_meta] backbone=${backbone}"
    echo "[cost_meta] seed=${seed}"
    echo "[cost_meta] env_name=${env_name}"
    echo "[cost_meta] model=${model}"
    echo "[cost_meta] config=${config}"
    echo "[cost_meta] centralized_critic=${centralized}"
    echo "[cost_meta] n_rollout_threads=${N_ROLLOUT_THREADS}"
    echo "[cost_meta] max_iterations=${MAX_ITERATIONS:-config_default}"
    echo "[cost_meta] max_steps=${MAX_STEPS:-config_default}"
    echo "[cost_meta] log_interval=${LOG_INTERVAL}"
    echo "[cost_meta] device=${DEVICE}"
    echo "[cost_meta] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader | sed 's/^/[cost_gpu] /' || true
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits | sed 's/^/[cost_gpu_process] /' || true
    else
        echo "[cost_gpu] nvidia-smi unavailable"
    fi
    python - <<'PYINFO'
try:
    import torch
    print(f"[cost_cuda] torch_cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[cost_cuda] torch_device_count={torch.cuda.device_count()}")
        print(f"[cost_cuda] torch_device_name={torch.cuda.get_device_name(0)}")
except Exception as exc:
    print(f"[cost_cuda] torch_probe_error={exc}")
try:
    import jax
    print(f"[cost_cuda] jax_devices={jax.devices()}")
except Exception as exc:
    print(f"[cost_cuda] jax_probe_error={exc}")
PYINFO
    python main.py \
        --env "${env_name}" \
        --config "${config}" \
        --model "${model}" \
        --use-centralized-critic "${centralized}" \
        --n-rollout-threads "${N_ROLLOUT_THREADS}" \
        --device "${DEVICE}" \
        --seed "${seed}" \
        "${extra_args[@]}" \
        "${common_args[@]}"
}

for env_key in "${ENVS[@]}"; do
    for backbone in "${BACKBONES[@]}"; do
        for variant in "${VARIANTS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                run_one "${env_key}" "${backbone}" "${variant}" "${seed}"
            done
        done
    done
done
