#!/bin/bash
# ============================================================
# Train JAX IPPO, MAPPO, PCI-IPPO, PCI-MAPPO, QMIX, and PCI-QMIX
# on LBForaging candidate maps.
# ============================================================

set -euo pipefail

use_recurrent_policy=true
seeds=(42 1 2 31 32)
device="cuda"
n_rollout_threads=16
qmix_rollout_threads=16
log_interval=2
run_in_background=false

# ALGORITHMS examples: "ippo mappo ipci mpci qmix pci-qmix".
algorithms=(${ALGORITHMS:-ippo mappo})
map_filter=(${MAPS:-})
exclude_maps=(${EXCLUDE_MAPS:-})
scenario_filter=(${SCENARIOS:-})
exclude_scenarios=(${EXCLUDE_SCENARIOS:-})

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# label sight field_size num_agents max_food force_coop
map_rows=(
    "2s-8x8-2p-2f-coop 2 8 2 2 true"
    "2s-10x10-3p-3f 2 10 3 3 false"
    "2s-15x15-3p-3f 2 15 3 3 false"
    "2s-15x15-4p-3f 2 15 4 3 false"
    "2s-15x15-4p-5f 2 15 4 5 false"
)

should_run_map() {
    local map_name="$1"
    local excluded

    for excluded in "${exclude_maps[@]}"; do
        if [[ "${excluded}" == "${map_name}" ]]; then
            return 1
        fi
    done

    if [[ ${#map_filter[@]} -eq 0 ]]; then
        return 0
    fi

    local wanted
    for wanted in "${map_filter[@]}"; do
        if [[ "${wanted}" == "${map_name}" ]]; then
            return 0
        fi
    done

    return 1
}

scenario_matches_token() {
    local token="$1"
    local map_name="$2"

    case "${token}" in
        "${map_name}")
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

should_run_scenario() {
    local map_name="$1"
    local token

    for token in "${exclude_scenarios[@]}"; do
        if scenario_matches_token "${token}" "${map_name}"; then
            return 1
        fi
    done

    if [[ ${#scenario_filter[@]} -eq 0 ]]; then
        return 0
    fi

    for token in "${scenario_filter[@]}"; do
        if scenario_matches_token "${token}" "${map_name}"; then
            return 0
        fi
    done

    return 1
}

get_config() {
    local algorithm="$1"

    case "${algorithm}" in
        ippo) echo "configs/lbf_configs/lbf_ippo.yaml" ;;
        mappo) echo "configs/lbf_configs/lbf_mappo.yaml" ;;
        ipci) echo "configs/lbf_configs/lbf_pci_ippo.yaml" ;;
        mpci) echo "configs/lbf_configs/lbf_pci_mappo.yaml" ;;
        qmix) echo "configs/lbf_configs/lbf_qmix.yaml" ;;
        pci-qmix|pci_qmix) echo "configs/lbf_configs/lbf_pci_qmix.yaml" ;;
        *)
            echo "Unknown algorithm: ${algorithm}" >&2
            return 1
            ;;
    esac
}

get_model() {
    local algorithm="$1"

    case "${algorithm}" in
        ippo) echo "jax-ippo" ;;
        mappo) echo "jax-mappo" ;;
        ipci) echo "jax-pci-ippo" ;;
        mpci) echo "jax-pci-mappo" ;;
        qmix) echo "jax-qmix" ;;
        pci-qmix|pci_qmix) echo "jax-pci-qmix" ;;
        *)
            echo "Unknown algorithm: ${algorithm}" >&2
            return 1
            ;;
    esac
}

get_lr() {
    echo "${LR:-0.0003}"
}

get_config_lr() {
    local config="$1"

    awk '
        /^[[:space:]]*network:[[:space:]]*$/ { in_network=1; next }
        /^[^[:space:]][^:]*:[[:space:]]*$/ && in_network { exit }
        in_network && /^[[:space:]]*lr:[[:space:]]*/ {
            value=$0
            sub(/^[[:space:]]*lr:[[:space:]]*/, "", value)
            sub(/[[:space:]]*#.*/, "", value)
            gsub(/[[:space:]]+$/, "", value)
            print value
            exit
        }
    ' "${config}"
}

is_qmix_algorithm() {
    local algorithm="$1"

    [[ "${algorithm}" == "qmix" || "${algorithm}" == "pci-qmix" || "${algorithm}" == "pci_qmix" ]]
}

run_training() {
    local algorithm="$1"
    local map_name="$2"
    local sight="$3"
    local field_size="$4"
    local num_agents="$5"
    local max_food="$6"
    local force_coop="$7"
    local seed="$8"
    local config
    local model
    local lr=""
    local lr_args=()
    local causal_args=()
    local causal_lambda_value=""
    local rollout_threads="${n_rollout_threads}"

    config="$(get_config "${algorithm}")"
    model="$(get_model "${algorithm}")"

    if [[ "${algorithm}" == "ipci" || "${algorithm}" == "mpci" ]]; then
        causal_lambda_value="${PCI_CAUSAL_LAMBDA:-1.0}"
        causal_args=(--causal-lambda "${causal_lambda_value}")
    elif [[ "${algorithm}" == "pci-qmix" || "${algorithm}" == "pci_qmix" ]]; then
        causal_lambda_value="${PCI_QMIX_CAUSAL_LAMBDA:-1.0}"
        causal_args=(--causal-lambda "${causal_lambda_value}")
    fi

    if is_qmix_algorithm "${algorithm}"; then
        lr="$(get_config_lr "${config}")"
        lr_args=(--qmix-update-epochs "${QMIX_UPDATE_EPOCHS:-8}")
        rollout_threads="${QMIX_ROLLOUT_THREADS:-${qmix_rollout_threads}}"
    else
        lr="$(get_lr)"
        lr_args=(--lr-actor "${lr}" --lr-critic "${lr}")
    fi

    local run_name_postfix="config_lr"
    if [[ -n "${lr}" ]]; then
        run_name_postfix="lr${lr}"
    fi
    if [[ -n "${causal_lambda_value}" ]]; then
        run_name_postfix="${run_name_postfix}_CL${causal_lambda_value}"
    fi
    local wandb_args=(
        --use-wandb
        --wandb-project pci-marl-lbf
        --wandb-group "${map_name}"
        --wandb-run-name-postfix "${run_name_postfix}"
    )

    local recurrent_args=()
    if ! is_qmix_algorithm "${algorithm}"; then
        recurrent_args=(--use-recurrent-policy "${use_recurrent_policy}")
    fi

    if [[ -n "${lr}" ]]; then
        echo "[${algorithm} map=${map_name} lr=${lr} seed=${seed}] start"
    else
        echo "[${algorithm} map=${map_name} lr=config seed=${seed}] start"
    fi

    python main.py --env JaxLBF \
        --config "${config}" \
        --model "${model}" \
        --sight "${sight}" \
        --field-size "${field_size}" \
        --num-agents "${num_agents}" \
        --max-food "${max_food}" \
        --force-coop "${force_coop}" \
        --max-episode-steps "${MAX_EPISODE_STEPS:-50}" \
        --grid-observation true \
        "${recurrent_args[@]}" \
        "${causal_args[@]}" \
        "${lr_args[@]}" \
        --n-rollout-threads "${rollout_threads}" \
        --log-interval "${LOG_INTERVAL:-${log_interval}}" \
        --device "${device}" \
        --seed "${seed}" \
        "${wandb_args[@]}"
}

for algorithm in "${algorithms[@]}"; do
    for row in "${map_rows[@]}"; do
        read -r map_name sight field_size num_agents max_food force_coop <<< "${row}"

        if ! should_run_map "${map_name}"; then
            continue
        fi

        if ! should_run_scenario "${map_name}"; then
            continue
        fi

        for seed in "${seeds[@]}"; do
            if [[ "${run_in_background}" == true ]]; then
                run_training "${algorithm}" "${map_name}" "${sight}" "${field_size}" "${num_agents}" "${max_food}" "${force_coop}" "${seed}" &
            else
                run_training "${algorithm}" "${map_name}" "${sight}" "${field_size}" "${num_agents}" "${max_food}" "${force_coop}" "${seed}"
            fi
        done
    done
done

if [[ "${run_in_background}" == true ]]; then
    wait
fi
