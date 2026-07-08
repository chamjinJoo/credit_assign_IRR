#!/bin/bash
# ============================================================
# Train JAX IPPO, MAPPO, PCI-IPPO, PCI-MAPPO, QMIX, and PCI-QMIX on SMAX scenarios.
#
# Vanilla rows are represented by setting allowed_target_window to the
# full enemy count, which disables the allowed-target restriction.
# ============================================================

set -euo pipefail

use_recurrent_policy=true
seeds=(42 1 2 31 32)
device="cuda"
n_rollout_threads=8
qmix_rollout_threads=16
run_in_background=false

# Scenario tokens accept map:label, map:target_window, or map:target_window:label.
# ALGORITHMS examples: "ippo mappo ipci mpci qmix pci-qmix".
algorithms=(${ALGORITHMS:-ippo mappo})
map_filter=(${MAPS:-})
exclude_maps=(${EXCLUDE_MAPS:-})
scenario_filter=(${SCENARIOS:-})
exclude_scenarios=(${EXCLUDE_SCENARIOS:-})

export CUBLAS_WORKSPACE_CONFIG=:4096:8
# export KMP_DUPLICATE_LIB_OK=TRUE # for Window

scenario_rows=(
    "3m 2 hard"
    "3m 3 vanilla"
    "2s3z 3 hard"
    "2s3z 4 medium"
    "2s3z 5 vanilla"
    "3s5z 6 hard"
    "3s5z 8 vanilla"
    "6h_vs_8z 6 hard"
    "6h_vs_8z 8 vanilla"
    "smacv2_5_units 3 hard"
    "smacv2_5_units 4 medium"
    "smacv2_5_units 5 vanilla"
    "smacv2_10_units 6 hard"
    "smacv2_10_units 8 medium"
    "smacv2_10_units 10 vanilla"
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
    local allowed_target_window="$3"
    local scenario_label="$4"

    case "${token}" in
        "${map_name}:${scenario_label}"|"${map_name}:${allowed_target_window}"|"${map_name}:${allowed_target_window}:${scenario_label}")
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

should_run_scenario() {
    local map_name="$1"
    local allowed_target_window="$2"
    local scenario_label="$3"
    local token

    for token in "${exclude_scenarios[@]}"; do
        if scenario_matches_token "${token}" "${map_name}" "${allowed_target_window}" "${scenario_label}"; then
            return 1
        fi
    done

    if [[ ${#scenario_filter[@]} -eq 0 ]]; then
        return 0
    fi

    for token in "${scenario_filter[@]}"; do
        if scenario_matches_token "${token}" "${map_name}" "${allowed_target_window}" "${scenario_label}"; then
            return 0
        fi
    done

    return 1
}

get_config() {
    local algorithm="$1"

    case "${algorithm}" in
        ippo) echo "configs/smax_configs/smax_ippo.yaml" ;;
        mappo) echo "configs/smax_configs/smax_mappo.yaml" ;;
        ipci) echo "configs/smax_configs/smax_pci_ippo.yaml" ;;
        mpci) echo "configs/smax_configs/smax_pci_mappo.yaml" ;;
        qmix) echo "configs/smax_configs/smax_qmix.yaml" ;;
        pci-qmix|pci_qmix) echo "configs/smax_configs/smax_pci_qmix.yaml" ;;
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
    local algorithm="$1"
    local map_name="$2"
    local allowed_target_window="$3"

    case "${algorithm}:${map_name}:${allowed_target_window}" in
        ippo:3m:2) echo "0.004" ;;
        ippo:3m:3) echo "0.004" ;;
        ippo:2s3z:3) echo "0.004" ;;
        ippo:2s3z:4) echo "0.004" ;;
        ippo:2s3z:5) echo "0.004" ;;
        ippo:3s5z:6) echo "0.004" ;;
        ippo:3s5z:8) echo "0.0005" ;;
        ippo:6h_vs_8z:6) echo "0.002" ;;
        ippo:6h_vs_8z:8) echo "0.001" ;;
        ippo:smacv2_5_units:3) echo "0.004" ;;
        ippo:smacv2_5_units:4) echo "0.004" ;;
        ippo:smacv2_5_units:5) echo "0.002" ;;
        ippo:smacv2_10_units:6) echo "0.001" ;;
        ippo:smacv2_10_units:8) echo "0.001" ;;
        ippo:smacv2_10_units:10) echo "0.001" ;;

        mappo:3m:2) echo "0.002" ;;
        mappo:3m:3) echo "0.002" ;;
        mappo:2s3z:3) echo "0.002" ;;
        mappo:2s3z:4) echo "0.002" ;;
        mappo:2s3z:5) echo "0.002" ;;
        mappo:3s5z:6) echo "0.002" ;;
        mappo:3s5z:8) echo "0.002" ;;
        mappo:6h_vs_8z:6) echo "0.002" ;;
        mappo:6h_vs_8z:8) echo "0.002" ;;
        mappo:smacv2_5_units:3) echo "0.002" ;;
        mappo:smacv2_5_units:4) echo "0.002" ;;
        mappo:smacv2_5_units:5) echo "0.002" ;;
        mappo:smacv2_10_units:6) echo "0.002" ;;
        mappo:smacv2_10_units:8) echo "0.002" ;;
        mappo:smacv2_10_units:10) echo "0.002" ;;

        ipci:3m:2) echo "0.0005" ;;
        ipci:3m:3) echo "0.0005" ;;
        ipci:2s3z:3) echo "0.002" ;;
        ipci:2s3z:4) echo "0.0005" ;;
        ipci:2s3z:5) echo "0.001" ;;
        ipci:3s5z:6) echo "0.004" ;;
        ipci:3s5z:8) echo "0.002" ;;
        ipci:6h_vs_8z:6) echo "0.001" ;;
        ipci:6h_vs_8z:8) echo "0.001" ;;
        ipci:smacv2_5_units:3) echo "0.002" ;;
        ipci:smacv2_5_units:4) echo "0.002" ;;
        ipci:smacv2_5_units:5) echo "0.002" ;;
        ipci:smacv2_10_units:6) echo "0.002" ;;
        ipci:smacv2_10_units:8) echo "0.002" ;;
        ipci:smacv2_10_units:10) echo "0.002" ;;

        mpci:3m:2) echo "0.0005" ;;
        mpci:3m:3) echo "0.0005" ;;
        mpci:2s3z:3) echo "0.001" ;;
        mpci:2s3z:4) echo "0.002" ;;
        mpci:2s3z:5) echo "0.002" ;;
        mpci:3s5z:6) echo "0.002" ;;
        mpci:3s5z:8) echo "0.002" ;;
        mpci:6h_vs_8z:6) echo "0.002" ;;
        mpci:6h_vs_8z:8) echo "0.001" ;;
        mpci:smacv2_5_units:3) echo "0.001" ;;
        mpci:smacv2_5_units:4) echo "0.001" ;;
        mpci:smacv2_5_units:5) echo "0.001" ;;
        mpci:smacv2_10_units:6) echo "0.001" ;;
        mpci:smacv2_10_units:8) echo "0.001" ;;
        mpci:smacv2_10_units:10) echo "0.001" ;;

        qmix:3m:2|pci-qmix:3m:2|pci_qmix:3m:2) echo "0.00005" ;;
        qmix:3m:3|pci-qmix:3m:3|pci_qmix:3m:3) echo "0.00005" ;;
        qmix:2s3z:3|pci-qmix:2s3z:3|pci_qmix:2s3z:3) echo "0.00005" ;;
        qmix:2s3z:4|pci-qmix:2s3z:4|pci_qmix:2s3z:4) echo "0.00005" ;;
        qmix:2s3z:5|pci-qmix:2s3z:5|pci_qmix:2s3z:5) echo "0.00005" ;;
        qmix:3s5z:6|pci-qmix:3s5z:6|pci_qmix:3s5z:6) echo "0.00005" ;;
        qmix:3s5z:8|pci-qmix:3s5z:8|pci_qmix:3s5z:8) echo "0.00005" ;;
        qmix:6h_vs_8z:6|pci-qmix:6h_vs_8z:6|pci_qmix:6h_vs_8z:6) echo "0.00005" ;;
        qmix:6h_vs_8z:8|pci-qmix:6h_vs_8z:8|pci_qmix:6h_vs_8z:8) echo "0.00005" ;;
        qmix:smacv2_5_units:3|pci-qmix:smacv2_5_units:3|pci_qmix:smacv2_5_units:3) echo "0.00005" ;;
        qmix:smacv2_5_units:4|pci-qmix:smacv2_5_units:4|pci_qmix:smacv2_5_units:4) echo "0.00005" ;;
        qmix:smacv2_5_units:5|pci-qmix:smacv2_5_units:5|pci_qmix:smacv2_5_units:5) echo "0.00005" ;;
        qmix:smacv2_10_units:6|pci-qmix:smacv2_10_units:6|pci_qmix:smacv2_10_units:6) echo "0.00005" ;;
        qmix:smacv2_10_units:8|pci-qmix:smacv2_10_units:8|pci_qmix:smacv2_10_units:8) echo "0.00005" ;;
        qmix:smacv2_10_units:10|pci-qmix:smacv2_10_units:10|pci_qmix:smacv2_10_units:10) echo "0.00005" ;;
        *)
            echo "Missing LR for ${algorithm} ${map_name} target=${allowed_target_window}" >&2
            return 1
            ;;
    esac
}

run_training() {
    local algorithm="$1"
    local map_name="$2"
    local allowed_target_window="$3"
    local scenario_label="$4"
    local seed="$5"
    local config
    local model
    local lr
    local lr_args=()
    local lr_delta_critic
    local causal_args=()
    local causal_lambda_value=""
    local rollout_threads="${n_rollout_threads}"

    config="$(get_config "${algorithm}")"
    model="$(get_model "${algorithm}")"
    lr="$(get_lr "${algorithm}" "${map_name}" "${allowed_target_window}")"
    lr_delta_critic="${DELTA_CRITIC_LR:-${lr}}"

    if [[ "${algorithm}" == "ipci" || "${algorithm}" == "mpci" ]]; then
        causal_lambda_value="${PCI_CAUSAL_LAMBDA:-0.1}"
        causal_args=(--causal-lambda "${causal_lambda_value}" --lr-delta-critic "${lr_delta_critic}")
    elif [[ "${algorithm}" == "pci-qmix" || "${algorithm}" == "pci_qmix" ]]; then
        causal_lambda_value="${PCI_QMIX_CAUSAL_LAMBDA:-1.0}"
        causal_args=(--causal-lambda "${causal_lambda_value}" --lr-bridge "${LR_BRIDGE:-0.002}")
    fi
    if [[ "${algorithm}" == "qmix" || "${algorithm}" == "pci-qmix" || "${algorithm}" == "pci_qmix" ]]; then
        lr_args=(--lr "${lr}" --qmix-update-epochs "${QMIX_UPDATE_EPOCHS:-8}")
        rollout_threads="${QMIX_ROLLOUT_THREADS:-${qmix_rollout_threads}}"
    else
        lr_args=(--lr-actor "${lr}" --lr-critic "${lr}")
    fi

    local wandb_group="${map_name}"
    local run_name_postfix="${scenario_label}_lr${lr}"
    if [[ -n "${causal_lambda_value}" ]]; then
        run_name_postfix="${run_name_postfix}_CL${causal_lambda_value}"
    fi
    local wandb_args=(
        --use-wandb
        --wandb-group "${wandb_group}"
        --wandb-run-name-postfix "${run_name_postfix}"
    )

    echo "[${algorithm} map=${map_name} target=${allowed_target_window} ${scenario_label} lr=${lr} delta_lr=${lr_delta_critic} seed=${seed}] start"

    local recurrent_args=()
    if [[ "${algorithm}" != "qmix" && "${algorithm}" != "pci-qmix" && "${algorithm}" != "pci_qmix" ]]; then
        recurrent_args=(--use-recurrent-policy "${use_recurrent_policy}")
    fi

    python main.py --env SMAX \
        --config "${config}" \
        --model "${model}" \
        --map-name "${map_name}" \
        "${recurrent_args[@]}" \
        "${causal_args[@]}" \
        --allowed-target-window "${allowed_target_window}" \
        "${lr_args[@]}" \
        --n-rollout-threads "${rollout_threads}" \
        --device "${device}" \
        --seed "${seed}" \
        "${wandb_args[@]}"
}

for algorithm in "${algorithms[@]}"; do
    for row in "${scenario_rows[@]}"; do
        read -r map_name allowed_target_window scenario_label <<< "${row}"

        if ! should_run_map "${map_name}"; then
            continue
        fi

        if ! should_run_scenario "${map_name}" "${allowed_target_window}" "${scenario_label}"; then
            continue
        fi

        for seed in "${seeds[@]}"; do
            if [[ "${run_in_background}" == true ]]; then
                run_training "${algorithm}" "${map_name}" "${allowed_target_window}" "${scenario_label}" "${seed}" &
            else
                run_training "${algorithm}" "${map_name}" "${allowed_target_window}" "${scenario_label}" "${seed}"
            fi
        done
    done
done

if [[ "${run_in_background}" == true ]]; then
    wait
fi
