#!/bin/bash
set -euo pipefail

# Train all active Torch models on Vote.
algorithms=(${ALGORITHMS:-ippo mappo pci_ippo qmix pci_qmix})
run_in_background=false

for algorithm in "${algorithms[@]}"; do
    script="scripts/vote/train_${algorithm}.sh"
    if [ ! -f "${script}" ]; then
        echo "Unknown Vote algorithm: ${algorithm}" >&2
        exit 1
    fi

    echo "[${algorithm}] start"
    if [ "${run_in_background}" = "true" ]; then
        bash "${script}" &
    else
        bash "${script}"
    fi
done

if [ "${run_in_background}" = "true" ]; then
    wait
fi
