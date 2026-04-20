#!/bin/bash
# Wait for the current MAPPO launcher to finish, then rerun MAPPO with
# SMAX allowed-target masking disabled.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <watch_pid> [log_path]" >&2
  exit 2
fi

watch_pid="$1"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
config="configs_jax/smax_mappo.yaml"
launcher="scripts_jax/smax/train_jax_mappo.sh"
log_path="${2:-/tmp/smax_mappo_allowed_targets_false_after_${watch_pid}.log}"

cd "${repo_root}"

{
  echo "[$(date -Is)] watcher started; waiting for PID ${watch_pid}"
  while kill -0 "${watch_pid}" 2>/dev/null; do
    sleep 60
  done

  echo "[$(date -Is)] PID ${watch_pid} ended; setting use_allowed_targets=false"
  sed -i -E 's/^([[:space:]]*use_allowed_targets:)[[:space:]]*(true|false)[[:space:]]*$/\1 false/' "${config}"

  if ! rg -q '^[[:space:]]*use_allowed_targets:[[:space:]]*false[[:space:]]*$' "${config}"; then
    echo "[$(date -Is)] failed to verify use_allowed_targets=false in ${config}" >&2
    exit 1
  fi

  rg -n 'use_allowed_targets' "${config}"
  echo "[$(date -Is)] launching ${launcher}"
  exec bash "${launcher}"
} >> "${log_path}" 2>&1
