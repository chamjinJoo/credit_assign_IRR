#!/usr/bin/env bash
# Repo-local sbatch_after_sync variant: post-job sync fetches only slurms/job_outputs/<job_id>/.

set -euo pipefail

CLUSTER_SYNC_LIB_DIR="${CLUSTER_SYNC_LIB_DIR:-/usr/local/lib/cluster_sync}"
# shellcheck source=/usr/local/lib/cluster_sync/sync_common.sh
source "$CLUSTER_SYNC_LIB_DIR/sync_common.sh"

usage() {
  cat <<USAGE
Usage: scripts/sbatch_after_sync_job_outputs.sh [--safe-sync] [--foreground] [--poll-interval SEC] [--] SBATCH_ARGS...

Run sync_to_head, submit a Slurm job with sbatch, then fetch only
slurms/job_outputs/<jobid>/ after the job leaves the Slurm queue. Default
submit mode detaches: a background watcher waits for the job, syncs back that
bundle directory, and removes .sync_lock_<jobid> from the target
folder. Active .sync_lock_* files make later sync_to_head runs omit --delete and
cancel if bml-head has newer files.

Wrapper options:
  --safe-sync         Use sync_to_head --safe before submitting.
  --sync-force        Kept for compatibility; force sync is now the default.
  --foreground        Do not detach; run sbatch --wait, then sync back here.
  --detach            Explicitly request the default detached behavior.
  --poll-interval SEC Detached watcher squeue polling interval. Default: 30.
  -h, --help          Show this help.

Examples:
  scripts/sbatch_after_sync_job_outputs.sh run.sbatch
  scripts/sbatch_after_sync_job_outputs.sh -- --partition=gpu run.sbatch
  scripts/sbatch_after_sync_job_outputs.sh --foreground run.sbatch
USAGE
}

json_escape_file() {
  local file="$1"
  python3 -c 'import json, pathlib, sys; print(json.dumps(pathlib.Path(sys.argv[1]).read_text(errors="replace")))' "$file"
}

write_sync_fail_json() {
  local root="$1"
  local job_id="$2"
  local status="$3"
  local mode="$4"
  local stdout_file="${5:-}"
  local stderr_file="${6:-}"
  mkdir -p "$root" 2>/dev/null || true
  python3 - "$root" "$job_id" "$status" "$mode" "$stdout_file" "$stderr_file" <<'PYJSON'
import json
import pathlib
import sys
from datetime import datetime, timezone

root, job_id, status, mode, stdout_file, stderr_file = sys.argv[1:]
ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')

def read_text(path):
    if not path:
        return ''
    p = pathlib.Path(path)
    if not p.exists():
        return ''
    return p.read_text(errors='replace')

payload = {
    'time': ts,
    'job_id': job_id,
    'root': root,
    'mode': mode,
    'sync_status': int(status),
    'message': 'sync_from_head failed',
    'sync_stdout': read_text(stdout_file),
    'sync_stderr': read_text(stderr_file),
}
out = pathlib.Path(root) / f'sync_fail_{ts}.json'
out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
print(f'sync_from_head failed; wrote {out}', file=sys.stderr)
PYJSON
}

safe_sync=0
detach=1
poll_interval=30

while (($#)); do
  case "$1" in
    --safe-sync)
      safe_sync=1
      shift
      ;;
    --sync-force)
      safe_sync=0
      shift
      ;;
    --foreground)
      detach=0
      shift
      ;;
    --detach|--no-wait)
      detach=1
      shift
      ;;
    --poll-interval)
      shift
      if (($# == 0)) || [[ ! "$1" =~ ^[0-9]+$ ]] || ((10#$1 < 1)); then
        echo "error: --poll-interval requires a positive integer" >&2
        exit 2
      fi
      poll_interval="$1"
      shift
      ;;
    --poll-interval=*)
      poll_interval="${1#*=}"
      if [[ ! "$poll_interval" =~ ^[0-9]+$ ]] || ((10#$poll_interval < 1)); then
        echo "error: --poll-interval requires a positive integer" >&2
        exit 2
      fi
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if (($# == 0)); then
  echo "error: missing sbatch arguments" >&2
  usage >&2
  exit 2
fi

command -v sbatch >/dev/null 2>&1 || {
  echo "error: sbatch is not installed or not in PATH" >&2
  exit 127
}
command -v squeue >/dev/null 2>&1 || {
  echo "error: squeue is not installed or not in PATH" >&2
  exit 127
}
command -v python3 >/dev/null 2>&1 || {
  echo "error: python3 is required to write sync_fail JSON" >&2
  exit 127
}

root="$(sync_abs_dir .)"
pending_lock=".sync_lock_pending_$(hostname)_$$"
job_lock=""
created_lock=0
cleanup_job_lock() {
  if ((created_lock == 1)); then
    if [[ -n "$job_lock" ]]; then
      sync_remote_ssh "rm -f '$root/$job_lock'" >/dev/null 2>&1 || true
    else
      sync_remote_ssh "rm -f '$root/$pending_lock'" >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup_job_lock EXIT

sync_to_args=()
if ((safe_sync == 1)); then
  sync_to_args+=(--safe)
fi

"$CLUSTER_SYNC_LIB_DIR/sync_to_head.sh" "${sync_to_args[@]}"

sync_remote_ssh "mkdir -p '$root' && printf 'status=pending\\nhost=%s\\nroot=%s\\ntime=%s\\n' '$(hostname)' '$root' '$(date -Is)' > '$root/$pending_lock'"
created_lock=1

sbatch_args=("$@")

if ((detach == 0)); then
  has_wait=0
  for arg in "${sbatch_args[@]}"; do
    if [[ "$arg" == "--wait" ]]; then
      has_wait=1
      break
    fi
  done
  if ((has_wait == 0)); then
    sbatch_args=(--wait "${sbatch_args[@]}")
  fi

  set +e
  sbatch_output="$(sbatch "${sbatch_args[@]}" 2>&1)"
  sbatch_status=$?
  set -e
  echo "$sbatch_output"
  job_id="$(sed -n 's/^Submitted batch job //p' <<< "$sbatch_output" | tail -n 1)"
  if ((sbatch_status != 0)); then
    exit "$sbatch_status"
  fi
  if [[ -n "$job_id" ]]; then
    job_lock=".sync_lock_${job_id}"
    sync_remote_ssh "mv -f '$root/$pending_lock' '$root/$job_lock' && printf 'job_id=%s\\nhost=%s\\nroot=%s\\ntime=%s\\nmode=foreground\\n' '$job_id' '$(hostname)' '$root' '$(date -Is)' > '$root/$job_lock'" >/dev/null || true
  fi
  sync_stdout="$(mktemp)"
  sync_stderr="$(mktemp)"
  sync_target="$root/slurms/job_outputs/${job_id:-unknown}"
  mkdir -p "$sync_target"
  echo "sync_from_head target: $sync_target"
  set +e
  "$CLUSTER_SYNC_LIB_DIR/sync_from_head.sh" "$sync_target" > >(tee "$sync_stdout") 2> >(tee "$sync_stderr" >&2)
  sync_status=$?
  set -e
  if ((sync_status != 0)); then
    write_sync_fail_json "$root" "${job_id:-unknown}" "$sync_status" "foreground" "$sync_stdout" "$sync_stderr"
    rm -f "$sync_stdout" "$sync_stderr"
    exit "$sync_status"
  fi
  rm -f "$sync_stdout" "$sync_stderr"
  exit "$sbatch_status"
fi

for arg in "${sbatch_args[@]}"; do
  if [[ "$arg" == "--wait" ]]; then
    echo "error: do not pass sbatch --wait in detached mode; use sbatch_after_sync --foreground instead." >&2
    exit 2
  fi
done

has_parsable=0
for arg in "${sbatch_args[@]}"; do
  if [[ "$arg" == "--parsable" ]]; then
    has_parsable=1
    break
  fi
done
if ((has_parsable == 0)); then
  sbatch_args=(--parsable "${sbatch_args[@]}")
fi

submit_output="$(sbatch "${sbatch_args[@]}")"
job_id="${submit_output%%;*}"
job_id="${job_id%%$'\n'*}"
if [[ -z "$job_id" ]]; then
  echo "error: could not parse sbatch job id from: ${submit_output}" >&2
  exit 1
fi

job_lock=".sync_lock_${job_id}"
sync_remote_ssh "mv -f '$root/$pending_lock' '$root/$job_lock' && printf 'job_id=%s\\nhost=%s\\nroot=%s\\ntime=%s\\nmode=detached\\n' '$job_id' '$(hostname)' '$root' '$(date -Is)' > '$root/$job_lock'"

job_dir="${SYNC_TMP_ROOT:-/tmp/cluster_sync/${USER:-$(id -un)}}/jobs"
mkdir -p "$job_dir"
lock_id="$(sync_lock_hash "$root")"
watcher_script="$job_dir/watch_${job_id}_${lock_id}.sh"
log_file="$job_dir/watch_${job_id}_${lock_id}.log"
state_file="$job_dir/watch_${job_id}_${lock_id}.squeue"

cat > "$watcher_script" <<'WATCHER'
#!/usr/bin/env bash
# Repo-local sbatch_after_sync variant: post-job sync fetches only slurms/job_outputs/<job_id>/.
set -u

job_id="$1"
root="$2"
cluster_sync_lib_dir="$3"
head_host="$4"
sync_rsh="$5"
poll_interval="$6"
state_file="$7"
job_lock="$8"
log_file="$9"

remote_ssh() {
  # Intentionally split sync_rsh into command + args.
  # shellcheck disable=SC2086
  $sync_rsh "$head_host" "$@"
}

json_escape_file() {
  local file="$1"
  python3 -c 'import json, pathlib, sys; print(json.dumps(pathlib.Path(sys.argv[1]).read_text(errors="replace")))' "$file"
}

write_sync_fail_json() {
  local status="$1"
  local stdout_file="${2:-}"
  local stderr_file="${3:-}"
  mkdir -p "$root" 2>/dev/null || true
  python3 - "$root" "$job_id" "$status" "detached" "$stdout_file" "$stderr_file" <<'PYJSON'
import json
import pathlib
import sys
from datetime import datetime, timezone

root, job_id, status, mode, stdout_file, stderr_file = sys.argv[1:]
ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')

def read_text(path):
    if not path:
        return ''
    p = pathlib.Path(path)
    if not p.exists():
        return ''
    return p.read_text(errors='replace')

payload = {
    'time': ts,
    'job_id': job_id,
    'root': root,
    'mode': mode,
    'sync_status': int(status),
    'message': 'sync_from_head failed',
    'sync_stdout': read_text(stdout_file),
    'sync_stderr': read_text(stderr_file),
}
out = pathlib.Path(root) / f'sync_fail_{ts}.json'
out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
print(f'[{datetime.now().astimezone().isoformat(timespec="seconds")}] sync_from_head failed; wrote {out}')
PYJSON
}

cleanup() {
  remote_ssh "rm -f '$root/$job_lock'" >/dev/null 2>&1 || true
  rm -f "$state_file" "$log_file" "$sync_stdout" "$sync_stderr" >/dev/null 2>&1 || true
}
trap cleanup EXIT

sync_stdout=""
sync_stderr=""
printf '[%s] watcher started for job %s in %s\n' "$(date -Is)" "$job_id" "$root"

while true; do
  if squeue -h -j "$job_id" > "$state_file" 2>&1; then
    if [[ ! -s "$state_file" ]]; then
      break
    fi
  else
    printf '[%s] warning: squeue failed while waiting for job %s; retrying\n' "$(date -Is)" "$job_id"
  fi
  sleep "$poll_interval"
done

printf '[%s] job %s left Slurm queue; syncing job output bundle only\n' "$(date -Is)" "$job_id"
mkdir -p "$root" 2>/dev/null || true
cd "$root" || { write_sync_fail_json 1; exit 1; }
sync_target="$root/slurms/job_outputs/$job_id"
mkdir -p "$sync_target" 2>/dev/null || true
printf '[%s] sync_from_head target: %s\n' "$(date -Is)" "$sync_target"
sync_stdout="$(mktemp)"
sync_stderr="$(mktemp)"
set +e
SYNC_HEAD_HOST="$head_host" SYNC_RSH="$sync_rsh" "$cluster_sync_lib_dir/sync_from_head.sh" "$sync_target" > "$sync_stdout" 2> "$sync_stderr"
sync_status=$?
set -u
cat "$sync_stdout"
cat "$sync_stderr" >&2
printf '[%s] sync_from_head finished with status %s\n' "$(date -Is)" "$sync_status"
if ((sync_status != 0)); then
  write_sync_fail_json "$sync_status" "$sync_stdout" "$sync_stderr"
fi
exit "$sync_status"
WATCHER
chmod +x "$watcher_script"

nohup "$watcher_script" \
  "$job_id" \
  "$root" \
  "$CLUSTER_SYNC_LIB_DIR" \
  "$SYNC_HEAD_HOST" \
  "$SYNC_RSH" \
  "$poll_interval" \
  "$state_file" \
  "$job_lock" \
  "$log_file" \
  > "$log_file" 2>&1 &
watcher_pid=$!

created_lock=0
trap - EXIT

echo "Submitted batch job ${job_id}"
echo "sync_from_head watcher pid: ${watcher_pid}"
echo "job lock: ${SYNC_HEAD_HOST}:${root}/${job_lock}"
