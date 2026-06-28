#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

SOURCE_ROOT="$REPO_ROOT/checkpoints_for_eval"
REMOTE_USER="chamjin"
REMOTE_HOST="143.248.74.43"
REMOTE_ROOT="/home/chamjin/research/pci_marl/checkpoints_for_eval"
DRY_RUN=false

print_usage() {
    cat <<'EOF'
Usage:
  bash scripts/send_checkpoints_for_eval.sh [--source-root PATH] [--remote-user USER] [--remote-host HOST] [--remote-root PATH] [--dry-run]

Description:
  Sync only the checkpoints/ directories under checkpoints_for_eval to the
  remote server. Existing remote checkpoint files are preserved because this
  script does not use --delete.

Options:
  --source-root PATH   Local checkpoints_for_eval root. Default: ./checkpoints_for_eval
  --remote-user USER   SSH user. Default: chamjin
  --remote-host HOST   SSH host. Default: 143.248.74.43
  --remote-root PATH   Remote destination root. Default: /home/chamjin/research/pci_marl/checkpoints_for_eval
  --dry-run            Show what would be transferred without copying files.
  -h, --help           Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-root)
            [[ $# -ge 2 ]] || { echo "Missing value for --source-root" >&2; exit 1; }
            SOURCE_ROOT="$2"
            shift 2
            ;;
        --remote-user)
            [[ $# -ge 2 ]] || { echo "Missing value for --remote-user" >&2; exit 1; }
            REMOTE_USER="$2"
            shift 2
            ;;
        --remote-host)
            [[ $# -ge 2 ]] || { echo "Missing value for --remote-host" >&2; exit 1; }
            REMOTE_HOST="$2"
            shift 2
            ;;
        --remote-root)
            [[ $# -ge 2 ]] || { echo "Missing value for --remote-root" >&2; exit 1; }
            REMOTE_ROOT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            print_usage >&2
            exit 1
            ;;
    esac
done

if [[ ! -d "$SOURCE_ROOT" ]]; then
    echo "Source root does not exist: $SOURCE_ROOT" >&2
    exit 1
fi

REMOTE_SPEC="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/"
REMOTE_RSYNC_PATH=$(printf "mkdir -p %q && rsync" "$REMOTE_ROOT")

RSYNC_ARGS=(
    -a
    --info=progress2
    --human-readable
    --prune-empty-dirs
    --include='*/'
    --include='*/checkpoints/'
    --include='*/checkpoints/***'
    --exclude='*'
    -e ssh
    --rsync-path="$REMOTE_RSYNC_PATH"
)

if [[ "$DRY_RUN" == true ]]; then
    RSYNC_ARGS+=(--dry-run --itemize-changes)
fi

rsync "${RSYNC_ARGS[@]}" "$SOURCE_ROOT/" "$REMOTE_SPEC"
