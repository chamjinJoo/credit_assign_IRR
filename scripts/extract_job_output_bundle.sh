#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
job_outputs_dir="$repo_root/slurms/job_outputs"
keep_archive=0
extract_all=0
bundle=""

usage() {
  echo "usage: scripts/extract_job_output_bundle.sh [--all] [--keep-archive] [slurms/job_outputs/<jobid>/checkpoints_<jobid>.tar.gz]" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --all)
      extract_all=1
      ;;
    --keep-archive)
      keep_archive=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "error: unknown option: $1" >&2
      usage
      exit 2
      ;;
    *)
      if [ -n "$bundle" ]; then
        echo "error: only one bundle path can be provided" >&2
        usage
        exit 2
      fi
      bundle="$1"
      ;;
  esac
  shift
done

if [ "$#" -gt 0 ]; then
  if [ -n "$bundle" ]; then
    echo "error: only one bundle path can be provided" >&2
    usage
    exit 2
  fi
  bundle="$1"
  shift
fi

if [ "$#" -gt 0 ]; then
  echo "error: unexpected argument: $1" >&2
  usage
  exit 2
fi

if [ "$extract_all" = "1" ] && [ -n "$bundle" ]; then
  echo "error: --all cannot be combined with a bundle path" >&2
  usage
  exit 2
fi

cleanup_leftover_job_dirs() {
  local dir
  local first_archive

  if [ "$keep_archive" = "1" ] || [ ! -d "$job_outputs_dir" ]; then
    return
  fi

  while IFS= read -r -d '' dir; do
    first_archive=""
    shopt -s nullglob
    for first_archive in "$dir"/checkpoints_*.tar.gz; do
      break
    done
    shopt -u nullglob

    if [ -z "$first_archive" ] && [ -f "$dir/manifest.txt" ]; then
      rm -f "$dir/manifest.txt"
    fi

    rmdir "$dir" 2>/dev/null || true
  done < <(find "$job_outputs_dir" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)

  rmdir "$job_outputs_dir" 2>/dev/null || true
}

resolve_bundle_path() {
  local candidate="$1"

  if [ -f "$candidate" ]; then
    printf '%s\n' "$candidate"
  elif [ -f "$repo_root/$candidate" ]; then
    printf '%s\n' "$repo_root/$candidate"
  else
    return 1
  fi
}

collect_all_bundles() {
  local entry

  if [ ! -d "$job_outputs_dir" ]; then
    return
  fi

  while IFS= read -r -d '' entry; do
    bundles+=("${entry#* }")
  done < <(find "$job_outputs_dir" -type f -name 'checkpoints_*.tar.gz' -printf '%T@ %p\0' 2>/dev/null | sort -z -n)
}

collect_latest_bundle() {
  local entry

  if [ ! -d "$job_outputs_dir" ]; then
    return
  fi

  while IFS= read -r -d '' entry; do
    bundles+=("${entry#* }")
    break
  done < <(find "$job_outputs_dir" -type f -name 'checkpoints_*.tar.gz' -printf '%T@ %p\0' 2>/dev/null | sort -z -nr)
}

extract_bundle() {
  local archive="$1"
  local bundle_dir

  echo "[extract] $archive -> $repo_root"
  tar -xzf "$archive" -C "$repo_root"

  if [ "$keep_archive" = "0" ]; then
    bundle_dir="$(dirname "$archive")"
    rm -f "$archive"
    rm -f "$bundle_dir/manifest.txt"
    rmdir "$bundle_dir" 2>/dev/null || true
    echo "[extract] removed archive and manifest after successful extraction"
  fi
}

bundles=()

if [ "$extract_all" = "1" ]; then
  collect_all_bundles
elif [ -n "$bundle" ]; then
  if ! bundle="$(resolve_bundle_path "$bundle")"; then
    echo "error: checkpoint bundle not found: $bundle" >&2
    usage
    exit 1
  fi
  bundles+=("$bundle")
else
  collect_latest_bundle
fi

if [ "${#bundles[@]}" -eq 0 ]; then
  cleanup_leftover_job_dirs
  echo "error: checkpoint bundle not found" >&2
  usage
  exit 1
fi

for bundle in "${bundles[@]}"; do
  extract_bundle "$bundle"
done

cleanup_leftover_job_dirs
echo "[extract] done (${#bundles[@]} bundle(s))"
