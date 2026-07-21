#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
job_outputs_dir="$repo_root/slurms/job_outputs"
keep_archive=0
extract_all=0
completed_only=0
bundle=""

usage() {
  echo "usage: scripts/extract_job_output_bundle.sh [--all|--completed-only] [--keep-archive] [slurms/job_outputs/<jobid>/checkpoints_<jobid>.tar.gz]" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --all)
      extract_all=1
      ;;
    --completed-only|--completed)
      completed_only=1
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

if [ "$completed_only" = "1" ] && [ -n "$bundle" ]; then
  echo "error: --completed-only cannot be combined with a bundle path" >&2
  usage
  exit 2
fi

if [ "$completed_only" = "1" ] && [ "$extract_all" = "1" ]; then
  echo "error: --completed-only cannot be combined with --all" >&2
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

manifest_value() {
  local manifest="$1"
  local key="$2"

  if [ ! -f "$manifest" ]; then
    return
  fi

  sed -n "s/^${key}=//p" "$manifest" | tail -n 1
}

remove_job_output_dir() {
  local dir="$1"

  rm -f "$dir"/checkpoints_*.tar.gz
  rm -f "$dir/manifest.txt"
  rmdir "$dir" 2>/dev/null || true
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

collect_completed_bundles_and_prune_incomplete() {
  local dir
  local manifest
  local status
  local checkpoint_bundle
  local archive_name
  local archive
  local first_archive

  if [ ! -d "$job_outputs_dir" ]; then
    return
  fi

  while IFS= read -r -d '' dir; do
    manifest="$dir/manifest.txt"

    if [ ! -f "$manifest" ]; then
      echo "[skip] missing manifest: $dir" >&2
      continue
    fi

    status="$(manifest_value "$manifest" status)"
    checkpoint_bundle="$(manifest_value "$manifest" checkpoint_bundle)"

    case "$status" in
      completed)
        if [ "$checkpoint_bundle" = "0" ]; then
          echo "[skip] completed without checkpoint bundle: $dir" >&2
          if [ "$keep_archive" = "0" ]; then
            remove_job_output_dir "$dir"
          fi
          continue
        fi

        archive_name="$(manifest_value "$manifest" archive)"
        archive=""
        if [ -n "$archive_name" ] && [ -f "$dir/$archive_name" ]; then
          archive="$dir/$archive_name"
        else
          first_archive=""
          shopt -s nullglob
          for first_archive in "$dir"/checkpoints_*.tar.gz; do
            break
          done
          shopt -u nullglob
          archive="$first_archive"
        fi

        if [ -n "$archive" ] && [ -f "$archive" ]; then
          bundles+=("$archive")
        else
          echo "[skip] completed manifest but archive is missing: $dir" >&2
          if [ "$keep_archive" = "0" ]; then
            remove_job_output_dir "$dir"
          fi
        fi
        ;;
      failed|terminated)
        echo "[prune] removing $status job output: $dir" >&2
        remove_job_output_dir "$dir"
        ;;
      "")
        echo "[skip] manifest without status: $dir" >&2
        ;;
      *)
        echo "[skip] non-completed status=$status: $dir" >&2
        ;;
    esac
  done < <(find "$job_outputs_dir" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null | sort -z)
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

if [ "$completed_only" = "1" ]; then
  collect_completed_bundles_and_prune_incomplete
elif [ "$extract_all" = "1" ]; then
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
  if [ "$completed_only" = "1" ]; then
    echo "error: completed checkpoint bundle not found" >&2
  else
    echo "error: checkpoint bundle not found" >&2
  fi
  usage
  exit 1
fi

for bundle in "${bundles[@]}"; do
  extract_bundle "$bundle"
done

cleanup_leftover_job_dirs
echo "[extract] done (${#bundles[@]} bundle(s))"
