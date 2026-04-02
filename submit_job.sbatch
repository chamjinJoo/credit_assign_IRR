#!/bin/bash

##### RESOURCE CONFIG (modifiable) #####

#####SBATCH --ntasks 8
#SBATCH -p rtx2080
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -J job
#SBATCH --time=0-24:00:00
#SBATCH --signal=B:TERM@600
#SBATCH -o slurms/logs/slurm-%x-%j.out
#SBATCH -e slurms/logs/slurm-%x-%j.err

set -euo pipefail

##### USER CONFIG (modifiable) #####
ENV_MODE="tar"                        # tar | conda
ENV_TARBALL="$SLURM_SUBMIT_DIR/slurms/bnn_conda_env.tar.gz"
CONDA_ENV_NAME="bnn"

#### WANDB setting ####
WANDB="on"   # on | off
WANDB_KEY_FILE="$HOME/.secrets/wandb/apikey"

#### Caching setting ####
KEEP_JOBTMP=0
KEEP_WORKCACHE=1

#### Run job based on your input ####
ENTRYPOINT=$1

# ENTRYPOINT="python -m multitask_rnn.training.sweep \
# --epochs 5000 \
# --eval-every 10 \
# --no-tqdm"
# # CONFIG_FILE="configs/sweep/multisen_rnn_sweep.yaml"
# CONFIG_FILE="configs/sweep/multisen_force_sweep.yaml"


########### DO NOT MODIFY BELOW ###########
#  [A1]
JOBID="$SLURM_JOB_ID"
SUBMIT_DIR="$SLURM_SUBMIT_DIR"
HOST="$(hostname)"

IS_GD1=0
if [ "$HOST" = "bmlgd1" ]; then
  IS_GD1=1
fi

if [ "$IS_GD1" = "1" ]; then
  # GD1: run in submit dir (no /scratch stage-in)
  PROJECT_KEY=""
  WORKROOT=""
  WORKDIR="$SUBMIT_DIR"

  # JOBTMP is unused in this mode, but keep it defined for cleanup logic consistency
  JOBTMP="/scratch/$USER/$JOBID"

else
  # Compute nodes: run on local /scratch with directory-based cache
  PROJECT_KEY="$(echo -n "$SUBMIT_DIR" | sha256sum | awk '{print $1}')"
  WORKROOT="/scratch/$USER/workcache"
  WORKDIR="$WORKROOT/$PROJECT_KEY"
  JOBTMP="/scratch/$USER/$JOBID"

  mkdir -p "$WORKROOT" "$JOBTMP"
fi

echo "[meta] HOST=$HOST"
echo "[meta] JOBID=$JOBID"
echo "[meta] SUBMIT_DIR=$SUBMIT_DIR"
echo "[meta] WORKDIR=$WORKDIR"
echo "[meta] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

#  [A2]
echo "[env] setup start"
if [ "$ENV_MODE" = "tar" ]; then
  ENV_CACHE_ROOT="/scratch/$USER/envcache"
  mkdir -p "$ENV_CACHE_ROOT"

  ENV_HASH="$(sha256sum "$ENV_TARBALL" | awk '{print $1}')"
  ENV_BASE_DIR="$ENV_CACHE_ROOT/$ENV_HASH"
  ENV_RUN_DIR="$JOBTMP/env"

  if [ ! -d "$ENV_BASE_DIR" ]; then
    echo "[env] unpack tarball -> $ENV_BASE_DIR"
    mkdir -p "$ENV_BASE_DIR"
    tar -xzf "$ENV_TARBALL" -C "$ENV_BASE_DIR"
  fi

  rm -rf "$ENV_RUN_DIR"
  mkdir -p "$ENV_RUN_DIR"
  rsync -a "$ENV_BASE_DIR/" "$ENV_RUN_DIR/"

  export PATH="$ENV_RUN_DIR/bin:$PATH"

  # execute conda-unpack in each job-only directory
  if [ -x "$ENV_RUN_DIR/bin/conda-unpack" ]; then
    "$ENV_RUN_DIR/bin/conda-unpack"
  fi

  # ENV_DIR="$ENV_CACHE_ROOT/$ENV_HASH"

  # if [ ! -d "$ENV_DIR" ]; then
  #   echo "[env] unpack tarball -> $ENV_DIR"
  #   mkdir -p "$ENV_DIR"
  #   tar -xzf "$ENV_TARBALL" -C "$ENV_DIR"
  # fi
  # export PATH="$ENV_DIR/bin:$PATH"
  # if [ -x "$ENV_DIR/bin/conda-unpack" ]; then
  #   "$ENV_DIR/bin/conda-unpack"
  # fi
else
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV_NAME"
fi
echo "[env] python=$(command -v python)"
python --version
echo "[env] setup done"


#  [A3]
echo "[wandb] setup"
if [ "$WANDB" = "off" ]; then
  export WANDB_MODE=disabled
  export WANDB_DISABLED=true
  echo "[wandb] disabled"
else
  if [ -f "$WANDB_KEY_FILE" ]; then
    export WANDB_API_KEY="$(tr -d ' \t\r\n' < "$WANDB_KEY_FILE")"
    export WANDB_SILENT=true

    echo "[wandb] key loaded"

    python - <<'PY'
import os, wandb
wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
PY
  else
    echo "[wandb] enabled, but key file not found: $WANDB_KEY_FILE"
    echo "[wandb] job may fail if your code calls wandb.init() without a key"
  fi
fi


#  [A4]
echo "[stage-in] rsync submit -> workcache start"
if [ "${IS_GD1:-0}" = "1" ]; then
  echo "[stage-in] GD1 mode: skip (running in SUBMIT_DIR)"
else
  mkdir -p "$WORKDIR"
  rsync -a \
    --info=progress2 \
    --human-readable \
    --exclude=.git \
    --exclude=__pycache__ \
    "$SUBMIT_DIR/" "$WORKDIR/"
fi
echo "[stage-in] done"

# [A4.5] PYTHONPATH setup (after stage-in)
if [[ -d "$WORKDIR/src" ]]; then
  if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="$WORKDIR/src:$PYTHONPATH"
  else
    export PYTHONPATH="$WORKDIR/src"
  fi
else
  echo "[warn] $WORKDIR/src not found. PYTHONPATH unchanged: ${PYTHONPATH:-}"
fi
echo "[meta] PYTHONPATH=$PYTHONPATH"

# CONFIG_FILE="$WORKDIR/$CONFIG_FILE"
# if [[ ! -f "$CONFIG_FILE" ]]; then
#   echo "ERROR: config not found: $CONFIG_FILE" >&2
#   echo "DEBUG: pwd=$(pwd)" >&2
#   echo "DEBUG: list configs:" >&2
#   ls -la "$WORKDIR/configs" >&2 || true
#   ls -la "$WORKDIR/configs/sweep" >&2 || true
#   exit 1
# fi

# ENTRYPOINT="$ENTRYPOINT --config \"$CONFIG_FILE\""

# Define staging before running
stage_out_and_cleanup() {
  echo "[trap] stage-out + cleanup"

  # [A6] stage-out
  echo "[stage-out] rsync workdir -> submit start"
  if [ "${IS_GD1:-0}" = "1" ]; then
    echo "[stage-out] GD1 mode: skip (running in SUBMIT_DIR)"
  else
    rsync -a \
      --info=progress2 \
      --human-readable \
      --update \
      --exclude=".git" \
      --exclude="__pycache__" \
      "$WORKDIR/" "$SUBMIT_DIR/"
  fi
  echo "[stage-out] done"

  # [A7] cleanup
  echo "[cleanup] start"
  if [ "$KEEP_JOBTMP" = "0" ]; then
    rm -rf "$JOBTMP"
    echo "[cleanup] removed $JOBTMP"
  else
    echo "[cleanup] kept $JOBTMP"
  fi

  if [ "$KEEP_WORKCACHE" = "0" ]; then
    if [ "${IS_GD1:-0}" = "1" ]; then
      echo "[cleanup] GD1 mode: skip removing WORKDIR (WORKDIR is SUBMIT_DIR)"
    else
      rm -rf "$WORKDIR"
      echo "[cleanup] removed $WORKDIR"
    fi
  else
    echo "[cleanup] kept $WORKDIR"
  fi

  echo "[cleanup] done"
  echo "[done] $SUBMIT_DIR"
}

trap stage_out_and_cleanup EXIT TERM INT


#  [A5]
cd "$WORKDIR"

echo "[run] $ENTRYPOINT"
stdbuf -oL -eL bash -c "$ENTRYPOINT"
echo "[run] done"
