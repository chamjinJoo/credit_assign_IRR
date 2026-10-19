#!/usr/bin/env bash
set -Eeuo pipefail

###############################################################################
# install.sh
#
# Conservative environment bootstrap for the "pcimarl" project on Linux.
#
# What this script does:
#   1) Ensure conda exists (install Miniconda if missing)
#   2) Create/refresh conda env "pcimarl" with Python 3.10.18
#   3) Install PyTorch 2.7.1 + CUDA 12.6
#   4) Install JAX 0.4.38 + CUDA12 plugin/pjrt (jaxmarl 0.1.0 compatible)
#   5) Sanitize requirements file exported from a prior environment
#   6) Install remaining requirements conservatively
#   7) Repair protobuf / wandb compatibility
#   8) Install CUDA nvcc helper package and expose ptxas/nvlink if available
#
# Notes:
# - This script intentionally does NOT do broad "pip -U ..." upgrades.
# - It avoids known bad upgrades that broke:
#     * jaxmarl 0.1.0 (requires jax <= 0.4.38)
#     * torch 2.7.1+cu126 exact CUDA dependency stack
#     * SMAC / SMACv2 protobuf compatibility
# - VMAS is excluded because it forced an incompatible pyglet pin and is unused.
#
# Usage:
#   bash install.sh
#
# Optional environment variables:
#   ENV_NAME=pcimarl
#   PYTHON_VERSION=3.10.18
#   REQUIREMENTS_FILE=requirements.txt
#   CONDA_ROOT=$HOME/miniconda3
###############################################################################

ENV_NAME="${ENV_NAME:-pcimarl}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10.18}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements.txt}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"

log()  { printf '\n[%s] %s\n' "INFO" "$*"; }
warn() { printf '\n[%s] %s\n' "WARN" "$*" >&2; }
die()  { printf '\n[%s] %s\n' "ERROR" "$*" >&2; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

detect_shell_rc() {
  if [[ -n "${BASH_VERSION:-}" ]]; then
    echo "$HOME/.bashrc"
  else
    echo "$HOME/.profile"
  fi
}

ensure_conda() {
  if command -v conda >/dev/null 2>&1; then
    log "conda already found: $(command -v conda)"
    return 0
  fi

  log "conda not found. Installing Miniconda under $CONDA_ROOT"
  need_cmd wget
  need_cmd bash

  if [[ ! -f "$MINICONDA_INSTALLER" ]]; then
    wget -O "$MINICONDA_INSTALLER" "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  fi

  bash "$MINICONDA_INSTALLER" -b -p "$CONDA_ROOT"

  local rcfile
  rcfile="$(detect_shell_rc)"

  # shellcheck disable=SC1091
  "$CONDA_ROOT/bin/conda" init bash || true
  # shellcheck disable=SC1091
  source "$CONDA_ROOT/etc/profile.d/conda.sh"

  "$CONDA_ROOT/bin/conda" config --set auto_activate_base false || true

  if ! grep -q 'miniconda3/etc/profile.d/conda.sh' "$rcfile" 2>/dev/null; then
    cat >> "$rcfile" <<'EOF'

# Added by install.sh for Miniconda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
EOF
  fi

  log "Miniconda installed. Future shells should load conda automatically."
}

activate_conda() {
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
  elif [[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
  else
    die "Unable to source conda.sh"
  fi
}

run_conda_relaxed_nounset() {
  local had_nounset=0
  case $- in
    *u*)
      had_nounset=1
      set +u
      ;;
  esac

  "$@"
  local status=$?

  if [[ $had_nounset -eq 1 ]]; then
    set -u
  fi

  return $status
}

create_env() {
  activate_conda

  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    warn "Conda env '$ENV_NAME' already exists. It will be reused."
  else
    log "Creating conda env '$ENV_NAME' with Python $PYTHON_VERSION"
    run_conda_relaxed_nounset conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
  fi

  run_conda_relaxed_nounset conda activate "$ENV_NAME"
  python -m pip install --upgrade pip setuptools wheel
}

assert_requirements_file() {
  [[ -f "$REQUIREMENTS_FILE" ]] || die "Requirements file not found: $REQUIREMENTS_FILE"
}

install_torch() {
  log "Installing PyTorch 2.7.1 + cu126"
  pip install --force-reinstall \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu126
}

install_jax_stack() {
  log "Installing JAX 0.4.38 stack compatible with jaxmarl 0.1.0"
  pip install --force-reinstall \
    jax==0.4.38 \
    jaxlib==0.4.38 \
    jax-cuda12-plugin==0.4.38 \
    jax-cuda12-pjrt==0.4.38 \
    jaxmarl==0.1.0 \
    jaxopt==0.8.5
}

build_sanitized_requirements() {
  assert_requirements_file

  log "Building sanitized requirements-rest.txt from $REQUIREMENTS_FILE"

  # 1) Drop local conda-build file:// references; they are not portable.
  # 2) Drop packages we manage explicitly or know to be conflict-prone.
  # 3) Drop VMAS because it forced incompatible pyglet constraints and is unused.
  awk '
    BEGIN { IGNORECASE = 0 }
    /@ file:\/\// { next }
    /^torch==/ { next }
    /^torchvision==/ { next }
    /^torchaudio==/ { next }
    /^jax==/ { next }
    /^jaxlib==/ { next }
    /^jax-cuda12-pjrt==/ { next }
    /^jax-cuda12-plugin==/ { next }
    /^jaxmarl==/ { next }
    /^jaxopt==/ { next }
    /^nvidia-/ { next }
    /^triton==/ { next }
    /^vmas==/ { next }
    { print }
  ' "$REQUIREMENTS_FILE" > requirements-rest.txt

  log "Sanitized requirements written to requirements-rest.txt"
}

install_remaining_requirements() {
  log "Installing remaining Python requirements"
  pip install -r requirements-rest.txt || {
    warn "requirements-rest.txt installation failed."
    warn "This often means a project-specific extra package still conflicts."
    warn "Inspect the failing package and handle it separately."
    exit 1
  }
}

repair_numpy_if_needed() {
  log "Ensuring NumPy stays compatible with SciPy 1.12 and JAX stack"
  pip install --force-reinstall "numpy==1.26.4"
}

repair_protobuf_and_wandb() {
  log "Repairing protobuf / wandb compatibility"

  # SMAC / SMACv2 require protobuf<3.21
  pip install --force-reinstall "protobuf==3.20.3"

  # wandb >= 0.22.3 supports long API keys; install without pulling protobuf back up.
  pip uninstall -y wandb >/dev/null 2>&1 || true
  pip install --force-reinstall --no-deps "wandb==0.22.3"
}

install_cuda_compiler_bits() {
  log "Installing CUDA nvcc helper package (for ptxas; nvlink may still require system/conda toolkit)"
  pip install --force-reinstall "nvidia-cuda-nvcc-cu12==12.9.86"
}

configure_path_for_ptxas() {
  local nvcc_bin
  nvcc_bin="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin"

  if [[ -d "$nvcc_bin" ]]; then
    export PATH="$nvcc_bin:$PATH"
    log "Added to PATH for current shell: $nvcc_bin"
  fi

  local rcfile
  rcfile="$(detect_shell_rc)"
  if [[ -d "$nvcc_bin" ]] && ! grep -Fq "$nvcc_bin" "$rcfile" 2>/dev/null; then
    {
      echo ""
      echo "# Added by install.sh for JAX CUDA toolchain visibility"
      echo "export PATH=\"$nvcc_bin:\$PATH\""
    } >> "$rcfile"
    log "Persisted PATH update to $rcfile"
  fi
}

diagnose_cuda_toolchain() {
  log "Diagnosing CUDA toolchain visibility"
  echo "CONDA_PREFIX=$CONDA_PREFIX"
  echo "PATH=$PATH"
  echo

  echo "which ptxas:"
  which ptxas || true
  echo

  echo "which nvlink:"
  which nvlink || true
  echo

  echo "candidate ptxas/nvlink files under env:"
  find "$CONDA_PREFIX" -type f \( -name ptxas -o -name nvlink -o -name libdevice10.bc \) 2>/dev/null || true
  echo
}

maybe_install_conda_cuda_nvcc() {
  if which nvlink >/dev/null 2>&1; then
    log "nvlink already visible in PATH"
    return 0
  fi

  warn "nvlink is still not visible."
  warn "Trying conda-forge/NVIDIA channel compiler package for better JAX compatibility."
  activate_conda
  run_conda_relaxed_nounset conda activate "$ENV_NAME"

  # Prefer compiler-only package first; if unavailable or insufficient, user can install full toolkit.
  if run_conda_relaxed_nounset conda install -c nvidia cuda-nvcc=12.9 -y; then
    true
  else
    warn "conda install -c nvidia cuda-nvcc=12.9 failed. You may need:"
    warn "  conda install -c nvidia cuda-toolkit=12.9 -y"
  fi
}

verify_versions() {
  log "Verifying key package versions"
  python - <<'PY'
import importlib
import sys

def safe_import(name):
    try:
        mod = importlib.import_module(name)
        return mod
    except Exception as e:
        print(f"{name}: IMPORT FAILED -> {e}")
        return None

print("python:", sys.version.split()[0])

np = safe_import("numpy")
if np: print("numpy:", np.__version__)

scipy = safe_import("scipy")
if scipy: print("scipy:", scipy.__version__)

torch = safe_import("torch")
if torch:
    print("torch:", torch.__version__)
    print("torch.cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("torch.cuda.device_count:", torch.cuda.device_count())
        try:
            print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))
        except Exception as e:
            print("torch device name query failed:", e)

jax = safe_import("jax")
if jax: print("jax:", jax.__version__)

jaxlib = safe_import("jaxlib")
if jaxlib: print("jaxlib:", jaxlib.__version__)

wandb = safe_import("wandb")
if wandb: print("wandb:", wandb.__version__)

pb = safe_import("google.protobuf")
if pb: print("protobuf:", pb.__version__)
PY
}

final_notes() {
  cat <<'EOF'

===============================================================================
INSTALL COMPLETE

Recommended next checks:
  conda activate <env-name>
  which ptxas
  which nvlink
  python -c "import wandb; print(wandb.__version__)"
  python -c "import jax; print(jax.devices())"

If 'which nvlink' is still empty:
  1) Check system CUDA:
       find /usr/local -type f -name nvlink 2>/dev/null
       find /usr -type f -name nvlink 2>/dev/null
  2) If found, add that CUDA bin directory to PATH.
  3) Otherwise try:
       conda install -c nvidia cuda-toolkit=12.9 -y

For W&B login:
  wandb login --relogin

Important:
  - Do NOT run: pip install -U "jax[cuda12]"
  - Do NOT broadly upgrade protobuf, numpy, jax, or torch afterward.
  - If JAX still only runs with:
       export XLA_FLAGS="--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found=true"
    then the remaining issue is almost certainly nvlink/toolchain visibility.

===============================================================================
EOF
}

main() {
  ensure_conda
  create_env

  log "Activated env: $ENV_NAME"
  log "Using requirements file: $REQUIREMENTS_FILE"

  install_torch
  install_jax_stack
  build_sanitized_requirements
  install_remaining_requirements
  repair_numpy_if_needed
  repair_protobuf_and_wandb
  install_cuda_compiler_bits
  configure_path_for_ptxas
  maybe_install_conda_cuda_nvcc
  diagnose_cuda_toolchain
  verify_versions
  final_notes
}

main "$@"
