#!/usr/bin/env bash
set -Eeuo pipefail

###############################################################################
# install.sh
#
# Conservative dependency installer for the "pcimarl" project on Linux.
#
# What this script does:
#   1) Install PyTorch 2.7.1 + CUDA 12.6
#   2) Install JAX 0.4.38 + CUDA12 plugin/pjrt (jaxmarl 0.1.0 compatible)
#   3) Sanitize requirements file exported from a prior environment
#   4) Install remaining requirements conservatively
#   5) Repair protobuf / wandb compatibility
#   6) Install CUDA nvcc helper package and expose ptxas/nvlink if available
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
#   REQUIREMENTS_FILE=requirements.txt
###############################################################################

REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements.txt}"

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

prepare_python() {
  need_cmd python
  python -m pip --version >/dev/null 2>&1 || die "pip is unavailable for $(command -v python)"
  python -m pip install --upgrade pip setuptools wheel
}

assert_requirements_file() {
  [[ -f "$REQUIREMENTS_FILE" ]] || die "Requirements file not found: $REQUIREMENTS_FILE"
}

install_torch() {
  log "Installing PyTorch 2.7.1 + cu126"
  python -m pip install --force-reinstall \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu126
}

install_jax_stack() {
  log "Installing JAX 0.4.38 stack compatible with jaxmarl 0.1.0"
  python -m pip install --force-reinstall \
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
  python -m pip install -r requirements-rest.txt || {
    warn "requirements-rest.txt installation failed."
    warn "This often means a project-specific extra package still conflicts."
    warn "Inspect the failing package and handle it separately."
    exit 1
  }
}

repair_numpy_if_needed() {
  log "Ensuring NumPy stays compatible with SciPy 1.12 and JAX stack"
  python -m pip install --force-reinstall "numpy==1.26.4"
}

repair_protobuf_and_wandb() {
  log "Repairing protobuf / wandb compatibility"

  # SMAC / SMACv2 require protobuf<3.21
  python -m pip install --force-reinstall "protobuf==3.20.3"

  # wandb >= 0.22.3 supports long API keys; install without pulling protobuf back up.
  python -m pip uninstall -y wandb >/dev/null 2>&1 || true
  python -m pip install --force-reinstall --no-deps "wandb==0.22.3"
}

install_cuda_compiler_bits() {
  log "Installing CUDA nvcc helper package (for ptxas; nvlink may still require a system toolkit)"
  python -m pip install --force-reinstall "nvidia-cuda-nvcc-cu12==12.9.86"
}

configure_path_for_ptxas() {
  local nvcc_bin
  local site_packages
  site_packages="$(python -c 'import site; print(site.getsitepackages()[0])')"
  nvcc_bin="$site_packages/nvidia/cuda_nvcc/bin"

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
  local site_packages
  site_packages="$(python -c 'import site; print(site.getsitepackages()[0])')"
  echo "site-packages=$site_packages"
  echo "PATH=$PATH"
  echo

  echo "which ptxas:"
  which ptxas || true
  echo

  echo "which nvlink:"
  which nvlink || true
  echo

  echo "candidate ptxas/nvlink files under site-packages:"
  find "$site_packages" -type f \( -name ptxas -o -name nvlink -o -name libdevice10.bc \) 2>/dev/null || true
  echo
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
  which ptxas
  which nvlink
  python -c "import wandb; print(wandb.__version__)"
  python -c "import jax; print(jax.devices())"

If 'which nvlink' is still empty:
  1) Check system CUDA:
       find /usr/local -type f -name nvlink 2>/dev/null
       find /usr -type f -name nvlink 2>/dev/null
  2) If found, add that CUDA bin directory to PATH.
  3) Otherwise install a system CUDA toolkit that provides nvlink.

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
  prepare_python

  log "Using Python: $(command -v python)"
  log "Using requirements file: $REQUIREMENTS_FILE"

  install_torch
  install_jax_stack
  build_sanitized_requirements
  install_remaining_requirements
  repair_numpy_if_needed
  repair_protobuf_and_wandb
  install_cuda_compiler_bits
  configure_path_for_ptxas
  diagnose_cuda_toolchain
  verify_versions
  final_notes
}

main "$@"
