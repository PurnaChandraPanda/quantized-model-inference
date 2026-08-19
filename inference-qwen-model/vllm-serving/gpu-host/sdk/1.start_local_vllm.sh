#!/usr/bin/env bash

set -Eeuo pipefail

# -----------------------------------------------------------------------------
# Model configuration
# -----------------------------------------------------------------------------

model_dir="${MODEL_DIR:-/tmp/qwen3.5-0.8b}"
served_model_name="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-0.8B}"

# Maximum prompt tokens + generated tokens for a single sequence.
max_model_len="${MAX_MODEL_LEN:-4096}"

# Maximum number of active sequences that vLLM can batch.
# 8 is a conservative starting point for an A100.
max_num_seqs="${MAX_NUM_SEQS:-8}"

# Fraction of currently available GPU memory vLLM may use.
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.90}"

host="${VLLM_HOST:-0.0.0.0}"
port="${VLLM_PORT:-8000}"

# Select the first GPU unless already explicitly configured.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Use the selected CUDA 12.9 toolkit.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# -----------------------------------------------------------------------------
# Remove CPU-only vLLM configuration
# -----------------------------------------------------------------------------

unset VLLM_TARGET_DEVICE || true
unset VLLM_CPU_KVCACHE_SPACE || true
unset VLLM_CPU_OMP_THREADS_BIND || true

# Do not inherit the CPU TCMalloc preload into the GPU environment.
unset LD_PRELOAD || true

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

error() {
    echo "ERROR: $*" >&2
    exit 1
}

on_error() {
    local exit_code=$?
    echo
    echo "vLLM startup failed with exit code ${exit_code}." >&2
    exit "${exit_code}"
}

trap on_error ERR

# -----------------------------------------------------------------------------
# Validate commands and paths
# -----------------------------------------------------------------------------

command -v nvidia-smi >/dev/null 2>&1 ||
    error "nvidia-smi was not found."

command -v nvcc >/dev/null 2>&1 ||
    error "nvcc was not found. Check CUDA_HOME and PATH."

command -v python >/dev/null 2>&1 ||
    error "python was not found in the active environment."

command -v vllm >/dev/null 2>&1 ||
    error "vllm was not found. Activate the vllm-cu129 environment."

[[ -d "${model_dir}" ]] ||
    error "Model directory does not exist: ${model_dir}"

[[ -f "${model_dir}/config.json" ]] ||
    error "config.json was not found under: ${model_dir}"

[[ "${max_model_len}" =~ ^[0-9]+$ ]] ||
    error "MAX_MODEL_LEN must be a positive integer."

[[ "${max_num_seqs}" =~ ^[0-9]+$ ]] ||
    error "MAX_NUM_SEQS must be a positive integer."

(( max_model_len > 0 )) ||
    error "MAX_MODEL_LEN must be greater than zero."

(( max_num_seqs > 0 )) ||
    error "MAX_NUM_SEQS must be greater than zero."

# -----------------------------------------------------------------------------
# Validate CUDA and vLLM Python packages
# -----------------------------------------------------------------------------

echo "Validating CUDA and vLLM environment..."

python - <<'PY'
import sys

import torch
import vllm

print(f"Python:               {sys.version.split()[0]}")
print(f"vLLM:                 {vllm.__version__}")
print(f"PyTorch:              {torch.__version__}")
print(f"PyTorch CUDA build:   {torch.version.cuda}")
print(f"CUDA available:       {torch.cuda.is_available()}")
#print(f"Visible GPU count:    {torch.cuda.device_count()}")

if vllm.__version__.endswith("+cpu"):
    raise RuntimeError(
        f"CPU-only vLLM is installed: {vllm.__version__}. "
        "Activate the CUDA 12.9 vLLM environment."
    )

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available to PyTorch. Check the NVIDIA driver, "
        "CUDA_VISIBLE_DEVICES, and installed PyTorch/vLLM packages."
    )

if torch.version.cuda != "12.9":
    raise RuntimeError(
        f"Expected PyTorch CUDA build 12.9, found {torch.version.cuda}."
    )

#for index in range(torch.cuda.device_count()):
#    properties = torch.cuda.get_device_properties(index)

#    print(f"GPU {index} name:         {properties.name}")
#    print(
#        f"GPU {index} capability:   "
#        f"{properties.major}.{properties.minor}"
#    )
#    print(
#        f"GPU {index} memory GiB:    "
#        f"{properties.total_memory / (1024 ** 3):.2f}"
#    )

print("CUDA/vLLM validation succeeded.")
PY

echo
echo "nvcc information:"
nvcc --version

echo
echo "NVIDIA GPU information:"
nvidia-smi \
    --query-gpu=index,name,driver_version,memory.total,memory.free \
    --format=csv

# -----------------------------------------------------------------------------
# Build command
# -----------------------------------------------------------------------------

vllm_command=(
    vllm
    serve
    "${model_dir}"
    --served-model-name
    "${served_model_name}"
    --dtype
    bfloat16
    --max-model-len
    "${max_model_len}"
    --max-num-seqs
    "${max_num_seqs}"
    --gpu-memory-utilization
    "${gpu_memory_utilization}"
    --host
    "${host}"
    --port
    "${port}"
    --trust-remote-code
)

# Optional diagnostic mode.
#
# ENFORCE_EAGER=true disables CUDA graphs and most compilation optimizations.
# It is useful for diagnosing startup or compilation failures, but generally
# should not be enabled for normal GPU serving.
if [[ "${ENFORCE_EAGER:-false}" == "true" ]]; then
    vllm_command+=(--enforce-eager)
fi

# Optional vLLM generation defaults instead of generation_config.json from the
# model directory.
if [[ "${USE_VLLM_GENERATION_CONFIG:-false}" == "true" ]]; then
    vllm_command+=(--generation-config vllm)
fi

# -----------------------------------------------------------------------------
# Start server
# -----------------------------------------------------------------------------

echo
echo "Starting vLLM GPU server"
echo "------------------------------------------------------------"
echo "Model directory:          ${model_dir}"
echo "Served model name:        ${served_model_name}"
echo "Maximum model length:     ${max_model_len}"
echo "Maximum active sequences: ${max_num_seqs}"
echo "GPU memory utilization:   ${gpu_memory_utilization}"
echo "CUDA_VISIBLE_DEVICES:     ${CUDA_VISIBLE_DEVICES}"
echo "CUDA_HOME:                ${CUDA_HOME}"
echo "Server address:           ${host}:${port}"
echo "Enforce eager:            ${ENFORCE_EAGER:-false}"
echo "------------------------------------------------------------"

printf 'Command:'
printf ' %q' "${vllm_command[@]}"
printf '\n\n'

# Replace this shell with the vLLM process so SIGTERM/SIGINT are handled
# correctly.
exec "${vllm_command[@]}"

