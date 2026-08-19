# Azure ML curated GPU inference image.
# This image already contains CUDA 12.4 components, Python 3.12,
# Conda, and Azure ML inference image integration.
FROM mcr.microsoft.com/azureml/curated/minimal-py312-cuda12.4-inference:latest

USER root

ARG DEBIAN_FRONTEND=noninteractive
ARG VLLM_VERSION="0.27.1"

# ----------------------------------------------------------------------
# Basic system dependencies
# ----------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        wget \
        gnupg && \
    rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------
# Add the NVIDIA CUDA 12.9 local repository
#
# This follows the same approach that worked on the local A100 machine.
# Do not install cuda-drivers inside the container. Azure ML injects the
# NVIDIA driver from the GPU host.
# ----------------------------------------------------------------------
WORKDIR /tmp

ARG CUDA_REPO_DEB="cuda-repo-ubuntu2204-12-9-local_12.9.0-575.51.03-1_amd64.deb"
ARG CUDA_REPO_URL="https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda-repo-ubuntu2204-12-9-local_12.9.0-575.51.03-1_amd64.deb"

RUN test -n "${CUDA_REPO_URL}" && \
    wget --no-verbose \
        --output-document="/tmp/${CUDA_REPO_DEB}" \
        "${CUDA_REPO_URL}" && \
    dpkg -i "/tmp/${CUDA_REPO_DEB}" && \
    cp /var/cuda-repo-ubuntu2204-12-9-local/cuda-*-keyring.gpg \
        /usr/share/keyrings/ && \
    rm -f "/tmp/${CUDA_REPO_DEB}"

# ----------------------------------------------------------------------
# Install only the CUDA 12.9 components needed by vLLM/PyTorch/Triton.
# ----------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cuda-compiler-12-9 \
        cuda-libraries-12-9 \
        cuda-libraries-dev-12-9 && \
    rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------
# Select CUDA 12.9
# ----------------------------------------------------------------------
ENV CUDA_HOME=/usr/local/cuda-12.9
ENV PATH=${CUDA_HOME}/bin:${PATH}

# Do not add the CUDA stubs directory:
#   /usr/local/cuda-12.9/lib64/stubs
#
# libcuda.so.1 must come from the NVIDIA driver injected by the GPU host.

# ----------------------------------------------------------------------
# Azure ML Conda environment
# ----------------------------------------------------------------------
ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/py312
ENV AZUREML_CONDA_DEFAULT_ENVIRONMENT=${AZUREML_CONDA_ENVIRONMENT_PATH}
ENV PATH=${AZUREML_CONDA_ENVIRONMENT_PATH}/bin:${PATH}

WORKDIR /var/azureml-app

COPY requirements.txt /tmp/requirements.txt
COPY conda.yml /tmp/conda.yml

RUN conda env create \
        -p "${AZUREML_CONDA_ENVIRONMENT_PATH}" \
        -f /tmp/conda.yml \
        -q && \
    rm -f /tmp/conda.yml /tmp/requirements.txt && \
    conda clean -a -y

# ----------------------------------------------------------------------
# Install exactly the same CUDA 12.9 vLLM variant used locally
# ----------------------------------------------------------------------
RUN pip install \
        --no-cache-dir \
        "vllm==${VLLM_VERSION}+cu129" \
        --extra-index-url "https://wheels.vllm.ai/${VLLM_VERSION}/cu129" \
        --extra-index-url "https://download.pytorch.org/whl/cu129"

# ----------------------------------------------------------------------
# Reconcile Azure ML inference-server/OpenTelemetry packages after vLLM
# ----------------------------------------------------------------------

# Remove incompatible opentelemetry related pacakges
RUN pip uninstall -y \
  azure-monitor-opentelemetry \
  azure-monitor-opentelemetry-exporter \
  azure-core-tracing-opentelemetry \
  opentelemetry-resource-detector-azure

# Install again the compatible opentelemetry related pacakges
RUN pip install \
  --no-cache-dir \
  --force-reinstall \
  "azure-core-tracing-opentelemetry==1.0.0b13" \
  "azure-monitor-opentelemetry==1.8.9" \
  "azure-monitor-opentelemetry-exporter==1.0.0b56" \
  "opentelemetry-resource-detector-azure==0.1.5"

# ----------------------------------------------------------------------
# Verify CUDA 12.9 compiler
# ----------------------------------------------------------------------
RUN test -x /usr/local/cuda-12.9/bin/nvcc && \
    /usr/local/cuda-12.9/bin/nvcc --version

# ----------------------------------------------------------------------
# GPU vLLM runtime configuration
# ----------------------------------------------------------------------
ENV TOKENIZERS_PARALLELISM=false

ENV VLLM_SERVED_MODEL_NAME="Qwen/Qwen3.5-0.8B"
ENV VLLM_MAX_MODEL_LEN="4096"
ENV VLLM_MAX_NUM_SEQS="8"
ENV VLLM_GPU_MEMORY_UTILIZATION="0.90"

ENV VLLM_HOST="0.0.0.0"
ENV VLLM_PORT="8000"

ENV VLLM_STARTUP_TIMEOUT_SECONDS="900"
ENV VLLM_REQUEST_TIMEOUT_SECONDS="300"


USER dockeruser