FROM mcr.microsoft.com/azureml/curated/minimal-py311-inference:latest

# Switch to root to install packages
USER root

## Install vllm related pre-requisites
RUN apt-get update && \
    apt-get install -y \
        libnuma-dev \
        libtcmalloc-minimal4 && \
    rm -rf /var/lib/apt/lists/*

# Setup enviroment variables for new conda environment
ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/py311
ENV AZUREML_CONDA_DEFAULT_ENVIRONMENT=$AZUREML_CONDA_ENVIRONMENT_PATH
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH
ENV VLLM_CPU_KVCACHE_SPACE=4
ENV VLLM_CPU_OMP_THREADS_BIND=0-7
ENV TOKENIZERS_PARALLELISM=false
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4

COPY requirements.txt .
COPY conda.yml .

RUN conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda.yml -q && \
    rm conda.yml && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y   

# Install the vLLM CPU distribution explicitly.
RUN ${AZUREML_CONDA_ENVIRONMENT_PATH}/bin/python \
    -m pip install \
    vllm==0.27.1+cpu \
    --extra-index-url https://wheels.vllm.ai/0.27.1/cpu/ \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --no-cache-dir

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


USER dockeruser
