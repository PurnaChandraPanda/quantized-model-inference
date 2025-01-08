# First stage: cuda setup
ARG CUDA_IMAGE="12.2.0-devel-ubuntu22.04"
FROM nvidia/cuda:${CUDA_IMAGE} AS cuda-stage

# Switch to root user
USER root

# RUN hostname
# RUN ls /usr/local
# RUN which nvcc
# RUN nvcc --version

ENV CUDA_HOME=/usr/local/cuda-12.2
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/compat:${CUDA_HOME}/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
ENV CUDAToolkit_ROOT=/usr/local/cuda-12.2

# Install cuda, openblas related packages
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y git build-essential
RUN apt-get install -y python3 python3-pip gcc wget
RUN apt-get install -y ocl-icd-opencl-dev opencl-headers clinfo
RUN apt-get install -y libclblast-dev libopenblas-dev
RUN mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Second stage: azureml base image setup
FROM mcr.microsoft.com/azureml/curated/minimal-py311-inference:latest

# Copy CUDA-related files from the first stage
COPY --from=cuda-stage /usr/local/cuda /usr/local/cuda

# RUN ls /usr/local

# Set environment variables
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Switch to root user again
USER root

RUN apt-get update && apt-get install -y net-tools

# Setting build related env vars
ENV GGML_CUDA=1
ENV FORCE_CMAKE=1

# Install depencencies
RUN python3 -m pip install --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context

# Install llama-cpp-python (build with cuda)
# Cuda arch: 80 is used for A100 machines in the args param {DCMAKE_CUDA_ARCHITECTURES}.
# Note: For respective cuda arch, the respective GPU SKU machine mapping cuda arch value should be used.
RUN CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_FORCE_CUBLAS=on -DLLAVA_BUILD=off -DCMAKE_CUDA_ARCHITECTURES=80" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir

# Install ml endpoint scoring related packages
RUN pip install mlflow
RUN pip install transformers

# Run the server
CMD python3 -m llama_cpp.server
