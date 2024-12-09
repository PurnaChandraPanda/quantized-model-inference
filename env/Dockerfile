FROM mcr.microsoft.com/azureml/curated/minimal-py311-inference:latest

# Switch to root user
USER root

# Install blas related packages + llama-cpp-python
RUN apt update && apt install -y libopenblas-dev ninja-build build-essential pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/*
RUN apt-get update && apt-get install -y net-tools
    
RUN python -m pip install --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context

RUN CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install llama_cpp_python

# Install ml endpoint scoring related packages
RUN pip install mlflow
RUN pip install transformers

# Run the server
CMD python3 -m llama_cpp.server
