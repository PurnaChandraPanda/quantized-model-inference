FROM mcr.microsoft.com/azureml/curated/minimal-py312-cuda12.4-inference:latest

# Setup enviroment variables for new conda environment
ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/py311
ENV AZUREML_CONDA_DEFAULT_ENVIRONMENT=$AZUREML_CONDA_ENVIRONMENT_PATH
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH
ENV LD_LIBRARY_PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH

# Switch to root to install packages
USER root
COPY requirements.txt .
COPY conda.yaml .
RUN conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda.yaml -q && \
    rm conda.yaml && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y   
USER dockeruser
