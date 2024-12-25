## Model quantized format
- .gguf

## Details
- Host quantized gguf format llm model on llama-cpp-python web server package.
- llama-cpp-python[server] module is packaged inside docker image.
- [llama-cpp-python/docker](https://github.com/abetlen/llama-cpp-python/tree/main/docker) discusses a lot on different variety of docker image configuration.
- This sample talks about the azureml extension way of [openblas docker image](https://github.com/abetlen/llama-cpp-python/blob/main/docker/openblas_simple/Dockerfile).
- In azureml enviornment, leveraging one of base images from azureml and then taking advantage of rest of llama-cpp-python docker files.
- With [llama-cpp-python[server]](https://github.com/abetlen/llama-cpp-python/blob/main/docs/server.md), the advantage is that it offers an OpenAI API compatible web server endpoints.
- The azureml scoring script takes cognizance of compability points with azureml as well as [llama-cpp-python[server]](https://llama-cpp-python.readthedocs.io/en/latest/server/).

## Pre-requisites
- In azureml compute instance, use v2 conda env and updated `azure-ai-ml` package.
```
conda activate azureml_py310_sdkv2
pip install -U azure-ai-ml
```
- This sample is tested with python package: `llama-cpp-python==0.3.2`, and also `0.3.5` (latest one).

## Run 
### Inference phi3 gguf model
[phi3-gguf-online-endpoint.ipynb](./inference-phi3q-gguf/phi3-gguf-online-endpoint.ipynb)
- Register model asset as custom model
- Register environment asset as custom environment
- Create managed online endpoint
- Create managed online deployment for the endpoint
- Test the online endpoint

### Inference tinyllama1.1b gguf model
[tinyllama-gguf-online-endpoint.ipynb](./inference-tinyllama1.1b-gguf/tinyllama-gguf-online-endpoint.ipynb)
- Register model asset as custom model
- Register environment asset as custom environment
- Create managed online endpoint
- Create managed online deployment for the endpoint
- Test the online endpoint

