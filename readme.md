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

## Cuda support in inferencing
- Docker image needs to be prepped with nvidia/cuda support. On top of it, azureml base image is used for azureml inferencing support.
- Current work on [docker with cuda in azureml](./env/gpu/Dockerfile) is an extension of [llama-cpp-python/docker/cuda](https://github.com/abetlen/llama-cpp-python/blob/main/docker/cuda_simple/Dockerfile).
- Note the critical part with right cuda support in A100 machines is in:
```
RUN CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_FORCE_CUBLAS=on -DLLAVA_BUILD=off -DCMAKE_CUDA_ARCHITECTURES=80" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir
```
- Over here, {DCMAKE_CUDA_ARCHITECTURES} holds importance which with `80` as value, is actually makes it cuda compatible.
- Additionally, on inferencing script side, a validation logic is added whether host is GPU enabled. If yes, add another param `--n_gpu_layers=-1` (in server end) as per [llama-cpp-python web server](https://github.com/abetlen/llama-cpp-python/tree/main?tab=readme-ov-file#openai-compatible-web-server) docs.

```python
        # Check in env if cuda is visible. 
        ## If yes, load model on GPUs for inferencing. Else, load model CPUs for inferencing.
        if 'NVIDIA_VISIBLE_DEVICES' in env:
            cmd = ["python", "-m", "llama_cpp.server", "--model", self._model_path, "--n_gpu_layers", "-1"]
            # Set the flag to True as its running on GPU
            self._is_cuda_visible = True
```
## Pre-requisites
- In azureml compute instance, use v2 conda env and updated `azure-ai-ml` package.
```
conda activate azureml_py310_sdkv2
pip install -U azure-ai-ml
```
- This sample is tested with python package: `llama-cpp-python==0.3.2`, and also `0.3.5` (latest one).

## Run 
In any model inferencing, following steps are carried out as base actions.
- Register model asset as custom model
- Register environment asset as custom environment
- Create managed online endpoint
- Create managed online deployment for the endpoint
- Test the online endpoint

### Inference phi3 gguf model
[phi3-gguf-online-endpoint.ipynb](./inference-phi3q-gguf/phi3-gguf-online-endpoint.ipynb)

### Inference tinyllama1.1b gguf model
[tinyllama-gguf-online-endpoint.ipynb](./inference-tinyllama1.1b-gguf/tinyllama-gguf-online-endpoint.ipynb)

### Inference phi3 gguf model - cuda support
[phi3-gguf-online-endpoint-gpu.ipynb](./inference-phi3q-gguf-gpu/phi3-gguf-online-endpoint-gpu.ipynb)

### Inference tinyllama1.1b gguf model - cuda support
[tinyllama-gguf-online-endpoint-gpu.ipynb](./inference-tinyllama1.1b-gguf-gpu/tinyllama-gguf-online-endpoint-gpu.ipynb)
