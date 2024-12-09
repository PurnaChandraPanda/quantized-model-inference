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
- This sample is tested with python package: llama-cpp-python==0.3.2.

## Run 
[phi3-gguf-online-endpoint.ipynb](./inference-phi3q-gguf/phi3-gguf-online-endpoint.ipynb)
- Register model asset as custom model
- Register environment asset as custom environment
- Create managed online endpoint
- Create managed online deployment for the endpoint
- Test the online endpoint

## Known issue
With llama-cpp-python==0.3.4, following error is noticed in llama_cpp server side.

```
Exception: 'coroutine' object is not callable
Traceback (most recent call last):
  File "/azureml-envs/minimal/lib/python3.11/site-packages/llama_cpp/server/errors.py", line 173, in custom_route_handler
    response = await original_route_handler(request)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/azureml-envs/minimal/lib/python3.11/site-packages/fastapi/routing.py", line 301, in app
    raw_response = await run_endpoint_function(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/azureml-envs/minimal/lib/python3.11/site-packages/fastapi/routing.py", line 212, in run_endpoint_function
    return await dependant.call(**values)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/azureml-envs/minimal/lib/python3.11/site-packages/llama_cpp/server/app.py", line 491, in create_chat_completion
    llama = llama_proxy(body.model)
            ^^^^^^^^^^^^^^^^^^^^^^^
TypeError: 'coroutine' object is not callable
INFO:     ::1:42174 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
```
Filed [issue](https://github.com/abetlen/llama-cpp-python/issues/1857#issue-2726895443) with owner llama-cpp-python for investigation.