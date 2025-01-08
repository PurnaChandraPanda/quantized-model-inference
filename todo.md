### 1
update engine logic to accommodate other llm lightweight inference servers, such as:
- vllm 
- llama.cpp (https://formulae.brew.sh/formula/llama.cpp#default)
- ollama

### 2
along with gguf, add support to host mlflow model format too

### 3
- add logic, how to bring in models/ from hf or model catalog
- also add logic how to quantize public models

### 4
explain runtime logs generated in endpoint - for cpu as well as gpu
