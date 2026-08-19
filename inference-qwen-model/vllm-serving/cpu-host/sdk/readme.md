# Local development - to inference qwen model
## create/ activate conda env for local test

- Create conda env
```
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create -n vllm python=3.11 -y
```

- Activate conda env
```
conda activate vllm
python --version
```

- Install packages for local development
```
pip install azure-ai-ml
pip install huggingface-hub
```

- Install vllm related dependencies
```
sudo apt-get update
sudo apt-get install -y \
    libnuma-dev \
    libtcmalloc-minimal4
```

- Install latest vllm cpu version - `0.27.1`

```
pip install "vllm==0.27.1+cpu" \
    --extra-index-url https://wheels.vllm.ai/0.27.1/cpu/ \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

---

## remove conda env

```
conda deactivate
conda remove --name vllm --all -y
```

## How to work

- Download model from hf

```
python 0.download_model.py
```

- Run local vllm server

```
./1.start_local_vllm.py
```

Update the `.env` file with vllm openai base URL details.

On another terminal, test the client app

```
python 2.test_chat_completion.py
```

As its working fine in local, let's start the preparation for hosting this model as "managed endpoint".

# Host the qwen model as managed endpoint

- Register the model asset

Clean up the `.cache` from model location

```
rm -rf /tmp/qwen3.5-0.8b/.cache
```

Register the model in ml workspace

```
python 3.register_model.py
```

- Register the environment asset

```
python 4.register_env.py
```

- Create online endpoint

```
python 5.create_online_endpoint.py
```

- Create online endpoint deployment/ update the traffic %ge

```
python 6.create_online_deployment.py
```

- Test online endpoint

```
python 7.test_online.endpoint.py

or

python 7.1.requests_.test_online.endpoint.py --endpoint-name qwen08b-cpu1-endpoint
```

- Review the logs for endpoint on LLM inference transaction details for request processed

```
2026-08-19 01:20:22,102 I [73] azmlinfsrv.print - CUDA is not available. Skipping nvidia-smi.
2026-08-19 01:20:22,102 I [73] azmlinfsrv.print - Sending request to local vLLM server: url=http://127.0.0.1:8000/v1/chat/completions, model=Qwen/Qwen3.5-0.8B, task_type=TaskType.CONVERSATIONAL, prompt_number=0
(APIServer pid=92) INFO 08-19 01:20:36 [loggers.py:310] Engine 000: Avg prompt throughput: 3.6 tokens/s, Avg generation throughput: 2.2 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.7%, Prefix cache hit rate: 0.0%
(APIServer pid=92) INFO:     127.0.0.1:50524 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=92) INFO:     127.0.0.1:50524 - "POST /tokenize HTTP/1.1" 200 OK
2026-08-19 01:20:41,839 I [73] azmlinfsrv.print - Inference Results: prompt_num=0, prompt_tokens=36, completion_tokens=82, generated_token_ids=81, inference_time_ms=19732.90, time_per_token_ms=240.65
2026-08-19 01:20:41,839 I [73] azmlinfsrv - POST /score 200 19737.772ms 459
2026-08-19 01:20:41,840 I [73] gunicorn.access - 127.0.0.1 - - [19/Aug/2026:01:20:41 +0000] "POST /score HTTP/1.0" 200 459 "-" "azure-ai-ml/1.34.1 azsdk-python-core/1.41.0 Python/3.11.15 (Linux-6.8.0-1059-azure-x86_64-with-glibc2.35)"
```






