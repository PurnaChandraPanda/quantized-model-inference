# Local development - to inference qwen model
## create/ activate conda env for local test

- Create conda env
```
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create -n vllm python=3.12 -y
```

- Activate conda env
```
conda activate vllm
python --version
```

- Install vllm - cuda version

Install latest vllm cpu version - `0.27.1`.
Per docs only this package is pre-built for `cuda 12.9`.

As local A100 has `cuda-13.0`, need to downgrade local into `cuda-12.9` first.


```
# Check local cuda version

nvidia-smi

readlink -f /usr/local/cuda || true
```

```
# Keep cuda-13.0 installed. Add 12.9 beside it.

cd /tmp

wget \
  https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda-repo-ubuntu2204-12-9-local_12.9.0-575.51.03-1_amd64.deb

sudo dpkg -i \
  cuda-repo-ubuntu2204-12-9-local_12.9.0-575.51.03-1_amd64.deb

sudo cp \
  /var/cuda-repo-ubuntu2204-12-9-local/cuda-*-keyring.gpg \
  /usr/share/keyrings/

sudo apt-get update

sudo apt-get install -y cuda-toolkit-12-9
```

```
# Check if cuda-12.9 is visible as installed

ls -la /usr/local | grep cuda
```

```
# Set the PATH for latest cuda toolkit `12.9`

export PATH=/usr/local/cuda-12.9/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

source ~/.bashrc

nvcc --version
```

- Activate the conda env and navigate into actual app folder

```
conda activate vllm

export PATH=/usr/local/cuda-12.9/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

source ~/.bashrc

nvcc --version

conda activate vllm
```

- Install vllm==0.27.1 cuda-12.9 variant

```
pip install \
  "vllm==0.27.1+cu129" \
  --extra-index-url https://wheels.vllm.ai/0.27.1/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129
```

- Install packages for local development
```
pip install azure-ai-ml
```

---

## remove conda env (for testing, might need to clean up)

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
export PATH=/usr/local/cuda-12.9/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

source ~/.bashrc

nvcc --version

conda activate vllm
```

```
./1.start_local_vllm.sh
```

Update the `.env` file with vllm openai base URL details.

On another terminal, test the client app

```
python 2.test_chat_completion.py
```

One last check: before going for cloud side.

## local test with azmlinfsrv

```
pip install azureml-inference-server-http==1.5.1
```

```
export PROJECT_ROOT="$(pwd)"
export AZUREML_ENTRY_SCRIPT="${PROJECT_ROOT}/../../cpu-host/sdk/onlinescoring/score.py"
export AZUREML_MODEL_DIR="/tmp/qwen3.5-0.8b"

azmlinfsrv \
  --entry_script "$AZUREML_ENTRY_SCRIPT" \
  --model_dir "$AZUREML_MODEL_DIR" \
  --port 31311
```

You might encounter opentelemtry package related errors.

### remove the opentelemtry packages

```
pip uninstall -y \
  azure-monitor-opentelemetry \
  azure-monitor-opentelemetry-exporter \
  azure-core-tracing-opentelemetry \
  opentelemetry-resource-detector-azure
```

### install opentelemtry packages again compatible with `azureml-inference-server-http==1.5.1`

```
pip install \
  --no-cache-dir \
  --force-reinstall \
  "azure-core-tracing-opentelemetry==1.0.0b13" \
  "azure-monitor-opentelemetry==1.8.9" \
  "azure-monitor-opentelemetry-exporter==1.0.0b56" \
  "opentelemetry-resource-detector-azure==0.1.5"
```

### try again

```
export PROJECT_ROOT="$(pwd)"
export AZUREML_ENTRY_SCRIPT="${PROJECT_ROOT}/../../cpu-host/sdk/onlinescoring/score.py"
export AZUREML_MODEL_DIR="/tmp/qwen3.5-0.8b"

azmlinfsrv \
  --entry_script "$AZUREML_ENTRY_SCRIPT" \
  --model_dir "$AZUREML_MODEL_DIR" \
  --port 31311
```

You will not notice any openetelemetry errors, but some other errors now. It means dependencies are fine.


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
python 7.test_online_endpoint.py

or

python 7.1.requests_.test_online.endpoint.py --endpoint-name qwen08b-cpu1-endpoint
```

- Review the logs for endpoint on LLM inference transaction details for request processed

```
2026-08-19 15:49:15,974 I [119] azmlinfsrv.print - Sending request to local vLLM server: url=http://127.0.0.1:8000/v1/chat/completions, model=Qwen/Qwen3.5-0.8B, task_type=TaskType.CONVERSATIONAL, prompt_number=0
(APIServer pid=148) INFO:     127.0.0.1:47754 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=148) INFO:     127.0.0.1:47754 - "POST /tokenize HTTP/1.1" 200 OK
2026-08-19 15:49:19,804 I [119] azmlinfsrv.print - Inference Results: prompt_num=0, prompt_tokens=36, completion_tokens=91, generated_token_ids=90, inference_time_ms=3825.81, time_per_token_ms=42.04
2026-08-19 15:49:19,804 I [119] azmlinfsrv - POST /score 200 4106.910ms 555
2026-08-19 15:49:19,804 I [119] gunicorn.access - 127.0.0.1 - - [19/Aug/2026:15:49:19 +0000] "POST /score HTTP/1.0" 200 555 "-" "azure-ai-ml/1.34.1 azsdk-python-core/1.41.0 Python/3.11.15 (Linux-6.8.0-1059-azure-x86_64-with-glibc2.35)"
(APIServer pid=148) INFO 08-19 15:49:22 [loggers.py:310] Engine 000: Avg prompt throughput: 3.6 tokens/s, Avg generation throughput: 9.1 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
```

# References
- [VLLM: get started](https://docs.vllm.ai/en/stable/getting_started/installation/index.html)

- [VLLM installation in gpu](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)

- [Qwen3.5-08B model](https://huggingface.co/Qwen/Qwen3.5-0.8B)




