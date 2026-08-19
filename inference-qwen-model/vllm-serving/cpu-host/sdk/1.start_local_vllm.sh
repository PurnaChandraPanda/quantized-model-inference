# Reserves 4 GB of system RAM for vLLM's CPU KV cache.
export VLLM_CPU_KVCACHE_SPACE=4
export VLLM_CPU_OMP_THREADS_BIND=0-7
export TOKENIZERS_PARALLELISM=false

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4

# local model path
model_dir=/tmp/qwen3.5-0.8b
llm_model=Qwen/Qwen3.5-0.8B
# model parameters
## input/prompt tokens + generated/output tokens <= 4096
max_model_length=4096
## max active/ batched sequences in a single request; 
## set to 1 for single sequence inference; set to >1 for batched inference (observe RAM/ latency/ throughput tradeoff)
max_num_seqs=1

vllm serve $model_dir \
  --served-model-name $llm_model \
  --dtype bfloat16 \
  --max-model-len $max_model_length \
  --max-num-seqs $max_num_seqs \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code


