## Note
- Download deepseek-r1 model quantized version of [gguf](https://huggingface.co/unsloth/DeepSeek-R1-GGUF) model.
- The gguf type "UD-IQ1_S" is on 1.58bit. This has fair amount of accuracy, but not best.

## Download oss model from hf
Prep environment to before model download activity.
```
cd job-register-hf-model
python build-hf-env.py
```

Download and register model.
```
cd job-register-hf-model
python hf-model-job.py
```

## Run notebook
Once model is registered by the job, [notebook](./deepseekr1-gguf-online-gpu.ipynb) can be run to deploy the deepseek-r1 gguf format model as managed endpoint via llama-cpp.
