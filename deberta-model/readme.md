
- From azureml [huggingfacy registry](https://ml.azure.com/registries/HuggingFace/models/microsoft-deberta-v3-xsmall/version/6), the model `microsoft-deberta-v3-xsmall` is not allowed for explicit model download. I see maap deployment option is marked for it - by the owners. 
- As a user, we do not have access to view its artifacts, so can't download or register it.
- This model owner only allowed to deploy it as `managed endpoint` deploy only.

- Just in case same model version is needed, it needs to be downloaded cleanly from hf portal [microsoft/deberta-v3-xsmall](https://huggingface.co/microsoft/deberta-v3-xsmall)· Hugging Face.
- The [snapshot_download](./inference-deepseekr1-gguf-gpu/job-register-hf-model/src/main.py#L17) api is to be used for download and register then.


- Otherwise, if some other deberta series model needed from azureml platform directly, there's different placeholder to download from, i.e. azureml registry.

## How to use it?

```
conda activate azureml_py310_sdkv2
pip install -U azure-ai-ml
```

```
python model_download_register.py
```
