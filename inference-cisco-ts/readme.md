**Topics covered**

- [Download model from hf](#download-model-from-hf)
- [Inference the ts model locally](#inference-the-ts-model-locally)
- [Register model in ml workspace](#register-model-in-ml-workspace)
- [Register and build environment in ml workspace](#register-and-build-environment-in-ml-workspace)
- [Create/ test - online endpoint/ deployment](#create-test---online-endpoint-deployment)


## Download model from hf
```
python 0.download_model.py
```

## Inference the ts model locally

- Create a local GPU based ml compute instance with sku `Standard_NC24ads_A100_v4`
- Set system PATH for cuda-12.2
```
conda activate azureml_py310_sdkv2
export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}  
source ~/.bashrc
conda activate azureml_py310_sdkv2
nvcc --version
```

- Clone the repo (hack for identity enabled azureml workspace)
```
cd ~
git clone https://github.com/splunk/cisco-time-series-model.git
sudo mv ~/cisco-time-series-model ~/cloudfiles/code/Users/pupanda/gpt-use-case/foundation-models/cisco-ts-model/
```

- Install the package dependecies in local
```
cd cisco-time-series-model
pip install -r requirements.txt
pip install torch
```

- Local inference the model `cisco-ai/cisco-time-series-model-1.0-preview`
  - It depends on `timesfm` package for inference action.
  - Point to the directory where `modeling` directory is kept, where timesfm related inferencing code is kept.
  - Run the notebook `0.1.local-ts-inference.ipynb` for local run.

```python
from modeling import CiscoTsmMR, TimesFmHparams, TimesFmCheckpoint
```

```python

    # Check if cuda is available
    device = "gpu" if torch.cuda.is_available() else "cpu"

    # Define hyperparameters
    hparams = TimesFmHparams(
        num_layers=50,
        use_positional_embedding=False,
        backend=device,
    )

    # Local path of model
    model_dir = Path("cisco-ts-model-1.0-preview")
    model_path = model_dir / "torch_model.pt"   # adjust if your file has a different name

    # Load model checkpoint
    ckpt = TimesFmCheckpoint(path = model_path)

    # Initialize the model
    model = CiscoTsmMR(
        hparams=hparams,
        checkpoint=ckpt,
        use_resolution_embeddings=True,
        use_special_token=True,
    )
```

```python
# Model Inference
forecast_preds = model.forecast(input_series_1, horizon_len=128)

# Access forecast mean and quantiles of each series
mean_forecast = forecast_preds[0]['mean'] # (128,)
quantiles = forecast_preds[0]['quantiles'] # dict with keys as quantile levels (0.1, 0.2, ...., 0.9) and values as (128,) numpy arrays
```

## Register model in ml workspace
```
python 1.register_model.py
```

## Register and build environment in ml workspace
- Environment `Dockerfile` leverages azurelml cuda curated base image `mcr.microsoft.com/azureml/curated/minimal-py312-cuda12.4-inference:latest`.
- As `timesfm` is not compatible with python-3.12, `Dockerfile` is updated to create one more conda environment with base python version set as `3.11`.

```
python 2.register_env.py
```

## Create/ test - online endpoint/ deployment
- Actual cisco timeseries model `load` and `forecast` related wrapper logic is kept in [modeling](https://github.com/splunk/cisco-time-series-model/tree/main/1.0-preview/modeling) folder.
- For online endpoint inferencing to work, copy the [modeling](https://github.com/splunk/cisco-time-series-model/tree/main/1.0-preview/modeling) folder into `onlinescoring` folder for score script flow to work.
- Define the score script `init()` with model initialization logic.
```python

from modeling import CiscoTsmMR, TimesFmHparams, TimesFmCheckpoint

    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "cisco-ts-model-1.0-preview/torch_model.pt"
    )

    # Check if cuda is available
    device = "gpu" if torch.cuda.is_available() else "cpu"

    # Define hyperparameters
    hparams = TimesFmHparams(
        num_layers=50,
        use_positional_embedding=False,
        backend=device,
    )

    # Load model checkpoint
    ckpt = TimesFmCheckpoint(path = model_path)

    # Initialize the model
    model = CiscoTsmMR(
        hparams=hparams,
        checkpoint=ckpt,
        use_resolution_embeddings=True,
        use_special_token=True,
    )
```

- Define the score script `run()` with model forecast and results parsing.
```python
preds = model.forecast(series_arr, horizon_len=horizon)

result = {"results": [_pack_one(p) for p in preds]}
response = json.dumps(result)
```

- Run the notebook (point to conda env: azureml_py310_sdkv2)
```
3.cisco_ts_online_ep.ipynb
```

- In deployment logs, `init` section results appear with model initialization details.

```
2026-01-31 07:38:35,319 I [119] azmlinfsrv.user_script - Found user script at /var/azureml-app/onlinescoring/score.py
2026-01-31 07:38:35,319 I [119] azmlinfsrv.user_script - run() is not decorated. Server will invoke it with the input in JSON string.
2026-01-31 07:38:35,319 I [119] azmlinfsrv.user_script - Invoking user's init function
2026-01-31 07:38:38,699 I [119] azmlinfsrv.print - Model loaded from path: /var/azureml-app/azureml-models/cisco-ts/2/cisco-ts-model-1.0-preview/torch_model.pt on gpu.
2026-01-31 07:38:38,700 I [119] azmlinfsrv.user_script - Users's init has completed successfully
```

- In deployment logs, `run` section results appear with model forecast telemetry.

1)
```
2026-01-31 07:44:20,823 I [119] azmlinfsrv.print - Received data:  {"values": [100.0, 101.2, 100.9, 102.5, 101.7, 103.1], "horizon": 5}
2026-01-31 07:44:21,128 I [119] azmlinfsrv - POST /score 200 305.413ms 1121
2026-01-31 07:44:21,128 I [119] gunicorn.access - 127.0.0.1 - - [31/Jan/2026:07:44:21 +0000] "POST /score HTTP/1.0" 200 1121 "-" "azure-ai-ml/1.31.0 azsdk-python-core/1.37.0 Python/3.10.19 (Linux-6.8.0-1044-azure-x86_64-with-glibc2.35)"
```

2)
```
2026-01-31 07:47:14,264 I [119] azmlinfsrv.print - Received data:  {"series": [[100.0, 101.2, 100.9, 102.5, 101.7, 103.1], [55.0, 56.1, 55.5, 56.7, 57.2, 57.8]], "horizon": 5}
2026-01-31 07:47:14,301 I [119] azmlinfsrv - POST /score 200 36.599ms 2226
2026-01-31 07:47:14,301 I [119] gunicorn.access - 127.0.0.1 - - [31/Jan/2026:07:47:14 +0000] "POST /score HTTP/1.0" 200 2226 "-" "azure-ai-ml/1.31.0 azsdk-python-core/1.37.0 Python/3.10.19 (Linux-6.8.0-1044-azure-x86_64-with-glibc2.35)"
```
