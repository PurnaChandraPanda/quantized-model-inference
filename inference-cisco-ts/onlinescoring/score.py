import os, json
import pandas as pd
import torch
from modeling import CiscoTsmMR, TimesFmHparams, TimesFmCheckpoint


# Set a default horizon if none is provided
H_DEFAULT = 7

def init():
    global model

    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    # Please provide your model's folder name if there is one
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

    print(f"Model loaded from path: {model_path} on {device}.")


def _to_series_array(payload):
    """
    Accept either:
      - {"values": [...], "horizon": 128}
      - {"series": [[...], [...]], "horizon": 128}
    Values can be floats/ints; Nones will be forward-filled.
    Returns: (list_of_np_arrays, horizon)
    """
    if "horizon" in payload:
        H = int(payload["horizon"])
    else:
        H = H_DEFAULT

    def _prep_one(arr):
        s = pd.Series(arr, dtype="float32")
        s = s.ffill()  # handle case when there are gaps
        return s.to_numpy()

    if "values" in payload:
        return ([_prep_one(payload["values"])], H)
    elif "series" in payload:
        return ([ _prep_one(a) for a in payload["series"] ], H)
    else:
        raise ValueError("Request must contain either 'values' or 'series'.")

def run(raw_data):
    """
    Called per request by the AzureML inference HTTP server.
    Body is a JSON string.
    """
    print("Received data: ", raw_data)

    # Parse the JSON request
    payload = json.loads(raw_data)

    # Convert input to series array
    series_arr, horizon = _to_series_array(payload)
    
    # Forecast with Cisco model: returns list of dicts with 'mean' and 'quantiles'
    preds = model.forecast(series_arr, horizon_len=horizon)

    def _pack_one(item):
        # quantiles keys are floats (e.g., 0.1, ...). Convert keys to strings for JSON.
        q = { str(k): v.tolist() for k, v in item["quantiles"].items() }
        return {"mean": item["mean"].tolist(), "quantiles": q}

    if len(preds) == 1:
        result = _pack_one(preds[0])
    else:
        result = {"results": [_pack_one(p) for p in preds]}

    response = json.dumps(result)

    return response
