from huggingface_hub import snapshot_download

def download_hf_model(model_dir):
    print("Downloading model from huggingface hub")
    print("model_dir ", model_dir)
    
    # Download the model from huggingface hub.
    ## This will create {model_dir} directory, and then download the model to that directory.
    snapshot_download(
        repo_id="cisco-ai/cisco-time-series-model-1.0-preview",
        local_dir=model_dir,
    )
    
    print("Downloaded model from huggingface hub")

if __name__ == "__main__":
    model_dir = "cisco-ts-model-1.0-preview"
    download_hf_model(model_dir)
    