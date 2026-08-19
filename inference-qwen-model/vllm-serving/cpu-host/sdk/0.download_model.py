from pathlib import Path
from huggingface_hub import snapshot_download

def download_hf_model(model_dir):
    print("Downloading model from huggingface hub")
    print("model_dir ", model_dir)
    
    # Download the model from huggingface hub.   
    downloaded_path = snapshot_download(
        repo_id="Qwen/Qwen3.5-0.8B",
        local_dir=model_dir,
    )
    
    print("Downloaded model from huggingface hub @ ", downloaded_path)

def main():
    # create temp model directory
    model_dir = Path("/tmp/qwen3.5-0.8b")
    model_dir.mkdir(parents=True, exist_ok=True)

    # download model
    download_hf_model(model_dir)

if __name__ == "__main__":
    main()
