import os, subprocess, json
import mlflow
import argparse
import time
from huggingface_hub import snapshot_download, hf_hub_download
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import ModelType
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential

def download_hf_model(model_dir):
    print("Downloading model from huggingface hub")
    print("model_dir ", model_dir)
    
    # Download the model from huggingface hub.
    ## This will create {model_dir} directory, and then download the model to that directory.
    snapshot_download(
        repo_id="unsloth/DeepSeek-R1-GGUF",
        local_dir=model_dir,
        allow_patterns=["*UD-IQ1_S*"], # Select quant type UD-IQ1_S for 1.58bit        
    )
    
    print("Downloaded model from huggingface hub")
    
    # mlflow run
    with mlflow.start_run() as run:
        # Set the timeout higher (in seconds) for the artifact upload
        os.environ["AZUREML_ARTIFACTS_DEFAULT_TIMEOUT"] = "3600"
        # Log the model
        mlflow.log_artifact(local_path=model_dir)

        # Print the model details
        print_model_details(model_dir)

        # Fetch the artifact uri root directory
        print(f"Artifact uri: {mlflow.get_artifact_uri()}")

        # print(f"mlflow.is_tracking_uri_set(): {mlflow.is_tracking_uri_set()}")

        # this will register the model in the mlflow format
        # mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/{model_dir}", 
        #                       name="deepseek-r1-gguf") 
                
        # in serverless: try AzureMLOnBehalfOfCredential
        ## will work: if only caller passed UserIdentityConfiguration() in the job definition
        credential = AzureMLOnBehalfOfCredential()
        # print(credential.get_token('https://management.azure.com/.default'))

        # read common runtime azureml context        
        _aml_context = json.loads(os.environ["AZUREML_CR_AZUREML_CONTEXT"])

        # instantiate the ml client
        ml_client = MLClient(credential, 
                             _aml_context["subscription_id"],
                             _aml_context["resource_group"],
                             _aml_context["workspace_name"])

        # register the model as custom model format
        ## docs reference: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-models?view=azureml-api-2&tabs=python#job-output
        run_model = Model(
            path=f"runs:/{run.info.run_id}/{model_dir}",
            name="deepseek-r1-gguf",
            description="Model created from run.",
            type="custom_model",
        )

        ml_client.models.create_or_update(run_model)            

    print("Model registered successfully")
    

def print_model_details(model_dir):
    print("model details >>>")
    # Define the command
    command = ["du", "-sh", f"{model_dir}/*"]
    # Run the command
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # Print the output
    print("Standard Output:")
    print(result.stdout)
    # Print any errors
    if result.stderr:
        print("Standard Error:")
        print(result.stderr)

def main(args):
    # Get all environment variables
    env_vars = os.environ

    # Print each key-value pair
    for key, value in env_vars.items():
        print(f"{key}: {value}")
    
    print("we re in main")
    download_hf_model(args.model_dir)

def parse_args():
    # setup argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-dir", type=str, default="./", help="output directory for model"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Print the key-value pairs
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    return args

# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # call main function
    main(args)