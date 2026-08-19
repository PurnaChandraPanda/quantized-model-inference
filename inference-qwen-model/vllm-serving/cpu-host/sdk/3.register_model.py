from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

def register_model(model_name, model_dir):

    # instantiate the ml client
    ml_client = MLClient.from_config(credential = DefaultAzureCredential())

    # register the model as custom model format
    run_model = Model(
        path=model_dir, 
        name=model_name,
        description="qwen 3.5 0.8b LLM model",
        type="custom_model",
    )

    ml_client.models.create_or_update(run_model)            

    print("Model registered successfully")

if __name__ == "__main__":
    # Name of ml model to be registered
    model_name = "Qwen35-08B" 
    # Path to the downloaded model directory
    ## with .amlignore at root of model_dir
    model_dir = "/tmp/qwen3.5-0.8b" 
    register_model(model_name, model_dir)
