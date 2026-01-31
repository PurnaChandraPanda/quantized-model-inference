from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

def register_model(model_name, model_dir):

    # instantiate the ml client
    ml_client = MLClient.from_config(credential = DefaultAzureCredential())

    # register the model as custom model format
    run_model = Model(
        path=model_dir, 
        name="cisco-ts",
        description="Cisco Time Series Model",
        type="custom_model",
    )

    ml_client.models.create_or_update(run_model)            

    print("Model registered successfully")

if __name__ == "__main__":
    # Name of ml model to be registered
    model_name = "cisco-ts" 
    # Path to the downloaded model directory
    ## with .amlignore at root of model_dir
    model_dir = "../../cisco-ts-model/cisco-ts-model-1.0-preview" 
    register_model(model_name, model_dir)