from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model

credential = DefaultAzureCredential()

# ✅ Workspace client
ml_client = MLClient.from_config(
    credential
)

# ✅ Use a temporary client directly for registry access
registry = MLClient(
    credential,
    # registry_name="huggingface"
    registry_name="azureml"
)

registry_model = registry.models.get(
    # name="microsoft-deberta-v3-xsmall",
    # version="6"
    name="microsoft-deberta-base",
    version="18"
)

# print(registry_model)
# print(registry_model.type)

print("downloading model")
# Download model
registry.models.download(
    name = registry_model.name,
    version = registry_model.version,
    download_path = f"/tmp/"
)
print("downloaded model ✅✅✅")

# ✅ Register into workspace
workspace_model = Model(
    name="deberta-new",
    path=f"/tmp/{registry_model.name}/mlflow_model_folder",
    type="mlflow_model",
    description="Imported from Azure ML Model Catalog"
)
registered_model = ml_client.models.create_or_update(workspace_model)
print(f"Registered: {registered_model.name}, version: {registered_model.version}")