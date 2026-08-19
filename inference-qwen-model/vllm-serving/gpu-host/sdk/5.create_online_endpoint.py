from azure.ai.ml import MLClient
from azure.identity import (
    DefaultAzureCredential,
)
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
)

# Create handle for azureml workspace
credential = DefaultAzureCredential()

## Initialize MLClient
ml_client = MLClient.from_config(
    credential
)

# Online endpoint name
online_endpoint_name = "qwen08b-gpu1-endpoint"

# managed endpoint
endpoint = ManagedOnlineEndpoint(
    name=online_endpoint_name,
    description="Online endpoint for qwen model",
    auth_mode="key",
)

# managed endpoint create async call
ml_client.begin_create_or_update(endpoint).wait()

print("created online endpoint: ", online_endpoint_name)
