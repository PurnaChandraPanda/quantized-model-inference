from azure.ai.ml import MLClient
from azure.identity import (
    DefaultAzureCredential,
)
from azure.ai.ml.entities import (
    CodeConfiguration, ManagedOnlineDeployment, OnlineRequestSettings
)

# Create handle for azureml workspace
credential = DefaultAzureCredential()

## Initialize MLClient
ml_client = MLClient.from_config(
    credential
)

# Online endpoint name
online_endpoint_name = "qwen08b-gpu1-endpoint"

# Online endpoint deployment name
deployment_name = "deploy01"

# Read existing model and environment
llm_model = "Qwen35-08B@latest"
llm_env = "qwen-gpu-env@latest"

# managed endpoint deployment
demo_deployment = ManagedOnlineDeployment(
    name=deployment_name,
    endpoint_name=online_endpoint_name,
    model=llm_model,
    environment=llm_env,
    code_configuration=CodeConfiguration(
        code="../../cpu-host/sdk/onlinescoring",
        scoring_script="score.py",
    ),
    instance_type="Standard_NC24ads_A100_v4", # gpu machine type
    instance_count=1,
    request_settings=OnlineRequestSettings(
        request_timeout_ms=120000,
    ),
    egress_public_network_access="disabled", # for private network case from managed compute
)

# wait for managed deployment create to complete
ml_client.online_deployments.begin_create_or_update(demo_deployment).result()

# Check if the deployment is healthy, then set traffic to 100% for the deployment
deployment = ml_client.online_deployments.get(
                name=deployment_name, 
                endpoint_name=online_endpoint_name)

if deployment.provisioning_state == "Succeeded":
    print(f"\nDeployment {deployment_name} is healthy. Setting traffic to 100% for this deployment.\n")

    # Get online endpoint object
    endpoint = ml_client.online_endpoints.get(name=online_endpoint_name)
    
    # Set the traffic %ge for deployment
    endpoint.traffic = {str(deployment.name): 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    print("endpoint + deployment: ready for use")

else:
    print("endpoint deployment provisioning state: ", deployment.provisioning_state)
