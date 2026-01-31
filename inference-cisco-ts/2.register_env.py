from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, BuildContext
from azure.identity import DefaultAzureCredential

def register_environment(env_name: str):

    # instantiate the ml client
    ml_client = MLClient.from_config(credential = DefaultAzureCredential())

    # create environment from dockerfile
    custom_env = Environment(
        name=env_name,
        description="Environment for Cisco Time Series Model Inference",
        build=BuildContext(
            path="./env"  # Path to the directory containing Dockerfile and dependencies
        )
    )
    
    # Register the custom environment
    ml_client.environments.create_or_update(custom_env)

    print("Environment is registered and be built now.")

if __name__ == "__main__":
    env_name = "cisco-ts-env"
    register_environment(env_name)