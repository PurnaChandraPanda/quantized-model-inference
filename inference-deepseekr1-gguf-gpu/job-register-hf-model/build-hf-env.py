from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

# Initialize the MLClient
ml_client = MLClient.from_config(
    DefaultAzureCredential(),
)

# Define the environment name
env_name="hf-env"

# Make the environments query fail-safe
envs = ml_client.environments.list()
env = next((e for e in envs if e.name == env_name), None)

# Flag to indicate if the environment is modified. 
# Set it manually:: If True, old env version is used. If False, new env version is created.
is_env_earlierone = False

if env is not None and is_env_earlierone:
    print(f"Environment {env_name} already exists.")
else:
    print("create env")
    # Create the environment out of conda.yml
    env_conda = Environment(
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file="env/conda.yml",
        name=env_name,
        description="Environment created from a conda env",
    )
    ml_client.environments.create_or_update(env_conda)

    print("Environment is in creating phase")
