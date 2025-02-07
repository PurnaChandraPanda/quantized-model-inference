from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment, UserIdentityConfiguration, JobResourceConfiguration
from azure.identity import DefaultAzureCredential

# Initialize the MLClient
ml_client = MLClient.from_config(
    DefaultAzureCredential(),
)

# Define the environment name
env="hf-env@latest"

# Create job definition
command_job = command(
    code="src",
    command="python main.py --model-dir ${{inputs.model_dir}}",
    inputs={
        "model_dir": "models"
    },
    environment=env,
    experiment_name="job-register-hf-model",
    ## compute: keep it empty to use serverless computes
    resources=JobResourceConfiguration(instance_type="STANDARD_E8DS_V4", instance_count=1),   # override serverless compute configuration
    identity=UserIdentityConfiguration(), # use caller's identity for remote run
)

# Submit the command job
ml_client.jobs.create_or_update(
    command_job, 
)

print("Job submitted successfully")