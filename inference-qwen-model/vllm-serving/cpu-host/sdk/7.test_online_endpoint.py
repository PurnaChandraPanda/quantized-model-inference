import json

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


ENDPOINT_NAME = "qwen08b-cpu1-endpoint"
DEPLOYMENT_NAME = "deploy01"
REQUEST_FILE = "payload/sample-request.json"


credential = DefaultAzureCredential()

ml_client = MLClient.from_config(
    credential=credential,
)

response = ml_client.online_endpoints.invoke(
    endpoint_name=ENDPOINT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    request_file=REQUEST_FILE,
)

print("Raw response:")
print(response)

try:
    parsed_response = json.loads(response)
    print("\nParsed response:")
    print(json.dumps(parsed_response, indent=2))
except (TypeError, json.JSONDecodeError):
    print("\nResponse was not a JSON string.")
