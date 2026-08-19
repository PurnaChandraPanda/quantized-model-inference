#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import requests
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Invoke an Azure ML managed online endpoint using key authentication."
    )

    parser.add_argument(
        "--endpoint-name",
        required=True,
        help="Azure ML online endpoint name.",
    )

    parser.add_argument(
        "--request-file",
        default="payload/sample-request.json",
        help="Path to the JSON request payload. Default: payload/sample-request.json",
    )

    parser.add_argument(
        "--deployment-name",
        default=None,
        help=(
            "Optional deployment name. When supplied, the request is routed "
            "directly to that deployment."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="HTTP request timeout in seconds. Default: 300",
    )

    return parser.parse_args()


def load_request_payload(request_file: str) -> Dict[str, Any]:
    request_path = Path(request_file).expanduser().resolve()

    if not request_path.is_file():
        raise FileNotFoundError(
            f"Request file does not exist: {request_path}"
        )

    try:
        with request_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Request file contains invalid JSON: {request_path}. "
            f"Line={exc.lineno}, column={exc.colno}, error={exc.msg}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            "The request JSON must contain a JSON object at its root."
        )

    print(f"Loaded request payload from: {request_path}")
    return payload


def create_ml_client() -> MLClient:
    credential = DefaultAzureCredential()

    # Reads subscription ID, resource group, and workspace name
    # from config.json in the current directory or its parent directories.
    return MLClient.from_config(credential=credential)


def get_endpoint_connection_details(
    ml_client: MLClient,
    endpoint_name: str,
) -> tuple[str, str]:
    endpoint = ml_client.online_endpoints.get(name=endpoint_name)

    if not endpoint.scoring_uri:
        raise RuntimeError(
            f"Endpoint '{endpoint_name}' does not have a scoring URI."
        )

    credentials = ml_client.online_endpoints.get_keys(name=endpoint_name)
    endpoint_key = credentials.primary_key

    if not endpoint_key:
        raise RuntimeError(
            f"Primary key was not returned for endpoint '{endpoint_name}'. "
            "Confirm that the endpoint uses key authentication."
        )

    print(f"Endpoint name: {endpoint.name}")
    print(f"Endpoint state: {endpoint.provisioning_state}")
    print(f"Scoring URI: {endpoint.scoring_uri}")
    print("Endpoint key retrieved successfully.")

    return endpoint.scoring_uri, endpoint_key


def invoke_endpoint(
    scoring_uri: str,
    endpoint_key: str,
    payload: Dict[str, Any],
    timeout: int,
    deployment_name: str | None = None,
) -> Any:
    headers = {
        "Authorization": f"Bearer {endpoint_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Supplying this header bypasses endpoint traffic allocation and routes
    # the request directly to the specified deployment.
    if deployment_name:
        headers["azureml-model-deployment"] = deployment_name
        print(f"Routing request to deployment: {deployment_name}")
    else:
        print("Using endpoint traffic configuration for deployment routing.")

    response = requests.post(
        scoring_uri,
        headers=headers,
        json=payload,
        timeout=timeout,
    )

    print(f"HTTP status: {response.status_code}")
    print(f"Request ID: {response.headers.get('x-request-id')}")

    if not response.ok:
        print("Endpoint error response:")
        print(response.text)
        response.raise_for_status()

    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        return response.text


def main() -> None:
    args = parse_arguments()

    payload = load_request_payload(args.request_file)
    ml_client = create_ml_client()

    scoring_uri, endpoint_key = get_endpoint_connection_details(
        ml_client=ml_client,
        endpoint_name=args.endpoint_name,
    )

    result = invoke_endpoint(
        scoring_uri=scoring_uri,
        endpoint_key=endpoint_key,
        payload=payload,
        timeout=args.timeout,
        deployment_name=args.deployment_name,
    )

    print("\nInference response:")

    if isinstance(result, (dict, list)):
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(result)


if __name__ == "__main__":
    main()