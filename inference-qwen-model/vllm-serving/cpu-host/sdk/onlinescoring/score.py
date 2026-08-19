"""
Azure ML scoring entry point for a locally hosted vLLM server.

Flow:
    Azure ML inference server
        -> score.py init()
        -> VllmEngine starts `vllm serve`
        -> score.py run()
        -> VllmEngine.run()
        -> VllmClient
        -> http://127.0.0.1:8000/v1/chat/completions
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from constants import ALL_TASKS, TaskType
from engine import VllmEngine
from inference_payload import InferencePayload, InferenceResult


logger = logging.getLogger(__name__)


# Azure ML calls init() once during scoring-worker initialization.
vllm_engine: Optional[VllmEngine] = None

# Default inference task if task_type isn't supplied in the request.
default_task_type: str = TaskType.CONVERSATIONAL


def init() -> None:
    """
    Initialize the vLLM inference engine.

    Azure ML calls this function when the scoring container starts.

    Initialization performs the following:

    1. Reads AZUREML_MODEL_DIR.
    2. Finds the Hugging Face model directory containing config.json.
    3. Reads vLLM settings from environment variables.
    4. Starts `vllm serve` as a subprocess through VllmEngine.
    5. Waits until the vLLM health endpoint responds.
    6. Verifies that the configured served model is registered.
    """

    global vllm_engine
    global default_task_type

    logger.info("score.py init() started")

    # Do not print every environment variable. Azure ML environment variables
    # can include authentication material, storage configuration, and secrets.
    safe_environment_variables = (
        "AZUREML_MODEL_DIR",
        "VLLM_TARGET_DEVICE",
        "VLLM_CPU_KVCACHE_SPACE",
        "VLLM_CPU_OMP_THREADS_BIND",
        "TOKENIZERS_PARALLELISM",
        "LD_PRELOAD",
        "VLLM_SERVED_MODEL_NAME",
        "VLLM_MAX_MODEL_LEN",
        "VLLM_MAX_NUM_SEQS",
    )

    for variable_name in safe_environment_variables:
        logger.info(
            "%s=%s",
            variable_name,
            os.getenv(variable_name),
        )

    azureml_model_dir = os.getenv("AZUREML_MODEL_DIR")

    model_path = _resolve_model_path(azureml_model_dir)

    served_model_name = os.getenv(
        "VLLM_SERVED_MODEL_NAME",
        "Qwen/Qwen3.5-0.8B",
    )

    max_model_len = _get_positive_integer_environment_variable(
        variable_name="VLLM_MAX_MODEL_LEN",
        default_value=4096,
    )

    max_num_seqs = _get_positive_integer_environment_variable(
        variable_name="VLLM_MAX_NUM_SEQS",
        default_value=1,
    )

    logger.info("Resolved model path: %s", model_path)
    logger.info("Served model name: %s", served_model_name)
    logger.info("Maximum model length: %s", max_model_len)
    logger.info("Maximum active sequences: %s", max_num_seqs)

    vllm_engine = VllmEngine(
        model_path=model_path,
        served_model_name=served_model_name,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )

    # Give the vLLM subprocess an independent environment dictionary.
    child_environment = os.environ.copy()

    # This call should block until:
    #   - the vLLM process starts,
    #   - /health succeeds, and
    #   - /v1/models contains served_model_name.
    vllm_engine.load_model(env=child_environment)

    default_task_type = TaskType.CONVERSATIONAL

    logger.info(
        "score.py init() completed; vLLM is ready for inference"
    )


def run(
    raw_data: Union[
        str,
        bytes,
        bytearray,
        Dict[str, Any],
    ]
) -> Dict[str, Any]:
    """
    Process one Azure ML online-endpoint request.

    Parameters
    ----------
    raw_data:
        Request supplied by the Azure ML inference server. It can be:

        - a parsed dictionary,
        - a JSON string,
        - UTF-8 JSON bytes, or
        - UTF-8 JSON bytearray.

    Returns
    -------
    Dict[str, Any]
        JSON-serializable inference response.

    Raises
    ------
    RuntimeError
        When the engine is not initialized or inference fails.

    ValueError
        When the request is not valid JSON or doesn't match the
        expected inference request format.
    """

    logger.info("score.py run() started")

    if vllm_engine is None:
        raise RuntimeError(
            "vLLM engine is not initialized. "
            "score.py init() must complete successfully before run()."
        )

    try:
        data = _parse_raw_data(raw_data)

        inference_results, result_dictionary = _send_request(data)

        for inference_result in inference_results:
            inference_result.print_results()

        logger.info(
            "Inference completed with %d result(s)",
            len(inference_results),
        )

        logger.info("score.py run() completed")

        return result_dictionary

    except Exception:
        # logger.exception preserves the complete stack trace in the
        # Azure ML deployment logs.
        logger.exception("Inference request failed")

        # Raising the exception allows Azure ML to return a failed scoring
        # request instead of silently returning HTTP 200 with an error body.
        raise


def _parse_raw_data(
    raw_data: Union[
        str,
        bytes,
        bytearray,
        Dict[str, Any],
    ]
) -> Dict[str, Any]:
    """
    Convert Azure ML scoring input into a dictionary.

    Supported input types:

    - dict
    - str containing a JSON object
    - bytes containing a UTF-8 JSON object
    - bytearray containing a UTF-8 JSON object

    The resulting JSON document must be an object. JSON arrays, scalar
    strings, numbers, booleans, and null are rejected.
    """

    if isinstance(raw_data, dict):
        # Return a shallow copy so downstream processing doesn't modify
        # the object owned by the Azure ML scoring server.
        return dict(raw_data)

    if isinstance(raw_data, (bytes, bytearray)):
        try:
            raw_data = raw_data.decode("utf-8")
        except UnicodeDecodeError as exception:
            raise ValueError(
                "The scoring request body isn't valid UTF-8."
            ) from exception

    if not isinstance(raw_data, str):
        raise TypeError(
            "raw_data must be a dictionary, JSON string, JSON bytes, "
            "or JSON bytearray. "
            f"Received type: {type(raw_data).__name__}"
        )

    if not raw_data.strip():
        raise ValueError("The scoring request body is empty.")

    try:
        parsed_data = json.loads(raw_data)
    except json.JSONDecodeError as exception:
        raise ValueError(
            "The scoring request body isn't valid JSON. "
            f"JSON parser error: {exception.msg}; "
            f"line={exception.lineno}; "
            f"column={exception.colno}"
        ) from exception

    if not isinstance(parsed_data, dict):
        raise ValueError(
            "The inference request body must be a JSON object. "
            f"Received JSON type: {type(parsed_data).__name__}"
        )

    return parsed_data


def _send_request(
    data: Dict[str, Any],
) -> Tuple[
    List[InferenceResult],
    Dict[str, Any],
]:
    """
    Validate a request, create InferencePayload, and invoke vLLM.

    Chat-completion requests return:

        {
            "output": "assistant response"
        }

    Text-generation requests return:

        {
            "output": [
                "first generated response",
                "second generated response"
            ]
        }
    """

    if vllm_engine is None:
        raise RuntimeError("vLLM engine is not initialized.")

    # Work with an independent dictionary. InferencePayload validation may
    # normalize some request fields.
    request_data = dict(data)

    # If task_type isn't provided, use the default configured during init().
    request_data.setdefault(
        "task_type",
        default_task_type,
    )

    payload = InferencePayload.from_dict(
        input_data=request_data,
        model_config=None,
    )

    # The revised InferencePayload.update_params() accepts no argument and
    # creates an independent copy of the current parameter dictionary.
    payload.update_params()

    logger.info(
        "Processing inference request: task_type=%s, parameters=%s",
        payload.task_type,
        payload.params,
    )

    # For text generation, a single string becomes a list containing one
    # prompt. For chat completion, the list of role/content messages remains
    # one conversation.
    payload.convert_query_to_list()

    inference_results = vllm_engine.run(payload)

    if not inference_results:
        raise RuntimeError("vLLM returned no inference results.")

    if payload.task_type == TaskType.CONVERSATIONAL:
        first_result = inference_results[0]

        if first_result.error:
            raise RuntimeError(
                "vLLM chat-completion request failed: "
                f"{first_result.error}"
            )

        result_dictionary: Dict[str, Any] = {
            "output": first_result.response,
        }

    elif payload.task_type == TaskType.TEXT_GENERATION:
        inference_errors = [
            result.error
            for result in inference_results
            if result.error
        ]

        if inference_errors:
            raise RuntimeError(
                "One or more vLLM text-generation requests failed: "
                f"{inference_errors}"
            )

        result_dictionary = {
            "output": [
                result.response
                for result in inference_results
            ]
        }

    else:
        # This guards against inconsistent definitions between constants.py,
        # inference_payload.py, score.py, and webclient.py.
        if payload.task_type not in ALL_TASKS:
            raise ValueError(
                f"Unsupported inference task: {payload.task_type}"
            )

        result_dictionary = {
            "output": [
                result.response
                for result in inference_results
            ]
        }

    return inference_results, result_dictionary


def _resolve_model_path(
    azureml_model_dir: Optional[str],
) -> str:
    """
    Resolve AZUREML_MODEL_DIR to the Hugging Face model directory.

    The returned directory must contain config.json.

    Azure ML may mount the registered model directly:

        AZUREML_MODEL_DIR/
            config.json
            tokenizer.json
            model.safetensors

    Or it may mount a parent structure:

        AZUREML_MODEL_DIR/
            model-folder/
                config.json
                tokenizer.json
                model.safetensors
    """

    if not azureml_model_dir:
        raise RuntimeError(
            "AZUREML_MODEL_DIR isn't set. "
            "Verify that a model is attached to the Azure ML deployment."
        )

    root_directory = Path(azureml_model_dir).expanduser().resolve()

    if not root_directory.exists():
        raise FileNotFoundError(
            "AZUREML_MODEL_DIR doesn't exist: "
            f"{root_directory}"
        )

    if not root_directory.is_dir():
        raise NotADirectoryError(
            "AZUREML_MODEL_DIR isn't a directory: "
            f"{root_directory}"
        )

    # Preferred case: the Azure ML mount is the Hugging Face model root.
    direct_config_file = root_directory / "config.json"

    if direct_config_file.is_file():
        return str(root_directory)

    # Otherwise look for a single nested Hugging Face model directory.
    config_files = sorted(root_directory.rglob("config.json"))

    if not config_files:
        raise FileNotFoundError(
            "No config.json was found under AZUREML_MODEL_DIR: "
            f"{root_directory}"
        )

    if len(config_files) > 1:
        candidate_directories = sorted(
            {
                str(config_file.parent)
                for config_file in config_files
            }
        )

        raise RuntimeError(
            "Multiple Hugging Face model directories were found under "
            f"AZUREML_MODEL_DIR '{root_directory}'. "
            "The model location is ambiguous. "
            f"Candidate directories: {candidate_directories}"
        )

    model_directory = config_files[0].parent

    return str(model_directory)


def _get_positive_integer_environment_variable(
    variable_name: str,
    default_value: int,
) -> int:
    """
    Read an environment variable and validate that it's a positive integer.
    """

    raw_value = os.getenv(
        variable_name,
        str(default_value),
    )

    try:
        parsed_value = int(raw_value)
    except ValueError as exception:
        raise ValueError(
            f"{variable_name} must be an integer. "
            f"Received: {raw_value!r}"
        ) from exception

    if parsed_value <= 0:
        raise ValueError(
            f"{variable_name} must be greater than zero. "
            f"Received: {parsed_value}"
        )

    return parsed_value
