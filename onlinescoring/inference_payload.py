"""This module provides the InferencePayload class that codifies the payload that is received in the scoring script."""

import json
import os
from dataclasses import dataclass
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer

from configs import SerializableDataClass
from constants import TaskType

DEFAULT_TOKENIZER_PATH = os.environ.get("model_path")
DEFAULT_MODEL_PATH = os.environ.get("model_path")

@dataclass
class InferencePayload(SerializableDataClass):
    """Json serializable dataclass that represents the input received from the server."""

    query: Union[str, List[str], List[Tuple[str, str]]]
    params: Dict[str, Any]
    task_type: str
    is_preview_format: bool

    @classmethod
    def from_dict(cls, input_data: Dict, model_config: Optional[Dict] = None):
        """Create an instance of InferencePayload from input data received from the server."""
        query, params, task_type, is_preview_format = get_request_data(input_data, model_config)
        return InferencePayload(query, params, task_type, is_preview_format)
    
    def update_params(self, new_params: Dict) -> None:
        """Update current parameters to the new parameters the InferencePayload should have."""
        self.params = new_params

    def convert_query_to_list(self) -> None:
        """Convert the query parameter into a list.

        The engine.run() expects a list of prompts. In the case of chat completion, a single string
        is produced and needs to be put inside of a list.
        """
        if not isinstance(self.query, list):
            self.query = [self.query]

def get_request_data(
    data,
    model_config: Optional[Dict] = None
) -> (Tuple)[Union[str, List[str]], Dict[str, Any], str, bool]:
    """Process and validate inference request.

    return type for chat-completion: str, dict, str, bool
    return type for text-generation: list, dict, str, bool
    """
    try:
        # print("get_request_data() :: data: ", data)
        is_preview_format = True
        inputs = data.get("input_data", None)
        task_type = data.get("task_type", TaskType.TEXT_GENERATION)
        # TODO: Update this check once all tasks are updated to use new input format
        if task_type != TaskType.TEXT_GENERATION:
            if not isinstance(inputs, dict):
                raise Exception("Invalid input data")

        if task_type == "chat-completion":
            task_type = TaskType.CONVERSATIONAL

        input_data = []  # type: Union[str, List[str]]
        params = {}  # type: Dict[str, Any]

        # Input format is being updated
        # Original text-gen input: {"input_data": {"input_string": ["<query>"], "parameters": {"k1":"v1", "k2":"v2"}}}
        # New text-gen input: {"input_data": ["<query>"], "params": {"k1":"v1", "k2":"v2"}}
        if task_type == TaskType.TEXT_GENERATION and "input_string" not in inputs:
            is_preview_format = False
            input_data = inputs
            params = data.get("params", {})
        else:
            input_data = inputs["input_string"]
            params = inputs.get("parameters", {})

        # if not isinstance(messages, list):
        if not isinstance(input_data, list):
            raise Exception("query is not a list")

        if not isinstance(params, dict):
            raise Exception("parameters is not a dict")

        if task_type == TaskType.CONVERSATIONAL:
            # print("chat-completion input_data: ", input_data)
            add_generation_prompt = params.pop("add_generation_prompt", True)

        return input_data, params, task_type, is_preview_format
    except Exception as e:
        task_type = data.get("task_type", TaskType.TEXT_GENERATION)
        if task_type == "chat-completion":
            correct_input_format = (
                '{"input_data": {"input_string": [{"role":"user", "content": "str1"}, '
                '{"role": "assistant", "content": "str2"} ....], "parameters": {"k1":"v1", "k2":"v2"}}}'
            )
        else:
            correct_input_format = (
                '{"input_data": ["str1", "str2", ...], '
                '"params": {"k1":"v1", "k2":"v2"}}'
            )

        raise Exception(
            json.dumps(
                {
                    "error": (
                        "Expected input format: \n" + correct_input_format
                    ),
                    "exception": str(e),
                },
            ),
        )

@dataclass
class InferenceResult:
    """Data class for storing inference results."""

    response: str
    inference_time_ms: float
    time_per_token_ms: float    
    prompt_num: int
    generated_tokens: List[Any] = None
    error: Optional[str] = None
    scores: Optional[List[Any]] = None
    n_prompt_tokens: Optional[int] = None
    n_completion_tokens: Optional[int] = None

    def _reset_gen_tokens(self):
        """Hide the gnerated tokens - save the space from printing."""
        self.generated_tokens = None

    def print_results(self):
        """Print the inference results of a single prompt.
        Also, mask the generated tokens."""
        if self.error:
            msg = f"## Inference Results ##\n ERROR: {self.error}"
        else:
            msg = f""" ## Prompt {self.prompt_num} Results ##\n Total Tokens Generated: {len(self.generated_tokens)}"""
        print(msg)

        # reset generated tokens
        self._reset_gen_tokens()

    