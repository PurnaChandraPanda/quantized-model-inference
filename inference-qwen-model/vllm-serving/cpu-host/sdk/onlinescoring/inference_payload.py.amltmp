"""
Request and response data structures for the Azure ML vLLM scoring service.

Request flow:

    Azure ML request
        -> score.py
        -> InferencePayload
        -> engine.py
        -> webclient.py
        -> local vLLM OpenAI-compatible server
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from configs import SerializableDataClass
from constants import TaskType


ChatMessage = Dict[str, Any]
ChatMessages = List[ChatMessage]
TextPrompts = List[str]

QueryType = Union[
    str,
    TextPrompts,
    ChatMessages,
    List[Tuple[str, str]],
]


@dataclass
class InferencePayload(SerializableDataClass):
    """
    Normalized inference request passed from score.py to engine.py.

    Attributes:
        query:
            For chat completion, this is an OpenAI-compatible messages list.

            Example:

                [
                    {
                        "role": "user",
                        "content": "Explain large language models."
                    }
                ]

            For text generation, this is a string prompt or list of
            string prompts.

        params:
            Generation parameters such as max_tokens, temperature,
            top_p, top_k, and presence_penalty.

        task_type:
            Normalized task type from constants.TaskType.

        is_preview_format:
            True when the request uses the older Azure ML input format.
    """

    query: QueryType
    params: Dict[str, Any]
    task_type: str
    is_preview_format: bool

    @classmethod
    def from_dict(
        cls,
        input_data: Dict[str, Any],
        model_config: Optional[Dict[str, Any]] = None,
    ) -> "InferencePayload":
        """
        Create an InferencePayload from an Azure ML scoring request.
        """

        query, params, task_type, is_preview_format = get_request_data(
            data=input_data,
            model_config=model_config,
        )

        return cls(
            query=query,
            params=params,
            task_type=task_type,
            is_preview_format=is_preview_format,
        )

    def update_params(
        self,
        new_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Replace or defensively copy generation parameters.

        Calling update_params() without an argument creates a copy of
        the existing params dictionary. This prevents downstream code
        from changing the original request dictionary.
        """

        if new_params is None:
            self.params = dict(self.params or {})
            return

        if not isinstance(new_params, dict):
            raise TypeError(
                "new_params must be a dictionary. "
                f"Received: {type(new_params).__name__}"
            )

        self.params = dict(new_params)

    def convert_query_to_list(self) -> None:
        """
        Normalize the query before passing it to VllmEngine.run().

        Chat completion:
            The messages list represents one conversation and remains
            a single list of message dictionaries.

        Text generation:
            A single string is converted to a one-item list.
        """

        if self.task_type == TaskType.CONVERSATIONAL:
            self.query = _normalize_chat_messages(self.query)
            return

        if self.task_type == TaskType.TEXT_GENERATION:
            self.query = _normalize_text_prompts(self.query)
            return

        raise ValueError(
            "Unsupported task type while normalizing query: "
            f"{self.task_type!r}"
        )


def get_request_data(
    data: Dict[str, Any],
    model_config: Optional[Dict[str, Any]] = None,
) -> Tuple[QueryType, Dict[str, Any], str, bool]:
    """
    Validate and normalize an Azure ML inference request.

    Chat-completion format:

        {
            "task_type": "chat-completion",
            "input_data": {
                "input_string": [
                    {
                        "role": "user",
                        "content": "Explain large language models."
                    }
                ],
                "parameters": {
                    "max_tokens": 200,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": 20
                }
            }
        }

    Current text-generation format:

        {
            "task_type": "text-generation",
            "input_data": [
                "Explain large language models."
            ],
            "params": {
                "max_tokens": 200
            }
        }

    Legacy text-generation format:

        {
            "task_type": "text-generation",
            "input_data": {
                "input_string": [
                    "Explain large language models."
                ],
                "parameters": {
                    "max_tokens": 200
                }
            }
        }
    """

    # Retained for compatibility with the existing interface.
    del model_config

    if not isinstance(data, dict):
        raise TypeError(
            "Inference request must be a JSON object. "
            f"Received: {type(data).__name__}"
        )

    raw_task_type = data.get(
        "task_type",
        TaskType.TEXT_GENERATION,
    )

    task_type = _normalize_task_type(raw_task_type)

    try:
        if task_type == TaskType.CONVERSATIONAL:
            return _parse_chat_request(
                data=data,
                task_type=task_type,
            )

        if task_type == TaskType.TEXT_GENERATION:
            return _parse_text_generation_request(
                data=data,
                task_type=task_type,
            )

        raise ValueError(f"Unsupported task type: {raw_task_type!r}")

    except Exception as exception:
        expected_format = _get_expected_input_format(task_type)

        error_details = {
            "error": "Invalid inference request.",
            "task_type": raw_task_type,
            "expected_input_format": expected_format,
            "exception": str(exception),
        }

        raise ValueError(
            json.dumps(
                error_details,
                indent=2,
            )
        ) from exception


def _parse_chat_request(
    data: Dict[str, Any],
    task_type: str,
) -> Tuple[ChatMessages, Dict[str, Any], str, bool]:
    """
    Parse a chat-completion request.

    Supported input:

        {
            "task_type": "chat-completion",
            "input_data": {
                "input_string": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ],
                "parameters": {
                    "max_tokens": 100
                }
            }
        }

    The key "messages" is also accepted instead of "input_string".
    """

    inputs = data.get("input_data")

    if not isinstance(inputs, dict):
        raise TypeError(
            "For chat completion, input_data must be a dictionary "
            "containing input_string or messages."
        )

    messages = inputs.get("input_string")

    if messages is None:
        messages = inputs.get("messages")

    if messages is None:
        raise ValueError(
            "Chat-completion input_data must contain either "
            "'input_string' or 'messages'."
        )

    parameters = inputs.get("parameters")

    if parameters is None:
        parameters = data.get("params", {})

    if not isinstance(parameters, dict):
        raise TypeError(
            "Chat-completion parameters must be a dictionary. "
            f"Received: {type(parameters).__name__}"
        )

    normalized_messages = _normalize_chat_messages(messages)
    normalized_parameters = dict(parameters)

    # The vLLM OpenAI chat-completions endpoint applies the model's chat
    # template itself. This parameter should therefore not be forwarded
    # as a generation parameter.
    normalized_parameters.pop("add_generation_prompt", None)

    return (
        normalized_messages,
        normalized_parameters,
        task_type,
        True,
    )


def _parse_text_generation_request(
    data: Dict[str, Any],
    task_type: str,
) -> Tuple[TextPrompts, Dict[str, Any], str, bool]:
    """
    Parse current or legacy text-generation request formats.
    """

    inputs = data.get("input_data")

    if inputs is None:
        raise ValueError(
            "The inference request must contain input_data."
        )

    is_preview_format: bool

    if isinstance(inputs, dict):
        if "input_string" not in inputs:
            raise ValueError(
                "Legacy text-generation input_data must contain "
                "'input_string'."
            )

        prompts = inputs["input_string"]
        parameters = inputs.get("parameters", {})
        is_preview_format = True

    elif isinstance(inputs, (str, list)):
        prompts = inputs
        parameters = data.get("params", {})
        is_preview_format = False

    else:
        raise TypeError(
            "For text generation, input_data must be a string, "
            "a list of strings, or a dictionary containing input_string."
        )

    if not isinstance(parameters, dict):
        raise TypeError(
            "Text-generation parameters must be a dictionary. "
            f"Received: {type(parameters).__name__}"
        )

    normalized_prompts = _normalize_text_prompts(prompts)

    return (
        normalized_prompts,
        dict(parameters),
        task_type,
        is_preview_format,
    )


def _normalize_task_type(task_type: Any) -> str:
    """
    Normalize supported task aliases into TaskType constants.
    """

    if task_type == TaskType.CONVERSATIONAL:
        return TaskType.CONVERSATIONAL

    if task_type == TaskType.TEXT_GENERATION:
        return TaskType.TEXT_GENERATION

    if not isinstance(task_type, str):
        raise TypeError(
            "task_type must be a string. "
            f"Received: {type(task_type).__name__}"
        )

    normalized_value = (
        task_type
        .strip()
        .lower()
        .replace("_", "-")
    )

    conversational_aliases = {
        "chat-completion",
        "chat-completions",
        "chat",
        "conversational",
        "conversation",
    }

    text_generation_aliases = {
        "text-generation",
        "text-completion",
        "text-completions",
        "completion",
        "completions",
        "generate",
    }

    if normalized_value in conversational_aliases:
        return TaskType.CONVERSATIONAL

    if normalized_value in text_generation_aliases:
        return TaskType.TEXT_GENERATION

    raise ValueError(
        f"Unsupported task_type value: {task_type!r}"
    )


def _normalize_chat_messages(query: Any) -> ChatMessages:
    """
    Validate and normalize an OpenAI-compatible chat messages list.

    Supported dictionary format:

        [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Hello"
            }
        ]

    The older tuple-based representation is also supported:

        [
            ("system", "You are a helpful assistant."),
            ("user", "Hello")
        ]

    Additional fields such as name, tool_call_id, tool_calls, refusal,
    and reasoning metadata are preserved.
    """

    if not isinstance(query, list):
        raise TypeError(
            "Chat input must be a list of message dictionaries. "
            f"Received: {type(query).__name__}"
        )

    if not query:
        raise ValueError(
            "Chat input must contain at least one message."
        )

    # Support the older tuple-based representation:
    #
    # [
    #     ("system", "You are a helpful assistant."),
    #     ("user", "Hello")
    # ]
    #
    # It is converted to OpenAI-compatible message dictionaries.
    if all(
        isinstance(item, tuple) and len(item) == 2
        for item in query
    ):
        tuple_messages: ChatMessages = []

        for index, item in enumerate(query):
            role, content = item

            if not isinstance(role, str) or not role.strip():
                raise ValueError(
                    f"Tuple message at index {index} must contain "
                    "a non-empty string role."
                )

            if not isinstance(content, str):
                raise TypeError(
                    f"Tuple message content at index {index} must "
                    "be a string. "
                    f"Received: {type(content).__name__}"
                )

            tuple_messages.append(
                {
                    "role": role.strip().lower(),
                    "content": content,
                }
            )

        query = tuple_messages

    normalized_messages: ChatMessages = []

    supported_roles = {
        "system",
        "developer",
        "user",
        "assistant",
        "tool",
    }

    for index, message in enumerate(query):
        if not isinstance(message, dict):
            raise TypeError(
                f"Chat message at index {index} must be a dictionary. "
                f"Received: {type(message).__name__}"
            )

        role = message.get("role")

        if not isinstance(role, str):
            raise TypeError(
                f"Chat message role at index {index} must be a string. "
                f"Received: {type(role).__name__}"
            )

        normalized_role = role.strip().lower()

        if not normalized_role:
            raise ValueError(
                f"Chat message at index {index} contains an empty role."
            )

        if normalized_role not in supported_roles:
            raise ValueError(
                f"Unsupported chat role at index {index}: "
                f"{role!r}. Supported roles are: "
                f"{sorted(supported_roles)}"
            )

        content_is_present = "content" in message
        content = message.get("content")

        # Assistant messages with tool_calls can legally have null content.
        # Tool-related messages may also have extra fields, which are
        # preserved below.
        has_tool_calls = bool(message.get("tool_calls"))

        if not content_is_present and not has_tool_calls:
            raise ValueError(
                f"Chat message at index {index} must contain "
                "'content' or 'tool_calls'."
            )

        if content is not None and not isinstance(content, str):
            raise TypeError(
                f"Chat message content at index {index} must be "
                "a string or null. "
                f"Received: {type(content).__name__}"
            )

        if normalized_role == "tool":
            tool_call_id = message.get("tool_call_id")

            if not isinstance(tool_call_id, str) or not tool_call_id.strip():
                raise ValueError(
                    f"Tool message at index {index} must contain "
                    "a non-empty string tool_call_id."
                )

        if "name" in message:
            name = message["name"]

            if name is not None and not isinstance(name, str):
                raise TypeError(
                    f"Chat message name at index {index} must be "
                    "a string or null. "
                    f"Received: {type(name).__name__}"
                )

        if "tool_calls" in message:
            tool_calls = message["tool_calls"]

            if tool_calls is not None and not isinstance(tool_calls, list):
                raise TypeError(
                    f"Chat message tool_calls at index {index} must "
                    "be a list or null. "
                    f"Received: {type(tool_calls).__name__}"
                )

        # Copy the message so that normalization never mutates the
        # dictionary supplied by the caller.
        normalized_message = dict(message)
        normalized_message["role"] = normalized_role

        # Preserve content=None for assistant tool-call messages.
        if content_is_present:
            normalized_message["content"] = content

        normalized_messages.append(normalized_message)

    if not normalized_messages:
        raise ValueError(
            "Chat input did not contain any valid messages."
        )

    return normalized_messages


def _normalize_text_prompts(query: Any) -> TextPrompts:
    """
    Normalize text-generation input into a non-empty list of strings.
    """

    if isinstance(query, str):
        if not query:
            raise ValueError(
                "Text-generation prompt cannot be empty."
            )

        return [query]

    if not isinstance(query, list):
        raise TypeError(
            "Text-generation input must be a string or a list "
            "of strings. "
            f"Received: {type(query).__name__}"
        )

    if not query:
        raise ValueError(
            "Text-generation input must contain at least one prompt."
        )

    normalized_prompts: TextPrompts = []

    for index, prompt in enumerate(query):
        if not isinstance(prompt, str):
            raise TypeError(
                f"Text prompt at index {index} must be a string. "
                f"Received: {type(prompt).__name__}"
            )

        if not prompt:
            raise ValueError(
                f"Text prompt at index {index} cannot be empty."
            )

        normalized_prompts.append(prompt)

    return normalized_prompts


def _get_expected_input_format(task_type: str) -> Dict[str, Any]:
    """
    Return a request example for validation error messages.
    """

    if task_type == TaskType.CONVERSATIONAL:
        return {
            "task_type": "chat-completion",
            "input_data": {
                "input_string": [
                    {
                        "role": "user",
                        "content": (
                            "Provide a short introduction to "
                            "large language models."
                        ),
                    }
                ],
                "parameters": {
                    "max_tokens": 200,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "presence_penalty": 2.0,
                    "top_k": 20,
                },
            },
        }

    return {
        "task_type": "text-generation",
        "input_data": [
            "Provide a short introduction to large language models."
        ],
        "params": {
            "max_tokens": 200,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 20,
        },
    }


@dataclass
class InferenceResult:
    """
    Result returned by VllmClient for one inference request.
    """

    response: Optional[str]
    inference_time_ms: Optional[float]
    time_per_token_ms: Optional[float]
    prompt_num: int = 0
    generated_tokens: Optional[List[Any]] = None
    error: Optional[str] = None
    scores: Optional[List[Any]] = None
    n_prompt_tokens: Optional[int] = None
    n_completion_tokens: Optional[int] = None

    def _reset_gen_tokens(self) -> None:
        """
        Remove generated token IDs after logging to reduce output size.
        """

        self.generated_tokens = None

    def print_results(self) -> None:
        """
        Print a compact inference summary without generated text.
        """

        generated_token_count = (
            len(self.generated_tokens)
            if self.generated_tokens is not None
            else self.n_completion_tokens or 0
        )

        if self.error:
            print(
                "Inference Results Error: "
                f"prompt_num={self.prompt_num}, "
                f"error={self.error}"
            )

            self._reset_gen_tokens()
            return

        inference_time_ms = self.inference_time_ms or 0.0
        time_per_token_ms = self.time_per_token_ms or 0.0
        prompt_tokens = self.n_prompt_tokens or 0
        completion_tokens = self.n_completion_tokens or 0

        print(
            "Inference Results: "
            f"prompt_num={self.prompt_num}, "
            f"prompt_tokens={prompt_tokens}, "
            f"completion_tokens={completion_tokens}, "
            f"generated_token_ids={generated_token_count}, "
            f"inference_time_ms={inference_time_ms:.2f}, "
            f"time_per_token_ms={time_per_token_ms:.2f}"
        )

        self._reset_gen_tokens()