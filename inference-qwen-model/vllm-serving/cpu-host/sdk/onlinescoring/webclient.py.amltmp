"""
HTTP client for a locally hosted vLLM OpenAI-compatible server.

Expected flow:

    Azure ML scoring server
        -> score.py
        -> VllmEngine
        -> VllmClient
        -> http://127.0.0.1:8000/v1/chat/completions

The base URL passed to VllmClient must be the server root:

    http://127.0.0.1:8000

Do not pass:

    http://127.0.0.1:8000/v1

because this client adds the /v1 paths itself.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

from constants import TaskType
from inference_payload import InferenceResult


ChatMessage = Dict[str, Any]
ChatMessages = List[ChatMessage]

TextPrompt = str
TextPrompts = List[str]

PromptType = Union[
    TextPrompt,
    TextPrompts,
    ChatMessages,
    List[Tuple[str, str]],
]


class VllmClient:
    """
    HTTP client for the vLLM OpenAI-compatible API server.

    Parameters
    ----------
    local_api_url:
        Root URL of the local vLLM server.

        Example:
            http://127.0.0.1:8000

    model_name:
        Exact, case-sensitive model ID registered by vLLM through
        --served-model-name.

        Example:
            Qwen/Qwen3.5-0.8B

    request_timeout_seconds:
        Maximum duration allowed for one inference HTTP request.
    """

    def __init__(
        self,
        local_api_url: str,
        model_name: str,
        request_timeout_seconds: int = 110,
    ) -> None:
        if not local_api_url:
            raise ValueError("local_api_url cannot be empty.")

        if not model_name:
            raise ValueError("model_name cannot be empty.")

        if request_timeout_seconds <= 0:
            raise ValueError(
                "request_timeout_seconds must be greater than zero."
            )

        self.local_api_url = local_api_url.rstrip("/")
        self.model_name = model_name
        self.request_timeout_seconds = request_timeout_seconds

        self.health_api_url = f"{self.local_api_url}/health"
        self.models_api_url = f"{self.local_api_url}/v1/models"
        self.chat_api_url = (
            f"{self.local_api_url}/v1/chat/completions"
        )
        self.completion_api_url = (
            f"{self.local_api_url}/v1/completions"
        )
        self.tokenize_api_url = f"{self.local_api_url}/tokenize"

        # Session reuses the local TCP connection across requests.
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "azureml-vllm-client",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    def generate(
        self,
        prompts: PromptType,
        params: Dict[str, Any],
        task_type: str,
    ) -> List[InferenceResult]:
        """
        Generate responses using the local vLLM server.

        Chat completion
        ---------------
        `prompts` must be one OpenAI-style message list:

            [
                {"role": "user", "content": "Hello"}
            ]

        Text generation
        ---------------
        `prompts` can be a string or a list of strings.

        Because the current vLLM server is configured with
        --max-num-seqs 1, multiple text prompts are processed
        sequentially rather than concurrently.
        """

        if not isinstance(params, dict):
            raise TypeError("params must be a dictionary.")

        # Do not mutate InferencePayload.params.
        request_params = dict(params)

        batch_size = self._pop_positive_integer(
            request_params,
            key="batch_size",
            default_value=1,
        )

        # Retained for compatibility with the existing input contract.
        # The OpenAI-compatible response contains only generated output.
        return_full_text = bool(
            request_params.pop("return_full_text", False)
        )

        if batch_size != 1:
            print(
                "Warning: batch_size is ignored because this deployment "
                "uses --max-num-seqs 1. Requests will be processed "
                "sequentially."
            )

        if task_type == TaskType.CONVERSATIONAL:
            messages = self._normalize_chat_messages(prompts)

            result = self._generate_on_prompt(
                prompt=messages,
                params=request_params,
                task_type=task_type,
                return_full_text=return_full_text,
                prompt_number=0,
            )

            return [result]

        if task_type == TaskType.TEXT_GENERATION:
            text_prompts = self._normalize_text_prompts(prompts)

            results: List[InferenceResult] = []

            for prompt_number, prompt in enumerate(text_prompts):
                result = self._generate_on_prompt(
                    prompt=prompt,
                    params=request_params,
                    task_type=task_type,
                    return_full_text=return_full_text,
                    prompt_number=prompt_number,
                )
                results.append(result)

            return results

        raise ValueError(f"Unsupported vLLM task type: {task_type!r}")

    def _generate_on_prompt(
        self,
        prompt: Union[str, ChatMessages],
        params: Dict[str, Any],
        task_type: str,
        return_full_text: bool,
        prompt_number: int,
    ) -> InferenceResult:
        """
        Send one chat-completion or text-completion request to vLLM.
        """

        request_params = dict(params)

        if task_type == TaskType.CONVERSATIONAL:
            api_url = self.chat_api_url

            payload: Dict[str, Any] = {
                "model": self.model_name,
                "messages": prompt,
                "stream": False,
                **request_params,
            }

        elif task_type == TaskType.TEXT_GENERATION:
            api_url = self.completion_api_url

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                **request_params,
            }

        else:
            raise ValueError(
                f"Unsupported vLLM task type: {task_type!r}"
            )

        print(
            "Sending request to local vLLM server: "
            f"url={api_url}, "
            f"model={self.model_name}, "
            f"task_type={task_type}, "
            f"prompt_number={prompt_number}"
        )

        start_time = time.monotonic()

        try:
            response = self.session.post(
                api_url,
                json=payload,
                timeout=self.request_timeout_seconds,
            )

        except requests.Timeout as exception:
            raise RuntimeError(
                "Request to local vLLM server timed out after "
                f"{self.request_timeout_seconds} seconds. "
                f"URL={api_url}, model={self.model_name}"
            ) from exception

        except requests.RequestException as exception:
            raise RuntimeError(
                "Failed to call local vLLM server. "
                f"URL={api_url}, model={self.model_name}, "
                f"error={exception}"
            ) from exception

        inference_time_ms = (
            time.monotonic() - start_time
        ) * 1000.0

        if not response.ok:
            error_body = self._safe_response_body(response)

            return InferenceResult(
                response=None,
                inference_time_ms=inference_time_ms,
                time_per_token_ms=None,
                prompt_num=prompt_number,
                generated_tokens=None,
                error=(
                    "vLLM request failed. "
                    f"HTTP status={response.status_code}, "
                    f"URL={api_url}, "
                    f"model={self.model_name}, "
                    f"response={error_body}"
                ),
                n_prompt_tokens=None,
                n_completion_tokens=None,
            )

        try:
            response_payload = response.json()

        except ValueError as exception:
            return InferenceResult(
                response=None,
                inference_time_ms=inference_time_ms,
                time_per_token_ms=None,
                prompt_num=prompt_number,
                generated_tokens=None,
                error=(
                    "vLLM returned HTTP success but the response was "
                    f"not valid JSON. Response={response.text}"
                ),
                n_prompt_tokens=None,
                n_completion_tokens=None,
            )

        try:
            generated_text = self._extract_generated_text(
                response_payload=response_payload,
                task_type=task_type,
            )

            usage = response_payload.get("usage") or {}

            prompt_tokens = self._safe_integer(
                usage.get("prompt_tokens")
            )

            completion_tokens = self._safe_integer(
                usage.get("completion_tokens")
            )

            generated_tokens = self._get_tokens(generated_text)

            # Prefer vLLM usage count. If absent, use the token IDs
            # returned from /tokenize.
            token_count = completion_tokens

            if not token_count and generated_tokens:
                token_count = len(generated_tokens)

            time_per_token_ms: Optional[float]

            if token_count and token_count > 0:
                time_per_token_ms = (
                    inference_time_ms / token_count
                )
            else:
                time_per_token_ms = None

            if (
                return_full_text
                and task_type == TaskType.TEXT_GENERATION
                and isinstance(prompt, str)
            ):
                generated_text = prompt + generated_text

            return InferenceResult(
                response=generated_text,
                inference_time_ms=inference_time_ms,
                time_per_token_ms=time_per_token_ms,
                prompt_num=prompt_number,
                generated_tokens=generated_tokens,
                error=None,
                n_prompt_tokens=prompt_tokens,
                n_completion_tokens=completion_tokens,
            )

        except Exception as exception:
            return InferenceResult(
                response=None,
                inference_time_ms=inference_time_ms,
                time_per_token_ms=None,
                prompt_num=prompt_number,
                generated_tokens=None,
                error=(
                    "Failed to process vLLM response. "
                    f"Exception={exception}, "
                    f"response={response_payload}"
                ),
                n_prompt_tokens=None,
                n_completion_tokens=None,
            )

    @staticmethod
    def _extract_generated_text(
        response_payload: Dict[str, Any],
        task_type: str,
    ) -> str:
        """
        Extract generated output from an OpenAI-compatible response.

        Chat completion:
            choices[0].message.content

        Text completion:
            choices[0].text
        """

        choices = response_payload.get("choices")

        if not isinstance(choices, list) or not choices:
            raise ValueError(
                "vLLM response doesn't contain a non-empty "
                "'choices' array."
            )

        first_choice = choices[0]

        if not isinstance(first_choice, dict):
            raise TypeError(
                "The first item in vLLM 'choices' isn't an object."
            )

        if task_type == TaskType.CONVERSATIONAL:
            message = first_choice.get("message")

            if not isinstance(message, dict):
                raise ValueError(
                    "Chat-completion response doesn't contain "
                    "'choices[0].message'."
                )

            content = message.get("content")

            if content is None:
                # Some models can return tool calls without textual
                # content. Preserve this as an empty string instead of
                # converting None to the string "None".
                return ""

            if not isinstance(content, str):
                raise TypeError(
                    "Chat-completion message content isn't a string."
                )

            return content

        if task_type == TaskType.TEXT_GENERATION:
            text = first_choice.get("text")

            if text is None:
                return ""

            if not isinstance(text, str):
                raise TypeError(
                    "Text-completion response text isn't a string."
                )

            return text

        raise ValueError(f"Unsupported task type: {task_type!r}")

    def _get_tokens(self, response_text: str) -> List[Any]:
        """
        Tokenize generated text by using vLLM's /tokenize endpoint.

        Tokenization is treated as optional because the generation
        response already supplies prompt_tokens and completion_tokens
        in the OpenAI-compatible usage object.
        """

        if not response_text:
            return []

        payload = {
            "model": self.model_name,
            "prompt": response_text,
        }

        try:
            response = self.session.post(
                self.tokenize_api_url,
                json=payload,
                timeout=min(self.request_timeout_seconds, 15),
            )

        except requests.RequestException as exception:
            print(
                "Tokenization request failed. Continuing with token "
                f"counts from the generation response. Error={exception}"
            )
            return []

        if not response.ok:
            print(
                "vLLM tokenization endpoint returned an error. "
                f"HTTP status={response.status_code}, "
                f"response={self._safe_response_body(response)}"
            )
            return []

        try:
            response_payload = response.json()

        except ValueError:
            print(
                "vLLM tokenization endpoint returned invalid JSON: "
                f"{response.text}"
            )
            return []

        # Current/older vLLM versions can use slightly different keys.
        tokens = response_payload.get("tokens")

        if not isinstance(tokens, list):
            tokens = response_payload.get("input_ids")

        if not isinstance(tokens, list):
            return []

        return tokens

    def is_healthy(self) -> bool:
        """
        Return True when the local vLLM health endpoint succeeds.
        """

        try:
            response = self.session.get(
                self.health_api_url,
                timeout=5,
            )
            return response.ok

        except requests.RequestException:
            return False

    def get_available_models(self) -> List[str]:
        """
        Return model IDs registered by the local vLLM server.
        """

        try:
            response = self.session.get(
                self.models_api_url,
                timeout=10,
            )
            response.raise_for_status()
            response_payload = response.json()

        except requests.RequestException as exception:
            raise RuntimeError(
                "Failed to retrieve models from local vLLM server. "
                f"URL={self.models_api_url}, error={exception}"
            ) from exception

        except ValueError as exception:
            raise RuntimeError(
                "The local vLLM /v1/models response isn't valid JSON."
            ) from exception

        models = response_payload.get("data")

        if not isinstance(models, list):
            return []

        model_ids: List[str] = []

        for model in models:
            if not isinstance(model, dict):
                continue

            model_id = model.get("id")

            if isinstance(model_id, str):
                model_ids.append(model_id)

        return model_ids

    def validate_model_registration(self) -> None:
        """
        Verify that the configured case-sensitive model name is registered.
        """

        available_models = self.get_available_models()

        if self.model_name not in available_models:
            raise RuntimeError(
                "Expected model isn't registered by vLLM. "
                f"Expected={self.model_name!r}, "
                f"available={available_models!r}"
            )

    def close(self) -> None:
        """
        Close reusable HTTP connections.
        """

        self.session.close()

    @staticmethod
    def _normalize_chat_messages(
        prompts: PromptType,
    ) -> ChatMessages:
        """
        Validate and normalize one OpenAI-style conversation.
        """

        if not isinstance(prompts, list) or not prompts:
            raise TypeError(
                "Chat-completion prompts must be a non-empty list "
                "of message dictionaries."
            )

        # Compatibility with the older tuple-based representation:
        #
        #     [
        #         ("user", "Hello"),
        #         ("assistant", "Hi")
        #     ]
        if all(
            isinstance(item, tuple) and len(item) == 2
            for item in prompts
        ):
            messages: ChatMessages = []

            for role, content in prompts:
                messages.append(
                    {
                        "role": str(role),
                        "content": str(content),
                    }
                )

            return messages

        messages = []

        for index, message in enumerate(prompts):
            if not isinstance(message, dict):
                raise TypeError(
                    f"Chat message at index {index} must be a "
                    "dictionary."
                )

            role = message.get("role")
            content = message.get("content")

            if not isinstance(role, str) or not role:
                raise ValueError(
                    f"Chat message at index {index} requires a "
                    "non-empty 'role'."
                )

            if content is not None and not isinstance(content, str):
                raise TypeError(
                    f"Chat message content at index {index} must "
                    "be a string or null."
                )

            # Preserve optional message properties, such as name or
            # tool_call_id, if supplied by the request.
            messages.append(dict(message))

        return messages

    @staticmethod
    def _normalize_text_prompts(
        prompts: PromptType,
    ) -> TextPrompts:
        """
        Normalize text-generation input into a non-empty list of strings.
        """

        if isinstance(prompts, str):
            if not prompts:
                raise ValueError(
                    "Text-generation prompt cannot be empty."
                )
            return [prompts]

        if not isinstance(prompts, list) or not prompts:
            raise TypeError(
                "Text-generation prompts must be a string or "
                "a non-empty list of strings."
            )

        text_prompts: TextPrompts = []

        for index, prompt in enumerate(prompts):
            if not isinstance(prompt, str):
                raise TypeError(
                    f"Text prompt at index {index} must be a string."
                )

            text_prompts.append(prompt)

        return text_prompts

    @staticmethod
    def _safe_response_body(
        response: requests.Response,
    ) -> Any:
        """
        Return response JSON when possible; otherwise return text.
        """

        try:
            return response.json()
        except ValueError:
            return response.text

    @staticmethod
    def _safe_integer(value: Any) -> Optional[int]:
        """
        Convert an API usage value into an integer when possible.
        """

        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _pop_positive_integer(
        values: Dict[str, Any],
        key: str,
        default_value: int,
    ) -> int:
        """
        Pop and validate a positive integer parameter.
        """

        raw_value = values.pop(key, default_value)

        try:
            parsed_value = int(raw_value)
        except (TypeError, ValueError) as exception:
            raise ValueError(
                f"{key} must be an integer. "
                f"Received: {raw_value!r}"
            ) from exception

        if parsed_value <= 0:
            raise ValueError(
                f"{key} must be greater than zero. "
                f"Received: {parsed_value}"
            )

        return parsed_value