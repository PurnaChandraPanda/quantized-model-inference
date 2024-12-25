from typing import Dict, List, Union, Tuple
import time
from concurrent.futures import ThreadPoolExecutor
import requests
import json
import os

from transformers import AutoTokenizer
from engine import InferenceResult
from constants import TaskType

class LllamcppClient():
    def __init__(self, local_api_url: str):
        self._local_api_url = local_api_url
        self._tokenize_api_url = f"{self._local_api_url}/extras/tokenize"
        self._api_url = None
        self._task_type = None

    def generate(self, prompts: Union[str, List[str], List[Tuple[str, str]]], params: Dict, task_type: TaskType) -> List[InferenceResult]:
        """Generate responses for the given prompts with the given parameters."""
        
        # pop _batch_size from params if it exists, set it to 1 by default (for testing only)
        batch_size = params.pop("_batch_size", 1)

        return_full_text = params.pop("return_full_text", True)

        # Set task type
        self._task_type = task_type

        results = []
        start_time = time.time()
        # with ThreadPoolExecutor(max_workers=batch_size) as executor:
        #     results = list(
        #         executor.map(
        #             self._generate_on_prompt(prompts, params, return_full_text)
        #         ),
        #     )
        results.append(
                        self._generate_on_prompt(prompts, params, return_full_text)
                        )
        # print("after ThreadPoolExecutor: ", len(results))

        inference_time_ms = (time.time() - start_time) * 1000
        for i, result in enumerate(results):
            result.inference_time_ms = inference_time_ms
            result.prompt_num = i
        
        return results

    def _generate_on_prompt(self, prompt: Union[str, List[str], List[Tuple[str, str]]], params: Dict, return_full_text: bool) -> InferenceResult:
        """Generate a response for a single prompt with the given parameters."""

        headers = {
            "user-agent": "llama-cpp client",
            "generate_openai_response": "true"
        }

        
        # As per task type, modify the prompt.
        if self._task_type == TaskType.CONVERSATIONAL:
            payload = {
                "messages": prompt,
                **params,
                "stream": False
            }
            # payload = {
            #     "messages": prompt,
            #     **params
            # }

            self._api_url = f"{self._local_api_url}/v1/chat/completions"
        elif self._task_type == TaskType.TEXT_GENERATION:
            payload = {
                "prompt": prompt,
                **params,
                "stream": False
            }

            self._api_url = f"{self._local_api_url}/v1/completions"

        # print("before requests.post: ", payload, self._api_url)
        start_time = time.time()
        response = requests.post(self._api_url, headers=headers, json=payload)
        end_time = time.time()
        if response.status_code == 200:
            output = json.loads(response.content)

            # print("after requests.post: ", output)

            # assume openai response format
            generated_text = output["choices"][0]["message"]["content"]
            prompt_tokens = output["usage"]["prompt_tokens"]
            completion_tokens = output["usage"]["completion_tokens"]

            inference_time_ms = (end_time - start_time) * 1000
            response_tokens = self._get_tokens(generated_text)
            time_per_token_ms = inference_time_ms / len(response_tokens) if len(response_tokens) > 0 else 0

            res = InferenceResult(
                generated_text, inference_time_ms, time_per_token_ms, 0, response_tokens,
                n_prompt_tokens=prompt_tokens, n_completion_tokens=completion_tokens
                )
        else:
            res = InferenceResult(None, None, None, None, None, None, error=response.content)

        return res

    def _get_tokens(self, response_text: str):
        """Load tokenizer and get tokens from a prompt. 
        In this case, will leverage /extras/tokenize api of server."""
        # if not hasattr(self, "tokenizer"):
        #     if getattr(self.engine_config, "tokenizer", None) is not None:
        #         self.tokenizer = AutoTokenizer.from_pretrained(self.engine_config.tokenizer, trust_remote_code=True)
        #     else:
        #         self.tokenizer = AutoTokenizer.from_pretrained(self.engine_config.model_id, trust_remote_code=True)
        # tokens = self.tokenizer.encode(response)

        headers = {
            "user-agent": "llama-cpp client"
        }
        payload = {
                "input": response_text
            }
        # print("tokenize payload: ", payload)
        response = requests.post(self._tokenize_api_url, headers=headers, json=payload)
        if response.status_code == 200:
            output = json.loads(response.content)
            tokens = output["tokens"]
        elif response.status_code == 500:
            tokens = []

        # print("tokens: ", tokens)
        return tokens