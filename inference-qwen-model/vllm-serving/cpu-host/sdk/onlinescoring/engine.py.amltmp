"""
engine.py

Starts and manages a local vLLM OpenAI-compatible API server as a child
process of the Azure ML scoring worker.

Flow:

Azure ML inference server
    -> score.py
        -> VllmEngine
            -> vllm serve subprocess
            -> VllmClient
                -> http://127.0.0.1:8000/v1/chat/completions
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

from inference_payload import InferencePayload, InferenceResult
from webclient import VllmClient


class VllmEngine:
    """
    Manage a local vLLM OpenAI-compatible inference server.

    Parameters
    ----------
    model_path:
        Local Hugging Face model directory containing config.json,
        tokenizer files, and model weights.

    served_model_name:
        Case-sensitive model ID exposed by vLLM through /v1/models.

    max_model_len:
        Maximum combined prompt and generated-token sequence length.

    max_num_seqs:
        Maximum number of active sequences handled by vLLM.

    host:
        Address on which vLLM listens. Use 0.0.0.0 for the server.

    port:
        Local vLLM API port. This must be different from the Azure ML
        inference-server port, which is normally 31311.

    startup_timeout_seconds:
        Maximum time allowed for vLLM model loading and warm-up.

    request_timeout_seconds:
        Maximum duration for one inference request sent to vLLM.
    """

    def __init__(
        self,
        model_path: str,
        served_model_name: str = "Qwen/Qwen3.5-0.8B",
        max_model_len: int = 4096,
        max_num_seqs: int = 1,
        host: str = "0.0.0.0",
        port: int = 8000,
        startup_timeout_seconds: int = 15 * 60,
        request_timeout_seconds: int = 110,
    ) -> None:
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.served_model_name = served_model_name
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs

        # vLLM binds to this address.
        self.server_host = host
        self.server_port = port

        # score.py should call the local child server through loopback,
        # not through 0.0.0.0.
        self.client_host = "127.0.0.1"

        self.startup_timeout_seconds = startup_timeout_seconds
        self.request_timeout_seconds = request_timeout_seconds

        self.process: Optional[subprocess.Popen] = None
        self._is_cuda_visible = False

        # webclient.py expects the server root. It adds /v1/... itself.
        self.client = VllmClient(
            local_api_url=(
                f"http://{self.client_host}:{self.server_port}"
            ),
            model_name=self.served_model_name,
            request_timeout_seconds=self.request_timeout_seconds,
        )

        # Ensure the child process is stopped if the scoring worker exits.
        atexit.register(self.stop)

    def load_model(
        self,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Validate the local model directory, start vLLM, and wait until ready.

        The method returns only after:

        1. The vLLM child process is still running.
        2. GET /health succeeds.
        3. GET /v1/models contains served_model_name.
        """
        child_environment = os.environ.copy()

        if env:
            child_environment.update(env)

        self._validate_configuration()
        self._validate_model_directory()
        self._start_server(child_environment)

    def _validate_configuration(self) -> None:
        """Validate constructor settings before starting vLLM."""
        if not self.served_model_name:
            raise ValueError("served_model_name cannot be empty.")

        if self.max_model_len <= 0:
            raise ValueError(
                "max_model_len must be greater than zero. "
                f"Received: {self.max_model_len}"
            )

        if self.max_num_seqs <= 0:
            raise ValueError(
                "max_num_seqs must be greater than zero. "
                f"Received: {self.max_num_seqs}"
            )

        if self.server_port <= 0 or self.server_port > 65535:
            raise ValueError(
                "port must be between 1 and 65535. "
                f"Received: {self.server_port}"
            )

        if self.startup_timeout_seconds <= 0:
            raise ValueError(
                "startup_timeout_seconds must be greater than zero."
            )

        if self.request_timeout_seconds <= 0:
            raise ValueError(
                "request_timeout_seconds must be greater than zero."
            )

    def _validate_model_directory(self) -> None:
        """
        Validate that model_path is a Hugging Face model directory.
        """
        model_directory = Path(self.model_path)

        if not model_directory.exists():
            raise FileNotFoundError(
                f"Model directory does not exist: {model_directory}"
            )

        if not model_directory.is_dir():
            raise NotADirectoryError(
                f"Model path is not a directory: {model_directory}"
            )

        config_file = model_directory / "config.json"

        if not config_file.is_file():
            raise FileNotFoundError(
                "config.json was not found in the resolved model directory. "
                f"Expected location: {config_file}"
            )

        print(f"Validated local model directory: {model_directory}")
        print(f"Found model configuration: {config_file}")

    @staticmethod
    def _cuda_is_available() -> bool:
        """
        Check whether CUDA is genuinely usable.

        NVIDIA_VISIBLE_DEVICES alone is insufficient because it can be
        populated even when the installed PyTorch or vLLM package is CPU-only.
        """
        try:
            import torch

            return bool(torch.cuda.is_available())
        except (ImportError, RuntimeError) as exception:
            print(f"CUDA availability check failed: {exception}")
            return False

    def _build_command(self) -> List[str]:
        """
        Construct the vllm serve command.

        The model directory is passed as the positional argument. The
        served-model-name provides the client-facing model ID.
        """
        command: List[str] = [
            "vllm",
            "serve",
            self.model_path,
            "--served-model-name",
            self.served_model_name,
            "--dtype",
            "bfloat16",
            "--max-model-len",
            str(self.max_model_len),
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--host",
            self.server_host,
            "--port",
            str(self.server_port),
            "--enforce-eager",
            "--trust-remote-code",
        ]

        return command

    def _start_server(
        self,
        env: Dict[str, str],
    ) -> None:
        """
        Start the vLLM subprocess and wait for API readiness.
        """
        if self.process is not None and self.process.poll() is None:
            print(
                "vLLM is already running with PID "
                f"{self.process.pid}."
            )
            return

        self._is_cuda_visible = self._cuda_is_available()

        if self._is_cuda_visible:
            print("CUDA is available to the vLLM subprocess.")
        else:
            print(
                "CUDA is not available. Starting the installed "
                "vLLM CPU backend."
            )

        command = self._build_command()

        print("Starting vLLM server with command:")
        print(" ".join(command))

        # start_new_session=True creates a separate process group.
        # vLLM may create EngineCore and Worker subprocesses, so stopping
        # the complete process group is safer than killing only the CLI PID.
        self.process = subprocess.Popen(
            command,
            env=env,
            stdout=None,
            stderr=None,
            start_new_session=True,
        )

        print(
            "Created vLLM subprocess with PID "
            f"{self.process.pid}."
        )

        try:
            self._wait_until_server_healthy()
        except Exception:
            self.stop()
            raise

        print(
            "vLLM server is ready. "
            f"PID={self.process.pid}, "
            f"model={self.served_model_name}, "
            f"URL=http://{self.client_host}:{self.server_port}"
        )

    def _wait_until_server_healthy(self) -> None:
        """
        Wait until vLLM is usable.

        A raw TCP-port check is insufficient because vLLM can bind its
        server socket before model loading and initialization are complete.
        """
        deadline = (
            time.monotonic() + self.startup_timeout_seconds
        )

        health_url = (
            f"http://{self.client_host}:{self.server_port}/health"
        )
        models_url = (
            f"http://{self.client_host}:{self.server_port}/v1/models"
        )

        last_error: Optional[Exception] = None

        while time.monotonic() < deadline:
            if self.process is None:
                raise RuntimeError(
                    "The vLLM subprocess was not created."
                )

            return_code = self.process.poll()

            if return_code is not None:
                raise RuntimeError(
                    "vLLM exited before becoming ready. "
                    f"Exit code: {return_code}"
                )

            try:
                health_response = requests.get(
                    health_url,
                    timeout=5,
                )
                health_response.raise_for_status()

                models_response = requests.get(
                    models_url,
                    timeout=10,
                )
                models_response.raise_for_status()

                models_payload = models_response.json()
                model_entries = models_payload.get("data", [])

                available_model_ids = {
                    model_entry.get("id")
                    for model_entry in model_entries
                    if isinstance(model_entry, dict)
                    and model_entry.get("id")
                }

                if (
                    self.served_model_name
                    not in available_model_ids
                ):
                    raise RuntimeError(
                        "vLLM is responding, but the expected model "
                        "is not registered. "
                        f"Expected={self.served_model_name!r}, "
                        f"available={sorted(available_model_ids)!r}"
                    )

                return

            except (
                requests.RequestException,
                ValueError,
                RuntimeError,
            ) as exception:
                last_error = exception

                print(
                    "Waiting for vLLM readiness: "
                    f"{exception}"
                )

                time.sleep(10)

        raise TimeoutError(
            "vLLM did not become healthy within the configured "
            f"{self.startup_timeout_seconds}-second startup period. "
            f"Last readiness error: {last_error}"
        )

    def run(
        self,
        payload: InferencePayload,
    ) -> List[InferenceResult]:
        """
        Send an inference payload to the local vLLM server.
        """
        if self.process is None:
            raise RuntimeError(
                "vLLM has not been started. Call load_model() first."
            )

        return_code = self.process.poll()

        if return_code is not None:
            raise RuntimeError(
                "The vLLM server is no longer running. "
                f"Exit code: {return_code}"
            )

        self._print_cuda_usage()

        return self.client.generate(
            prompts=payload.query,
            params=payload.params,
            task_type=payload.task_type,
        )

    def stop(self) -> None:
        """
        Stop the vLLM API server and its child process group.
        """
        if self.process is None:
            return

        if self.process.poll() is not None:
            self.process = None
            return

        process_id = self.process.pid

        print(
            "Stopping vLLM process group. "
            f"Leader PID={process_id}"
        )

        try:
            os.killpg(process_id, signal.SIGTERM)
            self.process.wait(timeout=30)

        except subprocess.TimeoutExpired:
            print(
                "vLLM did not stop after SIGTERM. "
                "Sending SIGKILL."
            )

            try:
                os.killpg(process_id, signal.SIGKILL)
            except ProcessLookupError:
                pass

            self.process.wait()

        except ProcessLookupError:
            # The process exited between poll() and killpg().
            pass

        finally:
            self.process = None

            try:
                self.client.close()
            except Exception as exception:
                print(
                    "Failed to close the vLLM HTTP client cleanly: "
                    f"{exception}"
                )

    def _print_cuda_usage(self) -> None:
        """
        Print nvidia-smi output only when CUDA is genuinely available.
        """
        if not self._is_cuda_visible:
            print(
                "CUDA is not available. Skipping nvidia-smi."
            )
            return

        try:
            subprocess.run(
                ["nvidia-smi"],
                check=False,
            )
        except (FileNotFoundError, OSError) as exception:
            print(
                "Failed to execute nvidia-smi: "
                f"{exception}"
            )