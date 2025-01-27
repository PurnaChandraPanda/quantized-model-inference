from typing import Dict, List, Any, Optional, Union
import os
import subprocess
import time
import socket
import requests
import json
from dataclasses import dataclass

from constants import ServerSetupParams, WebServer
from inference_payload import InferencePayload, InferenceResult
from webclient import LllamcppClient

class LlamacppEngine():

    def __init__(self, model_path: str):
        self._model_path = model_path
        self._client = LllamcppClient(local_api_url=f"http://{WebServer.HOST}:{WebServer.PORT}")
        self._is_cuda_visible: bool = False
    
    def load_model(self, env: Dict = None):
        """Load the model from the pretrained model specified in the engine configuration."""
        if env is None:
            env = os.environ.copy()
        self._start_server(env=env)

    def _start_server(self, env: Dict):
        """
        definition to start the llama-cpp server
        """
        # Check in env if cuda is visible. 
        ## If yes, load model on GPUs for inferencing. Else, load model CPUs for inferencing.
        if 'NVIDIA_VISIBLE_DEVICES' in env:
            cmd = ["python", "-m", "llama_cpp.server", "--model", self._model_path, "--n_gpu_layers", "-1"]
            # Set the flag to True as its running on GPU
            self._is_cuda_visible = True
        else:
            cmd = ["python", "-m", "llama_cpp.server", "--model", self._model_path]        
        print(f"Starting llama-cpp server with command: {cmd}")

        subprocess.Popen(cmd, env=env)
        self._wait_until_server_healthy(host=WebServer.HOST, port=WebServer.PORT)
        print("Starting llama-cpp server...")

    def _wait_until_server_healthy(self, host: str, port: int, timeout: float = 1.0):
        """Wait until the server is healthy."""
        start_time = time.time()
        while time.time() - start_time < ServerSetupParams.WAIT_TIME_MIN * 60:
            is_healthy = self._is_port_open(host, port, timeout)
            if is_healthy:
                if os.environ.get("LOGGING_WORKER_ID", "") == str(os.getpid()):
                    print("Server is healthy.")
                return
            if os.environ.get("LOGGING_WORKER_ID", "") == str(os.getpid()):
                print("Waiting for server to start...")
            time.sleep(30)
        raise Exception("Server did not become healthy within 15 minutes.")
    
    # Helper function to check if a port is open
    def _is_port_open(self, host: str = "localhost", port: int = 8000, timeout: float = 1.0) -> bool:
        """Check if a port is open on the given host."""
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except (ConnectionRefusedError, TimeoutError, OSError):
            return False

    def run(self, payload: InferencePayload) -> List[InferenceResult]:
        """
        Perform post query to the llama_cpp completion endpoint
        """
        # print("in engine run() :: ", payload)

        # # Print the CUDA usage
        # self._print_cuda_usage()

        # Perform prompt inferencing on the llm model
        inference_results = self._client.generate(payload.query, payload.params, payload.task_type)

        # # Print the CUDA usage
        # self._print_cuda_usage()
        
        return inference_results

    def _print_cuda_usage(self):
        """
        Print the CUDA memory usage.
        """
        if self._is_cuda_visible:
            try:
                cmd = ["nvidia-smi"]
                subprocess.run(cmd)
            except Exception as e:
                print(f"Failed to print CUDA usage: {e}")
        else:
            print("CUDA is not visible. No need to print CUDA usage.")
