import logging
import os
import subprocess
import json
import mlflow
from typing import Dict, List, Union, Tuple
from io import StringIO
from mlflow.pyfunc.scoring_server import infer_and_parse_data, predictions_to_json, _get_jsonable_obj

from engine import LlamacppEngine
from inference_payload import InferencePayload, InferenceResult
from constants import SupportedTask, ALL_TASKS  

def init():
    global model
    global input_schema
    global llama_engine
    global task_type

    # Get the environment variables
    env_vars = os.environ

    # Print each key-value pair
    for key, value in env_vars.items():
        print(f"{key}: {value}")

    # Print ports listening
    print("ports in active mode: ")
    result = subprocess.run(['netstat', '-tulnp'], stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))
    
    # find the llm model
    model_path = None
    file_found = None
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    if model_dir:
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                full_path = os.path.join(root, file)
                if file.endswith('.gguf'):
                    model_path = full_path
                    file_found = True
                    break
            if file_found:
                break
    
    print(f">>> model_path {model_path}")

    # Set the model path in env - will be used in other class
    os.environ["model_path"] = model_path

    local_env = os.environ.copy()
    
    # Load the model
    llama_engine = LlamacppEngine(model_path=model_path)
    llama_engine.load_model(env=local_env)

    # Set task type to chat completion for now
    task_type = SupportedTask.CHAT_COMPLETION

    # Print ports listening
    print("ports in active mode again: ")
    result = subprocess.run(['netstat', '-tulnp'], stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))
    
    print("init() done :: model is loaded")

def run(raw_data):
    print("run() started :: processing data")

    data = json.loads(raw_data)
    
    inference_results = None
    try:
        inference_results, result_dict = _send_request(data)
    except:
        if inference_results is None:
            return {}

    # print("before inference_results loop: print")
    for inference_result in inference_results:
        inference_result.print_results()
    
    # print("before inference_results loop: stats_dict")
    stats_dict = [vars(result) for result in inference_results]
    print(stats_dict)
    
    # print("before _get_jsonable_obj")
    response = _get_jsonable_obj(result_dict, pandas_orient="records")
    print("run() completed :: inferencing over")

    return response

def _send_request(data: Dict) -> Tuple[List[InferenceResult], Dict]:
    
    try:
        # update task type in input data dictionary
        data.update({"task_type": task_type})
        
        payload = InferencePayload.from_dict(data, None)
        payload.update_params(payload.params)
        print(
            f"Processing new request with parameters: {payload.params}",
        )

        results = {}
        inference_results = None

        if task_type == SupportedTask.CHAT_COMPLETION:
            payload.convert_query_to_list()

        # try inferencing
        inference_results = llama_engine.run(payload)
        
        # post processing the inferencing results
        if task_type == SupportedTask.CHAT_COMPLETION:
            outputs = {str(i): res.response for i, res in enumerate(inference_results)}
            results = {
                "output": f"{outputs['0']}",
            }  # outputs will only have one key for chat-completion
        else:
            assert task_type in ALL_TASKS and isinstance(
                payload.query,
                list,
            ), "query should be a list for text-generation"
                        
            results = [res.response for res in inference_results]

        return inference_results, results
    
    except Exception as e:
        print(e)
        raise Exception(
            json.dumps({"error": "Error in processing request", "exception": str(e)})
        )


