## Known issue with llama-cpp-python package
With llama-cpp-python==0.3.4, following error is noticed in llama_cpp server side.

```
Exception: 'coroutine' object is not callable
Traceback (most recent call last):
  File "/azureml-envs/minimal/lib/python3.11/site-packages/llama_cpp/server/errors.py", line 173, in custom_route_handler
    response = await original_route_handler(request)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/azureml-envs/minimal/lib/python3.11/site-packages/fastapi/routing.py", line 301, in app
    raw_response = await run_endpoint_function(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/azureml-envs/minimal/lib/python3.11/site-packages/fastapi/routing.py", line 212, in run_endpoint_function
    return await dependant.call(**values)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/azureml-envs/minimal/lib/python3.11/site-packages/llama_cpp/server/app.py", line 491, in create_chat_completion
    llama = llama_proxy(body.model)
            ^^^^^^^^^^^^^^^^^^^^^^^
TypeError: 'coroutine' object is not callable
INFO:     ::1:42174 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
```
Filed [issue](https://github.com/abetlen/llama-cpp-python/issues/1857#issue-2726895443) with owner llama-cpp-python for investigation.

### Solution
We have the solution released in llama-cpp-python==0.3.5 version of it. Otherwise, pin llama-cpp-python==0.3.2, where it also worked fine.
