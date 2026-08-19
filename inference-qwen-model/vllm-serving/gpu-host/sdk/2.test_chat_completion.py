from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Configured by environment variables
client = OpenAI()

messages = [
    {"role": "user", "content": "Give me a short introduction to large language models."},
]

chat_response = client.chat.completions.create(
    model="Qwen/Qwen3.5-0.8B",
    messages=messages,
    max_tokens=200,
    temperature=1.0,
    top_p=1.0,
    presence_penalty=2.0,
    extra_body={
        "top_k": 20,
    }, 
)

print("Chat response: ", chat_response)

print("Actual message from chat response: ", chat_response.choices[0].message.content)
