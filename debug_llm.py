import os
from openai import OpenAI

# Create client
client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["HF_TOKEN"],
)

# Make a simple test call
response = client.chat.completions.create(
    model=os.environ["MODEL_NAME"],
    messages=[
        {"role": "user", "content": 'Reply ONLY with: {"ok": true}'}
    ],
    temperature=0.0,
)

# Print output
print(response.choices[0].message.content)