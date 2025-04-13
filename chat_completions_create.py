from openai import AzureOpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

client = AzureOpenAI(
    api_version="2024-02-15-preview",
    azure_endpoint="https://swisshacks-aoai-westeurope.openai.azure.com/",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

completion = client.chat.completions.create(
    model="gpt-4o",  # Replace with your model dpeloyment name.
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "When was Microsoft founded?"},
    ],
)

print(completion.choices[0].message)
print(completion.model_dump_json(indent=2))
