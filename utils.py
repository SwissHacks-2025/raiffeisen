# from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAI

# Load environment variables from .env file



def get_client():
    load_dotenv()
    return AzureOpenAI(
                        api_version="2024-02-15-preview",
                        azure_endpoint="https://swisshacks-aoai-westeurope.openai.azure.com/",
                        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

if __name__ == "__main__":
    client = get_client()