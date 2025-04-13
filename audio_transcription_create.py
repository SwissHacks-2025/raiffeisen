from openai import AzureOpenAI
import os
from dotenv import load_dotenv


load_dotenv()

client = AzureOpenAI(
    api_version="2024-02-15-preview",
    azure_endpoint="https://swisshacks-aoai-westeurope.openai.azure.com/",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

audio_file = open("test.m4a", "rb")
transcript = client.audio.transcriptions.create(
    model="whisper", file=audio_file  # Replace with model deployment name
)
print(transcript)

# Access the text content of the transcription object
transcript_text = transcript.text

with open("transcript.txt", "w") as f:
    f.write(transcript_text)
