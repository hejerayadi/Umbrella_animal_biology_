import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
client = OpenAI(
    base_url=os.environ["AZURE_OPENAI_BASE_URL"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
)

response = client.responses.create(
    model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    input="Reply only with OK.",
    reasoning={"effort": "low"},
    max_output_tokens=100,
    store=False,
)

print(response.output_text)