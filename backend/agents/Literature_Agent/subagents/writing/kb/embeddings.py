import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client_emb = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")


def embed_text(text: str) -> list[float]:
    response = client_emb.embeddings.create(input=text, model=EMBEDDING_DEPLOYMENT)
    return response.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    response = client_emb.embeddings.create(input=texts, model=EMBEDDING_DEPLOYMENT)
    return [d.embedding for d in response.data]