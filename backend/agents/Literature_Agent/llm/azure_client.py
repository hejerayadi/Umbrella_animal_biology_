"""
Client Azure OpenAI centralisé.
Format d'endpoint /openai/v1 - pas d'api_version requis.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client: OpenAI | None = None


def get_azure_client() -> OpenAI:
    global _client
    if _client is None:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not endpoint or not api_key:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT et AZURE_OPENAI_API_KEY doivent être définis "
                "(fichier .env, voir .env.example)"
            )
        _client = OpenAI(base_url=endpoint, api_key=api_key)
    return _client


def get_deployment_name() -> str:
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    if not deployment:
        raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME doit être défini dans .env")
    return deployment
