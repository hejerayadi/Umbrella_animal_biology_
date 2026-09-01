"""
Connexion au LLM via l'API Responses (client.responses.create).
Ce deploiement (Azure AI Foundry, gpt-5) expose l'API compatible OpenAI v1,
pas l'API Chat Completions classique qu'utilise llm/client.py pour les autres
roles (ROUTER/WRITING). D'ou un client dedie, construit au premier appel
seulement (jamais a l'import, pour ne pas faire echouer tout l'agent si
la config Azure manque - meme philosophie que retrieval_knowledge/agent.py).
"""
from openai import OpenAI

from .config import get_azure_endpoint, get_azure_api_key, get_azure_deployment

_client = None


def get_llm_client() -> OpenAI:
    global _client
    if _client is None:
        endpoint = get_azure_endpoint()
        api_key = get_azure_api_key()
        if not endpoint or not api_key:
            raise RuntimeError(
                "Azure OpenAI non configure pour Scientific Analysis. "
                "Remplis AZURE_LITERATURE_DISCOVERY_ENDPOINT / _API_KEY / _DEPLOYMENT "
                "dans le .env du Literature_Agent."
            )
        _client = OpenAI(base_url=endpoint, api_key=api_key)
    return _client


def get_deployment_name() -> str:
    deployment = get_azure_deployment()
    if not deployment:
        raise RuntimeError("AZURE_LITERATURE_DISCOVERY_DEPLOYMENT manquant.")
    return deployment
