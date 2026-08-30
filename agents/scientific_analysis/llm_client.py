"""
Connexion au LLM.
Ce deploiement Azure AI Foundry expose l'API "Responses" compatible OpenAI v1
(client.responses.create), pas l'API Chat Completions classique.
On utilise donc le client OpenAI standard, pointe vers l'endpoint Azure.
"""
from openai import OpenAI

from config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT

_client = None


def get_llm_client() -> OpenAI:
    """Retourne une instance (mise en cache) du client OpenAI configure pour Azure."""
    global _client
    if _client is None:
        if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
            raise RuntimeError(
                "Configuration Azure manquante. "
                "Verifie AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY / "
                "AZURE_OPENAI_DEPLOYMENT dans ton .env. "
                "AZURE_OPENAI_ENDPOINT doit finir par '/openai/v1' "
                "(ex: https://ta-ressource.services.ai.azure.com/openai/v1)"
            )
        _client = OpenAI(base_url=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_API_KEY)
    return _client


def get_deployment_name() -> str:
    if not AZURE_OPENAI_DEPLOYMENT:
        raise RuntimeError("AZURE_OPENAI_DEPLOYMENT manquant dans ton .env")
    return AZURE_OPENAI_DEPLOYMENT