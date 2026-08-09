"""
Client Azure OpenAI centralisé.
Toute la config (endpoint, clé, deployment) est lue depuis les variables
d'environnement (.env) pour ne jamais coder les secrets en dur dans le code.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client: OpenAI | None = None


def get_azure_client() -> OpenAI:
    """
    Retourne une instance unique (singleton) du client Azure OpenAI.
    Évite de recréer une connexion à chaque appel.

    Ce déploiement utilise le nouveau format d'endpoint Azure "/openai/v1",
    qui ne nécessite PAS de paramètre api_version.
    """
    global _client
    if _client is None:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")

        if not endpoint or not api_key:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT et AZURE_OPENAI_API_KEY doivent être définis "
                "dans le fichier .env (voir .env.example)"
            )

        _client = OpenAI(
            base_url=endpoint,
            api_key=api_key,
        )
    return _client


def get_deployment_name() -> str:
    """Retourne le nom du déploiement Azure à utiliser (variable d'env)."""
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    if not deployment:
        raise ValueError(
            "AZURE_OPENAI_DEPLOYMENT_NAME doit être défini dans le fichier .env"
        )
    return deployment