"""
Configuration du sous-agent Scientific Analysis.

Qdrant : reutilise le meme cluster que la base de connaissances Writing
(QDRANT_URL / QDRANT_API_KEY dans le .env racine du Literature_Agent), avec
des collections prefixees "sci_" pour ne pas entrer en collision avec les
collections de subagents/writing/kb/.

Azure OpenAI : ce sous-agent utilise l'API "Responses" (client.responses.create),
pas l'API Chat Completions qu'utilise llm/client.py pour ROUTER/WRITING/DISCOVERY.
On reutilise neanmoins le meme ordre de resolution de variables que ce fichier
(role "DISCOVERY" d'abord, puis fallback agent-wide, puis fallback partage),
afin qu'il suffise de remplir le bloc AZURE_LITERATURE_DISCOVERY_* deja
present dans le .env.example racine.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# .env du Literature_Agent, adresse explicitement (meme logique que llm/client.py) :
# le service demarre depuis la racine du repo, donc une recherche relative
# pourrait charger un mauvais .env.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

# --- Qdrant (partage avec subagents/writing/kb/) ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

EMBEDDING_MODEL = os.getenv(
    "SCI_ANALYSIS_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
VECTOR_SIZE = int(os.getenv("SCI_ANALYSIS_VECTOR_SIZE", "384"))

COLLECTIONS = {
    "qa": "sci_qa_corpus",
    "contradiction": "sci_contradiction_claims",
    "gap": "sci_gap_orkg",
}

SCORE_THRESHOLD = float(os.getenv("SCI_ANALYSIS_SCORE_THRESHOLD", "0.65"))
CHUNK_MAX_CHARS = 800
CHUNK_OVERLAP = 100


# --- Azure OpenAI (API Responses), resolu comme llm/client.py mais pour cette API ---
_FIELD_ALIASES = {
    "endpoint": ("ENDPOINT",),
    "api_key": ("API_KEY",),
    "deployment": ("DEPLOYMENT", "DEPLOYMENT_NAME"),
}


def _setting(field: str) -> str | None:
    """Meme ordre de priorite que llm/client.py, role DISCOVERY d'abord."""
    prefixes = ("AZURE_LITERATURE_DISCOVERY_", "AZURE_LITERATURE_", "AZURE_OPENAI_")
    for prefix in prefixes:
        for suffix in _FIELD_ALIASES[field]:
            value = os.getenv(prefix + suffix)
            if value and value.strip():
                return value.strip()
    return None


def _normalise_endpoint(endpoint: str) -> str:
    """Garantit que l'endpoint finit par /openai/v1, format attendu par
    l'API Responses sur ce type de deploiement (Azure AI Foundry)."""
    trimmed = endpoint.rstrip("/")
    if trimmed.endswith("/openai/v1"):
        return trimmed
    # Si l'URL contient deja un chemin /openai/... different (ex: /openai/deployments/...),
    # on la ramene a la racine de la ressource avant d'ajouter /openai/v1.
    root = trimmed.split("/openai/")[0]
    return root + "/openai/v1"


def get_azure_endpoint() -> str | None:
    raw = _setting("endpoint")
    return _normalise_endpoint(raw) if raw else None


def get_azure_api_key() -> str | None:
    return _setting("api_key")


def get_azure_deployment() -> str | None:
    return _setting("deployment")


def is_qdrant_configured() -> bool:
    return bool(QDRANT_URL and QDRANT_API_KEY)


def is_azure_configured() -> bool:
    return bool(get_azure_endpoint() and get_azure_api_key() and get_azure_deployment())
