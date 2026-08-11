"""Scientific Analysis Sub Agent. Stub — à connecter à une vraie source de données."""


def run(capability: str, query: str) -> dict:
    return {
        "agent_name": "scientific_analysis",
        "capability": capability,
        "result": f"[STUB] Résultat simulé pour capability='{capability}' sur la requête: {query}",
        "success": True,
    }
