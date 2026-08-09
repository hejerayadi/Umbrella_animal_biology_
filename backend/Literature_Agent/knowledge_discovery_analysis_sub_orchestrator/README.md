# Knowledge Discovery & Analysis Sub-Orchestrator

Sous-système multi-agents pour l'analyse de littérature scientifique en génomique animale.

## Architecture

```
Knowledge Discovery & Analysis Sub-Orchestrator
├── Retrieval and Knowledge Processing Sub Agent
│     ├── synthesis
│     └── summary
└── Scientific Analysis Sub Agent
      ├── qa
      ├── contradiction_detection
      └── gap_detection
```

L'orchestrateur ne répond jamais directement aux questions : il route
uniquement vers l'agent enfant le plus pertinent, via function/tool calling
(Azure OpenAI). Les agents enfants sont responsables de l'exécution réelle
(RAG sur Qdrant, etc.) — actuellement implémentés en stub, à connecter.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # puis remplir avec tes vraies valeurs Azure
```

## Structure

- `agent_cards/` — fiches descriptives JSON de chaque agent (nom, capacités, schéma d'input)
- `data_classes/` — structures de données partagées (routing, réponses)
- `orchestrators/` — logique de routage + registry (charge les agent cards → tools)
- `agents/` — implémentation de chaque agent enfant
- `llm/` — client Azure OpenAI centralisé
- `tests/` — tests du routage

## Tester le routage

```bash
python -m orchestrators.knowledge_discovery_orchestrator
```

Ou via pytest :

```bash
pytest tests/test_routing.py -v
```

## Statut actuel

- [x] Client Azure OpenAI fonctionnel
- [x] Routage via tool calling validé (2 niveaux : agent → capability)
- [x] Structure de projet (agent cards, data classes)
- [ ] Connexion des agents enfants à Qdrant (RAG réel)
- [ ] Intégration LangGraph pour les workflows conditionnels/parallèles
- [ ] API d'exposition (FastAPI)
