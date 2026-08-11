# Knowledge Discovery & Analysis Sub-Orchestrator (séquentiel)

Routage 100% piloté par le LLM (tool calling, aucun if/else métier),
exécution des agents strictement séquentielle (pas de fan-out parallèle).

## Setup

```bash
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env         # puis remplir avec tes vraies valeurs Azure
```

## Tester

```bash
python -m orchestrators.graph
```

## Tests automatisés

```bash
pytest tests/test_routing.py -v
```

## Structure

```
agent_cards/            # Fiches JSON des agents enfants
data_classes/           # AgentCall, RoutingDecision, AgentResponse, AggregatedResponse
llm/                    # Client Azure OpenAI centralisé
orchestrators/
  registry.py            # Charge les agent cards -> tools LLM
  knowledge_discovery_orchestrator.py   # route() — routing logic (LLM only)
  graph.py               # Graphe LangGraph séquentiel (delegation + communication + aggregation)
agents/                 # Implémentation des agents enfants (stubs)
tests/                  # Tests pytest
```
