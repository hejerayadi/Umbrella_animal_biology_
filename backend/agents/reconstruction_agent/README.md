# Reconstruction Agent (Agent 5)

Agent LangGraph qui comble les régions `N` d'un génome incomplet (gaps de 10–500 bp)
en produisant des prédictions nucléotidiques, puis en réassemblant la séquence finale.
Expose un endpoint HTTP unique `POST /execute` sur le port **8006**, appelé exclusivement
par l'Orchestrateur global.

## Architecture LangGraph

Le graphe suit une topologie hub-and-spoke : le nœud `reasoning_core` (router conditionnel)
inspecte l'état et route dynamiquement vers les outils (`input_manager` → `gap_locator` →
`model_selector_predictor` → `validation_engine` → `prediction_assembler` → `output_formatter`).
Chaque outil retourne au hub, sauf `validation_engine` qui peut déléguer à
`evolution_agent_delegate` pour les espèces éteintes. Le checkpointer `MemorySaver`
persiste l'état par `thread_id` (= `session_id` du contexte).

## Installation

Depuis la racine du monorepo (Python ≥ 3.12) :

```bash
uv sync
```

Variables Azure OpenAI requises pour le fallback GPT (via `backend/orchestrator/llm.py`) : copier `backend/orchestrator/.env.example` → `backend/orchestrator/.env`.

Variables optionnelles spécifiques à l'agent (défauts entre parenthèses) :

| Variable | Usage |
|---|---|
| `QDRANT_URL` | URL Qdrant pour le contexte flanking (`http://localhost:6333`) |
| `QDRANT_COLLECTION` | Collection de fenêtres (`reconstruction_windows`) |
| `DNA_MODEL_DEVICE` | Device PyTorch DL : `auto`, `cuda`, `cpu` |
| `EMBEDDING_DIM` | Dimension vecteur k-mer (`256`) |
| `RAW_DATA_PATH` | Répertoire FASTA bruts pour l'ingestion (`/tmp/reconstruction_raw`) |

## Lancer l'agent

```bash
uv run python -m uvicorn backend.agents.reconstruction_agent.api:app --port 8006
```

## Lancer les tests

```bash
# Tests unitaires + property-based (recommandé)
uv run pytest tests/reconstruction_agent/ -v

# Tests co-localisés dans le package agent
uv run pytest backend/agents/reconstruction_agent/test_agent5_graph.py -v
uv run pytest backend/agents/reconstruction_agent/data_ingestion/dna_models/test_sprint1_checkpoint.py -v

# Tests d'intégration (services externes stubés, marqueur explicite)
uv run pytest tests/integration/ -m integration -v
```

## Contrat d'entrée / sortie

Entrée — `AgentRequest` JSON :

```json
{
  "instruction": "Reconstruct the genome.",
  "context": {
    "session_id": "uuid-…",
    "genome": "ACGT…NNNN…ACGT",
    "species_metadata": { "species_id": "Panthera tigris", "is_extinct": false },
    "sequence_type": "nuclear"
  }
}
```

Sortie — `ReconstructionResult` avec statuts possibles :

- `completed` — reconstruction terminée (y compris partielle via `error_code=PARTIAL_ASSEMBLY`)
- `needs_agent` — délégation vers `Genome` (génome absent) ou `Evolution` (espèce éteinte)
- `failed` — erreur boundary HTTP

## Pipeline de prédiction (par gap)

1. Extraction contexte flanking (1000 bp) via `RetrievalPipeline` (Qdrant) avec fallback tranche directe.
2. Cascade `DNAModelRouter` : DNABERT → Nucleotide Transformer → Seq2Seq (si checkpoint local) → Azure GPT-5.1.
3. Validation : confidence > 0.7 (+ `evolution_analysis` requis si `is_extinct=true`).
4. Jusqu'à 3 tentatives par gap (`previous_attempts`), puis gap exclu → `is_partial=true`.
5. Assemblage final par remplacement des régions N.

## Pipeline d'ingestion (séparé du graphe)

`IngestionPipeline` (`data_ingestion/pipeline.py`) : Collect → QC → Windowing → Mask → k-mer encode → Store (Qdrant + PostgreSQL). Clients canoniques : `data_ingestion/ncbi_client.py`, `data_ingestion/ena_client.py`.

## État actuel (honnête)

| Composant | Statut |
|---|---|
| Graphe LangGraph + routing dynamique | ✅ Opérationnel |
| Détection gaps 10–500 bp | ✅ Opérationnel |
| API FastAPI `/execute` | ✅ Opérationnel |
| Cascade DNABERT / NT / GPT fallback | ✅ Câblé (DL nécessite GPU + téléchargement HF) |
| Seq2Seq custom | ⚠️ Partial — code présent, inactif sans `seq2seq_checkpoint.pt` |
| Retrieval Qdrant | ⚠️ Partial — fallback séquence directe si Qdrant absent |
| Délégation Evolution Agent | ⚠️ Partial — `NEEDS_AGENT` OK ; reprise timeout non gérée si `evolution_analysis=""` |
| Stockage PostgreSQL ingestion | ⚠️ Partial — stubs SQL, pas de persistance réelle |
| Module évaluation (`evaluation.py`) | ⚠️ Partial — implémenté, non branché au runtime |
| Ingestion ENA/NCBI end-to-end | ⚠️ Partial — clients OK, pipeline complet non testé en prod |
