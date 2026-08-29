# Publication Support

Journal / venue recommendation. A leaf agent under the **Scientific Writing**
sub-orchestrator — it is not reached directly, the writing graph calls it.

```
LiteratureOrchestrator
 └─ ScientificWritingOrchestrator
     ├─ WritingSupportAgent        article text generation
     └─ PublicationSupportAgent    -> this package
```

## Pipeline

```
instruction (+ draft)
  → interpret_query      topic + explicit constraints (open access, publisher bans)
  → embed_query          BAAI/bge-small-en-v1.5
  → search_qdrant        candidates from the `journals` collection (OpenAlex-ingested)
  → hybrid ranking       Qdrant similarity + topic/subfield/field/domain relevance
  → rerank_journals      LLM re-ranks on topical fit alone
  → apply_preferences    constraints applied deterministically, after ranking
```

Preferences are applied *after* the LLM ranks, not inside the prompt. That is
what stops a stated preference from dragging an unrelated journal into the
results: only journals already scored as relevant can be promoted.

## Running the scripts

The package uses **relative imports**, so its modules resolve as part of the
`agents.Literature_Agent` package rather than only when the working directory
is this folder. Run them as modules, from `backend/`:

```bash
# create the journals collection
python -m agents.Literature_Agent.subagents.publication_support.qdrant.qdrant_setup

# ingest OpenAlex journals into it
python -m agents.Literature_Agent.subagents.publication_support.ingestion.qdrant_ingestion

# one-off query against the pipeline
python -m agents.Literature_Agent.subagents.publication_support.ingestion.retrieval

# the interactive agent
python -m agents.Literature_Agent.subagents.publication_support.agent
```

`python agent.py` no longer works — it never resolved the sibling imports from
anywhere except this directory, which is why the orchestrator could not import
this package at all.

## Configuration

Copy `.env.example` to `.env` here. Anything missing falls back to the
`Literature_Agent/.env` two levels up.

| Setting | Purpose |
|---|---|
| `QDRANT_URL` | Cluster URL. `CLUSTER_ENDPOINT` is still accepted as an alias. |
| `QDRANT_API_KEY` | |
| `QDRANT_JOURNALS_COLLECTION` | Defaults to `journals`. |
| `AZURE_OPENAI_ENDPOINT` / `_API_KEY` / `_DEPLOYMENT` | Query interpretation and LLM re-ranking. |

Nothing is built at import time — the Qdrant client, the embedding model and
the OpenAI client are all constructed on first use. A missing setting is a
call-time error that `PublicationSupportAgent` degrades from (falling back to
LLM-only recommendations, reported as `retrieval_backed: false`), never an
`ImportError` that would stop the whole agent from starting.

Dependencies are declared in the agent-wide
[`requirements.txt`](../../requirements.txt); `requirements_agents.txt` in this
folder predates that and is kept only as a record of what this package alone
needs.
