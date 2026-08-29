# Retrieval & Knowledge Processing

Finds real literature for a query. A leaf agent under the **Knowledge
Discovery** sub-orchestrator — it is not reached directly, discovery calls it.

```
LiteratureOrchestrator
 └─ KnowledgeDiscoveryOrchestrator
     └─ retrieval_knowledge   -> this package
```

## Pipeline

```
user query
  → QueryGenerator          Groq expands it into ~5 complementary queries
  → FirecrawlSearchEngine   web search + scrape
  → AcademicSearchEngine    OpenAlex, 2020-2025, sorted by citations
  → SemanticEmbeddingEngine paraphrase-multilingual-MiniLM-L12-v2
  → ranking + dedup         cosine similarity against the query
  → Qdrant                  semantic cache of articles and syntheses
```

## Layout

| File | Role |
|---|---|
| `agent.py` | The callable entry point. Discovery imports **only** this. |
| `finalagenttt.py` | The engine (originally a Colab notebook export). |

`agent.py` exists because the engine's only entry point is `main()`, an
interactive `input()` loop. It translates the engine's output into the
discovery payload documented in
[`../discovery/sources.py`](../discovery/sources.py), and never raises — a
failure comes back as an empty result so discovery can fall back to its
labelled placeholder instead of taking the agent down.

### `pmid` is always `None`

The engine searches Firecrawl and OpenAlex, never PubMed (`Config.PUBMED_*`
is declared but unused), so no PubMed identifier exists for these hits. The
Trait Discovery Agent writes the pmid it is handed straight into Neo4j as
evidence for a trait→gene edge, so emitting a fabricated one would persist an
invented citation into shared scientific state. It skips records with no pmid,
which is the intended outcome. The DOI is carried alongside for everything
that does not require a pmid.

## First-run setup: warm the model cache

The embedding model (~500MB) is **not** downloaded inside a request.
`search_literature` checks the local cache first and returns nothing if the
weights are missing, so discovery falls back to its labelled placeholder
rather than hanging a user's question for minutes on a download that
huggingface_hub retries for a long time before giving up.

Do it once, from `backend/`:

```bash
python -m agents.Literature_Agent.subagents.retrieval_knowledge.warm_cache
```

Downloads resume, so re-run it if the connection drops. Until it succeeds,
Knowledge Discovery answers from the placeholder and says so via
`is_placeholder` / `notice`.

## Running it standalone

From `backend/`:

```bash
python -m agents.Literature_Agent.subagents.retrieval_knowledge.finalagenttt
```

This folder was named `retrieval&knowledge processing`, which is not a Python
identifier — no `import` statement could name it, which is why nothing in the
repository referenced this code.

## Configuration

Copy `.env.example` to `.env` here. Anything missing falls back to the
`Literature_Agent/.env` two levels up.

| Setting | Purpose |
|---|---|
| `QDRANT_URL` / `QDRANT_API_KEY` | Semantic cache of articles and syntheses. |
| `QDRANT_COLLECTION` | Defaults to `lina_researchPaperArticles`. |
| `QDRANT_VECTOR_SIZE` | 384, must match `EMBEDDING_MODEL`. |
| `GROQ_API_KEY` / `GROQ_API_URL` / `GROQ_MODEL` | Query expansion, summaries, synthesis. |
| `FIRECRAWL_API_KEY` / `FIRECRAWL_BASE_URL` | Web search and scraping. |
| `EMBEDDING_MODEL` | sentence-transformers model, run locally. |

`missing_settings()` reports what is unset; `test_apis()` prints it before
touching the network. Nothing is built at import time — the embedding model
and the Qdrant client are constructed on first use, and `setup_collection()`
only ever *creates*, so running the agent cannot drop the indexed corpus.

> **These credentials were hardcoded in `finalagenttt.py` and are in the git
> history (commit `5222f64`). Rotate the Qdrant, Groq and Firecrawl keys.**
