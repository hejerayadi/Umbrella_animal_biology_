# Multimodal Species Recognition Agent

## Description

Identifies an animal species from **one photograph paired with one text instruction**, by
turning the image into a query vector and retrieving the nearest reference points from a
vector collection. The retrieved references are grouped by species, ranked, checked against
what the instruction says, and passed through a confidence gate that returns `identified`,
`uncertain` or `not_identified`.

## Objective

Bridge computer vision and biological knowledge: a user uploads an animal image, and the
platform returns a grounded identification plus the scientific context the other specialised
agents can supply.

## Request contract

Both parts are required. Image-only and text-only requests are invalid.

```json
{
  "instruction": "Identify this animal and explain the result.",
  "context": {
    "recognition_image": {
      "data_url": "data:image/jpeg;base64,<BASE64>",
      "filename": "observation.jpg"
    }
  }
}
```

`recognition_image` is the only key read for the image. Other context keys are never scanned
for images - a legitimate context may carry another agent's `generated_image`, which is not
this agent's input. JPEG, PNG and WEBP are accepted. Remote URLs and file paths are never
fetched.

## Response

`status` is `completed` for every finished workflow, **including `uncertain` and
`not_identified`** - those are real scientific outcomes, not service failures. `failed` means
the input was invalid or a dependency broke. `needs_agent` means recognition succeeded and the
user asked a follow-up another agent owns.

On `completed`, `output` carries exactly seven keys:

| Key | Meaning |
|---|---|
| `recognition` | decision, text alignment, `similarity_is_probability: false`, grounded explanation, clarification question |
| `species` | scientific name of the primary candidate, or `null` — the shared cross-agent key |
| `species_id` | normalised identifier, or `null` |
| `gbif_id` | mock-fixture GBIF identifier, or `null` — **never invented** |
| `ncbi_taxid` | mock-fixture NCBI identifier, or `null` — **never invented** |
| `recognition_candidates` | ranked distinct species with scores and reference counts |
| `recognition_provenance` | which providers actually produced this answer |

## Architecture

```
POST /execute (api.py)
   -> RecognitionAgent.run                 agent.py
      -> RecognitionWorkflow.run           workflows/graph.py
         validate image + text             validation.py
         analyze text (rules)              text_analysis.py
         mock BioCLIP-2 query vector       adapters/bioclip.py
         retrieve references               adapters/qdrant_{mock,real}.py
         aggregate + rank by species       domain/ranking.py
         mock taxonomy enrichment          adapters/taxonomy.py
         fuse text, apply confidence gate  domain/confidence.py
         build one AgentResult             workflows/graph.py
```

Nodes live in `workflows/nodes.py` and each takes the state plus what it needs, so every one
is testable on its own. Every terminal path - success, uncertain, no match, delegation,
failure - goes through a single result builder, so the shared contract cannot hold on some
branches and not others.

The agent never imports or HTTP-calls another agent. When the instruction asks for something
it does not own, it returns `needs_agent` with a capability hint; the Global Orchestrator's
Capability Resolver chooses who answers and may pick differently.

## What is real and what is mocked

| Component | Status |
|---|---|
| Paired input validation, safe decoding | **Real** |
| Workflow, ranking, confidence, fusion, provenance | **Real** |
| BioCLIP-2 execution | **Mocked** - deterministic test vectors. No package, no weights, no GPU. |
| Vector retrieval | **Local fixtures today.** The real Qdrant adapter exists but refuses to start until the collection contract is supplied. |
| GBIF / NCBI / IUCN | **Mocked** - fixture data, clearly labelled, never verified against a live database |
| Reasoning LLM | **Not called.** Text analysis is rule-based. |

The mock vectors encode nothing biological. **A similarity score is not a probability**, and
nothing in the output presents it as one.

> The local fixture retriever is for development and unit tests. It does **not** satisfy the
> Sprint 2 Qdrant deliverable, and its provenance says `mock_local_development` so a response
> produced from fixtures can never be mistaken for one produced from the real collection.

## Safety properties

- Image bytes and the shared context are excluded from `repr()`, `str()`, `model_dump()` and
  `model_dump_json()` by construction, not by redaction after the fact.
- Failures carry an error code and a fixed message. No request value ever reaches an
  exception, a log line or a response.
- The caller's `context` dict is never mutated; the agent works on a deep copy.
- Encoded size is checked before decoding, decoded size before parsing, header dimensions
  before pixel loading. Decompression bombs are refused from the header.

## Running it

From the repository root, with this agent's virtual environment:

```powershell
python -m venv backend\agents\multimodal_recognition_agent\.venv
backend\agents\multimodal_recognition_agent\.venv\Scripts\Activate.ps1
pip install -r backend\agents\multimodal_recognition_agent\requirements.txt
python -m uvicorn backend.agents.multimodal_recognition_agent.api:app --port 8005
```

Tests:

```powershell
python -m pytest backend\agents\multimodal_recognition_agent\tests -v
```

They need no credentials, no network, no Qdrant and no LLM.

## Configuration

See `.env.example`. Every `QDRANT_*` value belongs to the Qdrant ingestion owner; names are
declared there, values belong only in the git-ignored `.env`. Left empty, real retrieval stays
switched off rather than guessing a collection shape.

## Pending

The vector dimension, collection name, vector name, distance, payload schema, filterable
fields, dataset version and Top-K all come from the Sprint 2 Qdrant manifest and are not
invented here. The deterministic vector algorithm in `adapters/bioclip.py` is documented in
full so the ingestion side can reproduce it byte for byte; both sides must agree on it, and on
the SHA-256 definition in `validation.py`, before fixture-keyed retrieval means anything.
