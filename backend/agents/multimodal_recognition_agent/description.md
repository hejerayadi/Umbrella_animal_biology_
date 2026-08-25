# Multimodal Species Recognition Agent

## Description

Identifies an animal species from **one photograph paired with one non-empty text
instruction**, by classifying the image into a ranked list of taxonomic labels. The labels
are checked against what the instruction says, passed through a confidence gate that returns
`identified`, `uncertain` or `not_identified`, and then validated against mocked GBIF and
NCBI sources.

**This agent has one core function and no second one.** It does not search for similar
animals or similar images, it owns no reference-image corpus, and it uses no vector
database.

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
| `recognition` | decision, text alignment, `score_is_probability: false`, grounded explanation, clarification question |
| `species` | scientific name of the primary candidate, or `null` — the shared cross-agent key |
| `species_id` | normalised identifier, or `null` |
| `gbif_id` | mock-fixture GBIF identifier, or `null` — **never invented** |
| `ncbi_taxid` | mock-fixture NCBI identifier, or `null` — **never invented** |
| `recognition_candidates` | ranked distinct taxa with their classification scores |
| `recognition_provenance` | which providers actually produced this answer, and which of them are mocks |

## Architecture

```
POST /execute (api.py)
   -> RecognitionAgent.run                      agent.py
      -> RecognitionWorkflow.run                workflows/graph.py
         validate_image_and_text                validation.py
         plan_or_analyze_text                   text_analysis.py, adapters/reasoning_llm.py
         classify_with_mock_bioclip2            adapters/bioclip.py
         evaluate_confidence                    domain/confidence.py, domain/ranking.py
         validate_taxonomy_with_mock_...        adapters/taxonomy.py
         explain                                workflows/nodes.py
         delegate_if_needed                     workflows/nodes.py
         finalize -> one AgentResult            workflows/graph.py
```

Nodes live in `workflows/nodes.py` and each takes the state plus what it needs, so every one
is testable on its own. Every terminal path - success, uncertain, no match, delegation,
failure - goes through a single result builder, so the shared contract cannot hold on some
branches and not others.

`classify_with_mock_bioclip2` is the **only** source of species candidates. The confidence
gate closes before taxonomy runs, so neither GBIF nor NCBI can add, re-order or re-score a
candidate; and the reasoning model is given a fixed step vocabulary that contains no
retrieval or similarity step to ask for.

The agent never imports or HTTP-calls another agent. When the instruction asks for something
it does not own, it returns `needs_agent` with a capability hint; the Global Orchestrator's
Capability Resolver chooses who answers and may pick differently.

## What is real and what is mocked

| Component | Status |
|---|---|
| Paired input validation, safe decoding | **Real** |
| Workflow, ranking, confidence, delegation, provenance | **Real** |
| BioCLIP-2 label classification | **Mocked** — deterministic fixture keyed by image SHA-256. No package, no weights, no GPU. |
| GBIF / NCBI / IUCN | **Mocked** — fixture data, clearly labelled, never checked against a live database |
| Reasoning LLM | **Optional and disabled by default.** Rules are the default and the fallback. |

A `classification_score` is **not a probability**, and in Sprint 2 it is a deterministic test
value from a fixture rather than model output. Nothing in the response presents it otherwise.

> `fixtures/mock_bioclip_predictions.json` is a test oracle. It is not a scientific dataset
> and must never be presented as real BioCLIP-2 inference. An image whose SHA-256 is not in
> it returns **no candidates** — the agent never attaches a guessed taxon to an unknown
> photograph.

## Not in scope for this agent

Visual similarity search, reference-image retrieval, nearest-neighbour lookup and vector
databases. An instruction that asks only for similar animals or images is answered by species
classification, with `recognition.unsupported_capability` naming what was declined.

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

They need no credentials, no network, no database and no LLM.

## Documentation

`docs/sprint2-final-recognition-only.md` is the final Sprint 2 authority for this agent:
contracts, workflow, mock boundaries, confidence rules, test matrix, run commands, demo
runbook and Definition of Done.

## Configuration

See `.env.example`.
