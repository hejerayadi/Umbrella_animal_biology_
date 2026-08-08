# Sprint 2 — Final: Recognition Only

**Agent:** Multimodal Species Recognition Agent
**Port:** `8005` · **Endpoint:** `POST /execute`
**Status of this document:** this is the **final Sprint 2 authority** for this agent. Where any
older Recognition specification, architecture note or project document disagrees with it, this
document applies.

---

## 1. The decision

> The Recognition Agent has one core function: receive one animal image plus a non-empty text
> instruction and identify the most likely species. **It does not search for similar animals or
> similar images. It uses no vector database and no reference-image retrieval system.**

Everything that existed only to serve a vector-retrieval pipeline has been removed from this
agent: image embeddings as a result, Qdrant, reference-image hits, nearest-neighbour search,
cosine distance, per-species grouping of retrieved references, dataset and collection manifests,
and the similarity intent and similarity output fields.

This decision applies to `backend/agents/multimodal_recognition_agent/**` only. Qdrant code
owned by other agents or by shared project areas is untouched, and the general project
documents that list Qdrant as a platform-wide option are not a requirement for this agent.

### Explicitly not this agent's job

No reference-image corpus, no vector search, no nearest-neighbour lookup and **no similarity
feature** belong to the Recognition Agent. A request that asks only for visually similar animals
or images is answered by species classification, with `recognition.unsupported_capability` set
to `visual_similarity_search` and a warning saying what was declined. It is never silently
reinterpreted as a similarity search, because there is no such code path to reinterpret it into.

---

## 2. Real vs. mocked in Sprint 2

| Component | Sprint 2 status |
|---|---|
| Paired image + text validation, safe decoding | **Real** |
| LangGraph workflow, routing, single terminal path | **Real** |
| Ranking contract (validate / order / cap) | **Real** |
| Confidence gate and decision logic | **Real** |
| Text analysis, fusion, delegation logic | **Real** |
| HTTP service on port `8005`, shared `AgentResult` | **Real** |
| **BioCLIP-2 label classification** | **MOCKED** — `MockBioCLIP2Provider`, fixture keyed by image SHA-256. No BioCLIP package, no weights, no download, no GPU. |
| **GBIF** | **MOCKED** — `MockGBIFProvider`, fixture records. No live call. |
| **NCBI Taxonomy** | **MOCKED** — `MockNCBIProvider`, fixture records. No live call. |
| Reasoning LLM (GPT-5 mini) | **Optional, disabled by default.** Deterministic rules are the default path and the fallback. Tests need no credentials. |

Three separate mocks, named separately in every response, so no reader can mistake one for a
real service or assume "mocked" applied to only some of them.

---

## 3. Final workflow

```text
START
  -> validate_image_and_text
  -> plan_or_analyze_text
  -> classify_with_mock_bioclip2
  -> evaluate_confidence
  -> validate_taxonomy_with_mock_gbif_and_ncbi
  -> explain
  -> delegate_if_needed
  -> finalize
  -> END
```

Implemented as a real LangGraph `StateGraph` in `workflows/graph.py`; the node names above are
the graph's node keys verbatim.

Two ordering properties carry the safety guarantees:

1. **`classify_with_mock_bioclip2` is the only source of species candidates.** There is no
   second path — no embedding, no retrieval, no lookup — through which a taxon can enter the
   candidate list.
2. **The confidence gate closes before taxonomy runs.** GBIF and NCBI therefore cannot pick a
   species, change a score, change the order or change the decision. They can only annotate
   candidates that already exist.

Every stage routes conditionally: the moment a node writes `error_code`, the graph jumps
straight to `finalize`. **`finalize` is the single terminal node** — success, uncertain, no
match, delegation and failure all arrive there, so the shared contract cannot hold on some
branches and not others. No checkpointer and no store are attached; nothing survives a request.

---

## 4. Input contract

One request carries exactly two things, and **both are required**:

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

- exactly one JPEG, PNG or WEBP image, under the fixed key `recognition_image`;
- a **non-empty** text instruction (empty, whitespace-only and non-text all fail with
  `EMPTY_INSTRUCTION`);
- a required, safe `filename` (no separators, no `..`, no drive letter, no control characters);
- SHA-256 computed over the **base64-decoded encoded file bytes** — this digest is the key the
  mock classifier's fixture is indexed by;
- validation of base64, media type, declared-vs-actual format, byte size, image corruption,
  minimum dimensions and pixel area;
- remote URLs and file paths are **never** fetched.

Other context keys are never scanned for an image: a legitimate context may carry another
agent's `generated_image`, which is not this agent's input. The caller's `context` dict is never
mutated — the agent works on a deep copy with the image key removed.

**No logging, persistence, checkpoint or output ever contains raw image bytes or base64.**
`image_bytes` and `context` are declared `Field(repr=False, exclude=True)` on
`NormalizedRecognitionInput`, so they are absent from `repr()`, `str()`, `model_dump()` and
`model_dump_json()` **by construction** — Pydantic applies `exclude` at the serializer level and
a caller cannot switch it off.

### Ordering of the validation work

Each step is cheap relative to the one after it: the size of the *encoded* string is checked
before any decoding, the decoded size before any parsing, the header dimensions before any pixel
loading. A hostile 200 MB payload is rejected having cost one integer comparison; a
decompression bomb is rejected from its header.

---

## 5. Output contract

The shared `AgentResult` is unchanged: `status`, `target_agent`, `prompt_to_target_agent`,
`output`. `AgentStatus` values and the API error conversion are unchanged.

`completed` is returned for every finished workflow — **including `uncertain` and
`not_identified`**, which are real scientific outcomes, not service failures. `failed` means the
input was invalid or a dependency broke. `needs_agent` means recognition succeeded and the user
asked a follow-up another agent owns.

On `completed`, `output` is a dict with exactly seven keys:

```json
{
  "recognition": {
    "decision": "identified",
    "text_alignment": "neutral",
    "score_is_probability": false,
    "explanation": "The Sprint 2 BioCLIP-2 classification mock supports Panthera leo as the highest-ranked taxonomic label ...",
    "clarification_question": null
  },
  "species": "Panthera leo",
  "species_id": "panthera_leo",
  "gbif_id": 5219404,
  "ncbi_taxid": 9689,
  "recognition_candidates": [
    {
      "species_id": "panthera_leo",
      "scientific_name": "Panthera leo",
      "common_name": "lion",
      "rank": "species",
      "classification_score": 0.91,
      "gbif_id": 5219404,
      "ncbi_taxid": 9689,
      "taxonomy_status": "mock_verified"
    }
  ],
  "recognition_provenance": {
    "model_target": "BioCLIP-2",
    "recognition_provider": "MockBioCLIP2Provider",
    "recognition_mode": "mock_classification",
    "mock_provider_version": "sprint2-mock-bioclip2-classifier-v1",
    "top_k_requested": 5,
    "gbif_mode": "mock",
    "ncbi_mode": "mock",
    "taxonomy_degraded": false,
    "taxonomy_report": { "...": "per-candidate, per-source outcome" },
    "score_is_probability": false,
    "score_kind": "deterministic_sprint2_test_score",
    "text_analysis_mode": "rules",
    "workflow_engine": "langgraph",
    "reasoning_llm_enabled": false,
    "reasoning_llm_provider": "disabled",
    "plan_source": "deterministic",
    "plan_rejected": false,
    "explanation_source": "deterministic",
    "reasoning_llm_calls": 0,
    "reasoning_llm_used": false
  }
}
```

Conditional keys inside `recognition`: `request_better_image` and `better_image_reason` (only
from `not_identified`), `unsupported_capability`, and `warnings`.

`species`, `species_id`, `gbif_id` and `ncbi_taxid` are the **stable cross-agent keys** that
Genome, Biodiversity and Image Generation read. They are preserved unchanged.

`taxonomy_status` takes the repository's established values: `mock_verified` (both mocked
sources held an identifier), `partial` (one did), `unverified` (neither did — the value the
prompt's example writes as `mock_unverified`). None of them means anything was checked against a
live database.

### Fields deliberately removed from the output and provenance

`similarity_score`, `similarity_is_probability`, `reference_count`, `point_id`, `reference_id`,
`dataset_version`, `embedding_mode`, `embedding_provider`, `embedding_dimension`,
`embedding_dimension_source`, `retrieval_provider`, `retrieval_mode`, `collection`,
`qdrant_contract_frozen`. Every one of them asserted something about a vector pipeline that no
longer exists.

---

## 6. BioCLIP-2 mock classifier contract

```python
@runtime_checkable
class BioCLIP2Classifier(Protocol):
    provider_name: str
    version: str
    recognition_mode: str

    def classify(
        self,
        image: NormalizedRecognitionInput,
        top_k: int,
    ) -> list[BioCLIPTaxonPrediction]: ...
```

```python
class BioCLIPTaxonPrediction(BaseModel):
    species_id: str
    scientific_name: str
    common_name: str | None = None
    rank: Literal["species"] = "species"
    classification_score: float
```

A prediction is a **label**, not a database hit: it has no point id, no reference count, no
dataset version and no distance, because a classifier's output has none of those.

### `MockBioCLIP2Provider` behaviour

- returns a **deterministic, already-ranked** Top-K list;
- keyed by the **exact SHA-256** of the decoded image bytes;
- caps results at the configured `top_k`;
- validates the fixture's schema, score range `[0, 1]`, `rank == "species"`, key format, and
  its ranked-and-distinct claim;
- an **unknown hash returns an empty list** — which the confidence gate turns into
  `not_identified`. It never guesses, samples or borrows a biological label;
- never imports BioCLIP, PyTorch, model weights or a vector client, and opens no connection;
- declares `recognition_mode = "mock_classification"` and its provider version;
- labels its scores as **deterministic Sprint 2 test scores**, not calibrated probabilities and
  not scientific confidence.

A malformed, unordered or duplicated fixture raises `CLASSIFICATION_FIXTURE_INVALID` and the
request fails safely. It is never silently sorted into something that looks correct: in Sprint 2
the provider is a test oracle, and an oracle that lies is worth failing on.

### The fixture

`fixtures/mock_bioclip_predictions.json`:

```json
{
  "provider_version": "sprint2-mock-bioclip2-classifier-v1",
  "model_target": "BioCLIP-2",
  "recognition_mode": "mock_classification",
  "score_kind": "deterministic_sprint2_test_score",
  "images": {
    "<exact_sha256>": [
      {
        "species_id": "panthera_leo",
        "scientific_name": "Panthera leo",
        "common_name": "lion",
        "rank": "species",
        "classification_score": 0.91
      }
    ]
  }
}
```

> **The fixture is a test oracle only.** It is not a scientific dataset and must never be
> presented as real BioCLIP-2 inference.

### The later replacement point

```text
MockBioCLIP2Provider.classify(...)
          -> later replaced by ->
RealBioCLIP2Provider.classify(...)
```

The real provider will perform BioCLIP-2 / Tree-of-Life label classification and return Top-K
taxonomic labels through the **same `classify(image, top_k)` signature**. Swapping it in means
passing a different object to `RecognitionAgent(classifier=...)` and changing nothing else. It
**must not** require reintroducing Qdrant, a collection, an embedding or a reference corpus.

---

## 7. Mock GBIF and mock NCBI contracts

Both run **after** classification, on candidates it produced. They may validate a candidate,
supply its identifiers and normalise a synonym to its accepted name. They may **never** create a
species candidate the classifier did not return, re-order one, or re-score one.

`MockGBIFProvider` and `MockNCBIProvider` are independent sources behind the
`MockTaxonomyProvider` facade, so their outcomes are reported separately:

```json
"taxonomy_report": {
  "panthera_leo": {
    "gbif": {"mode": "mock", "available": true, "matched": true,
             "identifier": 5219404, "inconsistent_record": false},
    "ncbi": {"mode": "mock", "available": true, "matched": true,
             "identifier": 9689, "inconsistent_record": false},
    "status": "mock_verified"
  }
}
```

### The branches the workflow must handle

| Branch | Behaviour |
|---|---|
| Complete fixture match | both identifiers supplied → `mock_verified` |
| Partial match (e.g. GBIF present, NCBI absent) | the present identifier only → `partial`; the absent one stays `null` |
| No match | both `null` → `unverified` |
| Unavailable / simulated timeout | `available: false`, `taxonomy_degraded: true`, a warning, request still **completes** |
| Inconsistent fixture record | `matched: false`, `inconsistent_record: true`, **no identifier taken** |

**A missing identifier stays `null`.** It is never filled in, never inferred from a sibling
species, never approximated, and a non-integer fixture value is treated as absent rather than
coerced. `mock_verified` means "the fixture had everything" — it does not mean anything was
verified against a live database.

**No live GBIF, NCBI or IUCN call is made anywhere in Sprint 2 or its tests.** The config refuses
to start with `TAXONOMY_PROVIDER_MODE` set to anything but `mock`, the taxonomy adapter imports
no HTTP client, and no service host appears anywhere in the shipped runtime.

---

## 8. Classification, confidence and text rules

At most `RECOGNITION_TOP_K_SPECIES` distinct taxa are returned, already ordered by
`classification_score`. `domain/ranking.py` holds any provider to that: scores in `[0, 1]`,
non-increasing order, no duplicate species. A provider that breaks the contract fails the
request with `CLASSIFICATION_CONTRACT_VIOLATION` rather than being served a repaired answer.

### The confidence gate

Deterministic, and computed from the top-1 score and the top-1/top-2 margin:

| Condition | Decision |
|---|---|
| no candidates | `not_identified` |
| top-1 < `UNCERTAIN_MIN_SCORE` | `not_identified` |
| text alignment is `conflict` | `uncertain` (never `identified`) |
| top-1 ≥ `IDENTIFIED_MIN_SCORE` **and** margin ≥ `IDENTIFIED_MIN_MARGIN` | `identified` |
| otherwise | `uncertain` |

A single candidate is given a margin of 1.0 — there is no runner-up to be confused with.
`not_identified` additionally sets `request_better_image`; that is the one narrow follow-up the
validated decisions allow, and it is reachable only from `not_identified`, not as a general
clarification route.

> **The thresholds are workflow test boundaries, not scientific calibration.** They decide which
> branch a request takes; they say nothing about biological accuracy. In Sprint 2 the scores they
> compare are deterministic mock values, so `score_is_probability` is `false` in both the
> response body and the provenance, and the explanation says so in plain words.

Taxonomy unavailability stays visible (`taxonomy_degraded`, a warning, per-source report) and
never fabricates an identifier — and, because taxonomy runs after the gate, it cannot change the
decision either.

### What text may and may not do

**May:** detect intent (`recognition` or `scientific_follow_up`), detect language, extract a
location / habitat / suspected taxon hint, ground a short explanation, and decide whether a
completed identification should request another capability via `needs_agent`.

**May not:** perform the visual classification, add a taxon absent from the classifier's Top-K,
change a raw classification score, invent a GBIF or NCBI identifier, trigger a similarity
workflow, or call another agent.

Alignment: `agree` (the named species is the top candidate), `conflict` (the named species is
known but is not the top candidate), `neutral` (no usable or resolvable name). A conflict
downgrades; it never promotes, adds or removes a candidate, and never touches a score.

---

## 9. Reasoning LLM boundary

GPT-5 mini is the approved reasoning model. It is **optional and disabled by default**, and its
role is unchanged from the approved integration.

It may: analyse text, plan within a fixed step vocabulary, and phrase a short explanation
grounded in structured candidates.

It may not: receive raw image bytes, base64 or a data URL; perform visual recognition; generate
candidates; calculate or alter scores; invent taxonomy identifiers; or choose or call a peer
agent. These are enforced by construction — `PlanRequest`, `ExplainRequest`, `ReasoningRequest`
and `ReasoningResult` are frozen dataclasses with no field through which any of it could pass.

```python
ALLOWED_PLAN_STEPS   = ("classify_image", "score_confidence", "validate_taxonomy", "explain")
MANDATORY_PLAN_STEPS = ("classify_image", "score_confidence")
```

A plan naming anything outside that vocabulary — including `embed_image`, `retrieve_candidates`
or `find_similar` — is **rejected whole**, never patched up, and the deterministic plan is used.
The intent enum is `recognition | scientific_follow_up`; `similarity` is not a member, so a model
returning it has its entire answer discarded. An explanation naming a species the classifier did
not return, or presenting the score as a probability, is discarded too.

**Hard budget: two LLM calls per request, ever** — one to plan, one to explain. The budget lives
on a `ReasoningBudget` created fresh per request, so it cannot accumulate across requests. A
planner that failed forfeits the explanation call (one call, not two), because asking again on
the same request would be a retry wearing a different hat.

**Every automated test passes with no LLM credentials and no network access.**

---

## 10. Configuration

`.env.example` is the full list. There is deliberately **no `QDRANT_*` section, no vector
dimension, no collection name, no distance metric and no dataset version** — this agent runs no
vector search, so it has nothing to configure one with and reads no such variable. A stale
`QDRANT_URL` left in an operator's environment is simply ignored.

| Variable | Default | Meaning |
|---|---|---|
| `MAX_IMAGE_BYTES` | `10485760` | decoded byte ceiling |
| `MAX_IMAGE_PIXELS` | `25000000` | pixel-area ceiling |
| `MIN_IMAGE_WIDTH` / `MIN_IMAGE_HEIGHT` | `64` | minimum dimensions |
| `BIOCLIP_PROVIDER_MODE` | `mock` | **must be `mock`**; anything else refuses to start |
| `BIOCLIP_MOCK_PROVIDER_VERSION` | `sprint2-mock-bioclip2-classifier-v1` | version label in provenance |
| `RECOGNITION_CLASSIFICATION_FIXTURE_PATH` | *(empty)* | optional alternative oracle for a demo |
| `TAXONOMY_PROVIDER_MODE` | `mock` | **must be `mock`**; anything else refuses to start |
| `TEXT_ANALYZER_MODE` | `rules` | deterministic rules |
| `RECOGNITION_TOP_K_SPECIES` | `5` | maximum distinct taxa returned, and the `top_k` handed to the classifier |
| `IDENTIFIED_MIN_SCORE` | `0.75` | test boundary |
| `IDENTIFIED_MIN_MARGIN` | `0.08` | test boundary |
| `UNCERTAIN_MIN_SCORE` | `0.45` | test boundary |
| `RECOGNITION_LLM_PROVIDER_MODE` | `disabled` (code) / `fake` (example) | `disabled` \| `fake` \| `azure` |
| `RECOGNITION_LLM_MAX_CALLS_PER_REQUEST` | `2` | hard ceiling, clamped to 2 |
| `AZURE_OPENAI_*` | *(empty)* | names only; values belong in the git-ignored `.env` |

Pinned versions are the repository's: Pydantic `2.13.4`, pytest `8.3.2`. No unrelated dependency
was upgraded, and no vector or model dependency was added.

---

## 11. Revised test matrix

534 tests, all offline. Mapping from the required coverage to where it lives:

| # | Requirement | Where |
|---|---|---|
| 1 | valid image + non-empty instruction | `test_workflow.py`, `test_api.py` |
| 2 | missing image | `test_validation.py`, `test_workflow.py` |
| 3 | empty text | `test_validation.py`, `test_api.py`, `test_workflow.py` |
| 4 | invalid data URL / base64 | `test_validation.py`, `test_api.py` |
| 5 | unsupported media type | `test_validation.py` |
| 6 | corrupt image | `test_validation.py` |
| 7 | oversized bytes / pixel area | `test_validation.py` |
| 8 | filename required and safe | `test_validation.py` |
| 9 | exact SHA-256 of decoded bytes | `test_validation.py`, `test_bioclip.py` |
| 10 | `image_bytes` excluded from repr / serialization / state | `test_no_leak.py` |
| 11 | mock BioCLIP deterministic repeat | `test_bioclip.py`, `test_workflow.py` |
| 12 | known fixture returns ordered Top-K | `test_bioclip.py`, `test_workflow.py` |
| 13 | configured K caps candidates | `test_bioclip.py`, `test_ranking.py`, `test_workflow.py`, `test_phase4_gate.py` |
| 14 | unknown hash → `not_identified`, no random taxon | `test_bioclip.py`, `test_workflow.py` |
| 15 | malformed / unordered fixture fails safely | `test_bioclip.py`, `test_ranking.py` |
| 16 | clear top candidate → `identified` | `test_confidence.py`, `test_workflow.py` |
| 17 | close candidates → `uncertain` | `test_confidence.py`, `test_workflow.py` |
| 18 | generic text → neutral alignment | `test_workflow.py`, `test_phase3_gate.py` |
| 19 | text agreement does not alter the raw score | `test_phase3_gate.py`, `test_workflow.py` |
| 20 | text conflict prevents `identified` | `test_phase3_gate.py` (Gate 2) |
| 21 | text cannot introduce a missing species | `test_phase3_gate.py` (Gate 1) |
| 22 | mock GBIF complete / partial / missing / unavailable | `test_taxonomy.py`, `test_phase4_gate.py` |
| 23 | mock NCBI complete / partial / missing / unavailable | `test_taxonomy.py`, `test_phase4_gate.py` |
| 24 | an absent identifier is never invented | `test_taxonomy.py`, `test_workflow.py` |
| 25 | scientific follow-up after identification → `needs_agent` | `test_workflow.py` |
| 26 | uncertain / not-identified does not delegate | `test_workflow.py` |
| 27 | deterministic LLM fallback with no credentials | `test_phase3_gate.py`, `test_phase4_gate.py` |
| 28 | hard maximum of two LLM calls | `test_phase3_gate.py`, `test_phase4_gate.py`, `test_azure_provider.py` |
| 29 | malicious instruction cannot invoke tools / peers / escape enums | `test_phase3_gate.py`, `test_phase4_gate.py` |
| 30 | standalone `/execute` returns the shared schema | `test_api.py` |
| 31 | Recognition still starts on port `8005` | `test_recognition_only_gate.py` |
| 32 | completed output is mergeable into shared context | `test_recognition_only_gate.py` |
| 33 | no direct peer import, URL or HTTP call | `test_workflow.py`, `test_recognition_only_gate.py` |
| 34 | logs / output contain no image bytes, base64 or secrets | `test_no_leak.py`, `test_recognition_only_gate.py`, `test_azure_provider.py` |
| 35 | provenance says BioCLIP-2 mock, GBIF mock, NCBI mock | `test_recognition_only_gate.py`, `test_workflow.py` |
| 36 | guard against real BioCLIP-2 installation / execution | `test_recognition_only_gate.py`, `test_phase4_gate.py`, `test_bioclip.py` |
| 37 | guard against live GBIF / NCBI calls | `test_recognition_only_gate.py` |
| 38 | no Recognition runtime / config / dependency reference to Qdrant | `test_recognition_only_gate.py` |
| 39 | similarity absent from advertised capabilities and workflow | `test_recognition_only_gate.py`, `test_text_analysis.py`, `test_phase4_gate.py` |
| 40 | unique finalize path, every branch terminates safely | `test_phase4_gate.py`, `test_recognition_only_gate.py` |

Notably, the guard tests in `test_recognition_only_gate.py` parse the shipped source with `ast`
and check **identifiers, imports and string literals while ignoring comments and docstrings** —
so documentation may record that the vector architecture was removed, while executable code
cannot depend on it.

---

## 12. Run commands

From the repository root, with this agent's virtual environment:

```powershell
python -m pytest backend\agents\multimodal_recognition_agent\tests -v
```

```powershell
python -m compileall backend\agents\multimodal_recognition_agent
```

```powershell
python -m uvicorn backend.agents.multimodal_recognition_agent.api:app --port 8005
```

Interactive docs: `http://localhost:8005/docs`.

Regenerate the demo images and print their SHA-256 digests:

```powershell
python backend\agents\multimodal_recognition_agent\fixtures\make_demo_images.py
```

Hash a photograph of your own, to register it in the oracle:

```bash
python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" my_photo.jpg
```

No test needs a credential, a network connection, a database or a model.

---

## 13. Demo runbook

**Setup.** Run `make_demo_images.py`. It writes four PNGs into the git-ignored
`fixtures/demo_images/` and prints their digests. Three of them are already keys in
`mock_bioclip_predictions.json`; `demo_unknown.png` deliberately is not.

Start the service:

```powershell
python -m uvicorn backend.agents.multimodal_recognition_agent.api:app --port 8005
```

| Step | What to show | Expected |
|---|---|---|
| 1 | The service starts | listening on **`8005`**, `/docs` reachable |
| 2 | `POST /execute` with `demo_identified.png` + "Identify this animal." | `status: completed` |
| 3 | The Top-K in `recognition_candidates` | three ranked taxa, `Panthera leo` first at `0.91` |
| 4 | `recognition.decision` | **`identified`** |
| 5 | Repeat the same request | byte-identical output — determinism |
| 6 | `demo_uncertain.png` | **`uncertain`** (`0.81` vs `0.78`, margin below `0.08`) plus a clarification question |
| 7 | `demo_partial_taxonomy.png` | `Ursus maritimus`, `gbif_id: null`, `ncbi_taxid: 29073`, `taxonomy_status: "partial"` — the missing identifier stays null |
| 8 | Invalid input: empty instruction; `context: {}`; `data:image/png;base64,!!!` | `status: failed`, codes `EMPTY_INSTRUCTION`, `MISSING_IMAGE`, `INVALID_BASE64` — never a 500 |
| 9 | `demo_unknown.png` | **`not_identified`**, `species: null`, empty candidates, `request_better_image: true` — no fabricated species |
| 10 | `demo_identified.png` + "What is the evolutionary history of this animal?" | `status: needs_agent`, `target_agent: "Evolution"`, `output: null` — a hint, with no peer call, no URL and no port in the prompt |
| 11 | "Which species look similar to this one?" | answered by classification, with `unsupported_capability: "visual_similarity_search"` and a warning |
| 12 | `recognition_provenance` on any response | `model_target: BioCLIP-2`, `recognition_mode: mock_classification`, `gbif_mode: mock`, `ncbi_mode: mock`, `score_is_probability: false` |

---

## 14. Definition of Done

- [x] The workflow, validation, API, ranking, confidence, delegation, error handling and
      response building are **real** and tested.
- [x] BioCLIP-2 execution is **mocked** behind the final `classify(image, top_k)` interface.
- [x] GBIF and NCBI are **mocked**, with complete / partial / missing / unavailable /
      inconsistent branches covered.
- [x] No real BioCLIP-2 package, weights, inference, GPU or model download.
- [x] No live GBIF, NCBI or IUCN call, in the runtime or in the tests.
- [x] **No Qdrant** in the Recognition runtime, configuration, dependency list, fixtures,
      workflow or tests.
- [x] **No similarity capability** in the intent enum, plan vocabulary, graph, `card.json`,
      `description.md`, output fields or tests.
- [x] The agent answers `POST /execute` on port `8005` with the unchanged shared `AgentResult`.
- [x] `needs_agent` is a capability hint; no peer is imported, addressed or called.
- [x] Scores are labelled as deterministic Sprint 2 test values, never as probabilities.
- [x] Every automated test passes with no credentials and no network.
- [x] The mock classifier can be replaced by real BioCLIP-2 label classification through the
      same interface, with no vector database.

---

## 15. What Sprint 2 claims

> Sprint 2 provides a real, validated, deterministic Recognition Agent workflow whose BioCLIP-2
> classification boundary and GBIF/NCBI adapters are mocked. It supports photo-to-species Top-K
> classification and controlled confidence decisions, with no vector database, reference-image
> retrieval, or similarity feature. The mock classifier can later be replaced by real BioCLIP-2
> label classification through the same interface.

**No claim of scientific recognition accuracy is made for Sprint 2.**
