# API contract

All endpoints live under `/api/v1` and every response uses the same envelope.

## Response envelope

```json
{
  "data": { "...": "endpoint payload, null on error" },
  "meta": {
    "api_version": "v1",
    "timestamp": "2026-08-06T10:00:00Z",
    "request_id": "uuid",
    "trace_id": "uuid",
    "task_id": "uuid | null",
    "analysis_id": "uuid | null",
    "duration_ms": 412,
    "warnings": []
  },
  "error": null
}
```

- `data` — the endpoint payload; `null` whenever `error` is set.
- `meta` — always present. `request_id` and `trace_id` are echoed in the
  `X-Request-Id` and `X-Trace-Id` response headers. A caller (the Grand
  Orchestrator) may supply either header to propagate its own correlation ids.
- `error` — `null` on success, otherwise an RFC 9457 problem detail extended
  with a stable machine-readable `code`.

Error body:

```json
{
  "data": null,
  "meta": { "...": "as above" },
  "error": {
    "code": "VALIDATION_ERROR",
    "type": "https://umbrella.bio/problems/validation-error",
    "title": "Request validation failed",
    "status": 422,
    "detail": "The request body is invalid.",
    "instance": "/api/v1/protein-structure-analyses",
    "trace_id": "uuid",
    "errors": []
  }
}
```

Codes: `VALIDATION_ERROR`, `PROTEIN_NOT_FOUND`, `UPSTREAM_UNAVAILABLE`,
`AGENT_ERROR`, `UNAUTHORIZED`, `NOT_FOUND`, `INTERNAL_ERROR`.

## Endpoints

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/api/v1/protein-structure-analyses` | Receive a Grand Orchestrator task and run the workflow |
| `POST` | `/api/v1/knowledge/search` | Filtered semantic search over `protein_knowledge` |
| `POST` | `/api/v1/knowledge/protein/ingestions` | Ingest a controlled document set |
| `GET` | `/api/v1/health` | Liveness |
| `GET` | `/api/v1/ready` | Configuration and dependency readiness |
| `GET` | `/api/v1/metrics` | Process-local counters |

`POST /protein-structure-analyses` accepts the `AgentTask` contract
(`task_id`, `trace_id`, `source_agent`, `target_capability`, `schema_version`,
`idempotency_key`, `input`). Its envelope `data` is an inter-agent `AgentResult`:

```json
{
  "status": "completed | continue | needs_agent | failed",
  "target_agent": null,
  "prompt_to_target_agent": null,
  "output": {
    "status": "COMPLETED | PARTIAL | NEEDS_CLARIFICATION | NO_STRUCTURE_FOUND | FAILED",
    "validation_status": "ACCEPT | REVISE | ABSTAIN"
  },
  "continuation_reason": null,
  "retryable": false,
  "error": null
}
```

The outer lowercase `status` is a Grand Orchestrator routing decision. The
uppercase `output.status` is the detailed scientific workflow status.

Routing rules are deterministic:

| Condition | Agent status | Action |
|---|---|---|
| Usable structure result, including a scientifically usable `PARTIAL` result | `completed` | Consume `output`; warnings remain authoritative |
| Caller input needs clarification | `continue` | Correct the input and resubmit the same Protein Agent task |
| Temporary UniProt/PDB/AlphaFold failure prevents a result | `continue` | Retry the same Protein Agent task (`retryable=true`) |
| Canonical protein identity is unresolved | `needs_agent` | Call `genome_agent`, then resume with its accession/sequence evidence |
| Identity is known but no structural evidence can be supported | `needs_agent` | Call `literature_agent`, then resume with cited structural evidence |
| No usable result and no safe retry or specialist hand-off exists | `failed` | Stop and surface `error` |

`target_agent` and `prompt_to_target_agent` are present only for
`needs_agent`. `continuation_reason` is mandatory for `continue`, and `error`
is mandatory for `failed`.

`POST /knowledge/protein/ingestions` requires `X-API-Key` when
`INTERNAL_INGESTION_API_KEY` is set.

The generated OpenAPI document at `/docs` is the canonical field-level schema.
