# Recognition Agent — Sprint 4 evaluation benchmark

**This is a bounded Sprint 4 benchmark. It is not a general model-accuracy dataset.**

It exists to *measure the existing agent*, on a small, fully documented set of
cases, so Sprint 4 can report evidence instead of an assertion. It is far too
small to support any claim about BioCLIP-2's accuracy, about species recognition
in general, or about how the agent would behave on any other distribution of
images.

## What this package is allowed to be used for

- Executing the already-implemented Recognition Agent against documented cases
  and recording what it did (Sprint 4 Phase 2 and Phase 3).
- Computing the Sprint 4 evaluation criteria: correctness, relevance, task
  completion, agent/tool selection, response consistency, and the
  Recognition-specific evidence (Top-1, Top-5, decision, taxonomy identifiers,
  provenance truthfulness, latency, controlled-failure rate).
- Classifying weaknesses in a report.

## What it must never be used for

- Training or fine-tuning any model.
- Prompt tuning.
- Threshold tuning. The gate values `0.75 / 0.08 / 0.45` were fixed long before
  this dataset existed and must not be moved because of anything observed here.
- Provider, workflow or runtime modification of any kind.
- Any published accuracy figure presented as a general capability claim.

Sprint 4 requires *valid measurement and weakness analysis*, not a good score.
A case the agent gets wrong is a finding to report, never a case to rewrite.

## Ground-truth policy

1. **Ground truth never comes from the agent.** No expected value in
   `manifest.json` was copied, derived or confirmed from a Recognition Agent
   output. The agent has not been executed against this dataset.
2. **Species identity** comes from the image's own public source page on
   Wikimedia Commons (its file page and species category), and every selected
   image was additionally inspected by eye before it was accepted — which is how
   several category members that contain no live animal at all were caught and
   either rejected or deliberately re-purposed as negative cases.
3. **Taxonomy identifiers** were verified independently against the official
   sources on 2026-08-25:
   - GBIF `usageKey` via the GBIF species-match service, accepted only on
     `matchType == "EXACT"`, `rank == "SPECIES"` and `status == "ACCEPTED"`;
   - NCBI `taxid` via the official NCBI Taxonomy Browser, accepted only when an
     entry of rank *species* carried the exact binomial.
   Where the two disagree with the agent's own development fixture, **the
   official sources win and the difference is recorded** — see
   `docs/PHASE1_SPRINT4_EVALUATION_DATASET_REPORT.md`.
4. **Ambiguous, poor-quality and non-animal cases are never forced into
   exact-species scoring.** Each declares which metrics apply and which do not,
   in its `applicability` block.
5. **Invalid-input and dependency-failure cases test behaviour**, not biology.
   They expect a status and a fixed `error_code`, never a species.

## Images are never committed

No image byte, Base64 string, data URL or private path is stored in this
repository. Each case records a public source page and a public asset URL.
`fetch_assets.py` downloads them into `assets/`, which is git-ignored.

Every listed image is under a free licence (CC0, CC BY, CC BY-SA, or public
domain) and carries its licence and attribution in the manifest. No private
photograph is used.

## Layout

| Path | Role |
| --- | --- |
| `manifest.json` | The dataset. Every case, its expectations and its provenance. |
| `manifest_schema.py` | Accepted vocabularies, imported from the agent's own runtime contracts, plus the deterministic loader. |
| `fetch_assets.py` | Minimal, deterministic, opt-in downloader. Not a runner. |
| `assets/` | Git-ignored. Downloaded images live here. |

The validation tests live with the rest of the agent's suite, in
`tests/test_sprint4_phase1_evaluation_dataset.py`, so they run as part of the
complete Recognition offline suite.

## Fetching the images

```bash
backend/agents/multimodal_recognition_agent/.venv/Scripts/python.exe \
  backend/agents/multimodal_recognition_agent/evaluation/fetch_assets.py
```

It is opt-in, sequential, paced, and writes only into `assets/`. It never runs
during a test: the whole validation suite is offline and touches no network.
