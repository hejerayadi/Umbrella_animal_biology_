"""Sprint 4 evaluation benchmark and runner for the Multimodal Species
Recognition Agent.

**Phase 1** contributed the data and its validation: `manifest.json` describes a
bounded set of evaluation cases, `manifest_schema.py` holds the accepted-value
vocabularies those cases are checked against (all derived from the agent's own
runtime contracts), and `fetch_assets.py` downloads the images the manifest
points at.

**Phase 2** added the measurement machinery: `runner.py` executes the benchmark
through the agent's ordinary public contract, `evaluators.py` holds fifteen
deterministic evaluators, `results_schema.py` defines the result models and the
redaction boundary, `rubric.py` documents the human scoring rubric, and
`report.py` renders a run as Markdown.

Two properties still hold and are enforced by tests:

* **This package never modifies the agent.** It imports `RecognitionAgent`,
  builds an `AgentRequest` and reads the `AgentResult` — the same public contract
  `api.py` uses. It changes no prompt, no threshold, no provider and no node.
* **The dataset modules stay free of the runner.** `manifest_schema.py` and
  `fetch_assets.py` describe and fetch data; neither may call the agent.

Importing this package has no side effects on runtime behaviour, opens no
connection and reads no credential.

Nothing here may be used to train, fine-tune, prompt-tune or threshold-tune the
agent. See `README.md` and the `_benchmark_note` field inside `manifest.json`.
"""
