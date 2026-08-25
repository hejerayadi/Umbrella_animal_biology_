"""Sprint 4 evaluation benchmark for the Multimodal Species Recognition Agent.

This package is DATA AND VALIDATION ONLY. It contains a manifest describing a
bounded set of evaluation cases, the accepted-value vocabularies those cases are
checked against, and a small deterministic helper that downloads the images the
manifest points at.

It deliberately contains no runner, no scorer and no call into the agent: this
package cannot execute `RecognitionAgent`, and importing it has no effect on
runtime behaviour. The Phase 2 runner is a separate piece of work.

Nothing here may be used to train, fine-tune, prompt-tune or threshold-tune the
agent. See `README.md` and the `_benchmark_note` field inside `manifest.json`.
"""
