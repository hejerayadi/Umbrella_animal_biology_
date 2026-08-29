"""Download the embedding model this subagent needs, once, up front.

`search_literature` refuses to run against a cold cache - the first
`SentenceTransformer(...)` call pulls ~500MB, and huggingface_hub retries a
stalled transfer for a long time before giving up, which would hang a user's
question rather than fail it. So the download is a deliberate setup step
rather than something that happens inside a request.

Run from `backend/`:

    python -m agents.Literature_Agent.subagents.retrieval_knowledge.warm_cache

Downloads resume, so re-running after a dropped connection continues from
where it stopped rather than starting over.
"""
from __future__ import annotations

import sys
import time

# Transfers from the HF CDN drop fairly often on some networks; each attempt
# resumes, so retrying is cheap and usually finishes what the last one started.
ATTEMPTS = 8
BACKOFF_SECONDS = 5

# The repo publishes the same weights several ways - safetensors, a duplicate
# pytorch_model.bin, and onnx and openvino exports. Pulling all of it costs
# ~4GB for a model the loader reads ~500MB of. Take the safetensors and the
# tokenizer/config files around it, and skip the rest.
IGNORE = [
    "onnx/*",
    "openvino/*",
    "pytorch_model.bin",
    "tf_model.h5",
    "rust_model.ot",
    "*.ckpt",
    "*.msgpack",
]


def main() -> int:
    from huggingface_hub import snapshot_download

    from .finalagenttt import EMBEDDING_MODEL

    name = EMBEDDING_MODEL
    if "/" not in name:
        name = f"sentence-transformers/{name}"

    for attempt in range(1, ATTEMPTS + 1):
        try:
            path = snapshot_download(
                name, max_workers=1, etag_timeout=60, ignore_patterns=IGNORE
            )
        except Exception as exc:  # noqa: BLE001
            print(f"attempt {attempt}/{ATTEMPTS} failed: {type(exc).__name__}: {exc}")
            time.sleep(BACKOFF_SECONDS)
            continue
        print(f"cached {name}\n  -> {path}")
        return 0

    print(
        f"could not download {name} after {ATTEMPTS} attempts.\n"
        "Knowledge Discovery will fall back to the labelled placeholder in "
        "subagents/discovery/sources.py until this succeeds."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
