"""Callable entry point for the Retrieval & Knowledge Processing subagent.

`finalagenttt.py` is the engine - LLM query expansion, Firecrawl web search,
OpenAlex academic search, sentence-transformer ranking and a Qdrant semantic
cache - but its only way in is `main()`, an interactive `input()` loop. This
module is the adapter that makes it callable from the Knowledge Discovery
sub-orchestrator, and translates its output into the discovery payload that
`subagents/discovery/sources.py` documents.

Nothing here is imported at module scope from the engine: `finalagenttt`
builds a sentence-transformer on first use and reads Qdrant/Groq/Firecrawl
settings from its `.env`, so it is imported inside `search_literature()`.
That keeps an unconfigured or uninstalled retrieval stack a call-time failure
this module degrades from, rather than an ImportError that would take the
whole Literature Agent service down at start.
"""
from __future__ import annotations

import contextlib
import io
import logging
from typing import Any

_logger = logging.getLogger(__name__)

# How many engine results become discovery records. The engine itself fans out
# to several generated queries per call, so this caps the answer, not the work.
DEFAULT_LIMIT = 5


def is_configured() -> bool:
    """Whether a real search can be attempted. Never raises."""
    try:
        from .finalagenttt import missing_settings

        return not missing_settings()
    except Exception:  # noqa: BLE001 - missing deps must not be fatal here
        return False


def embedding_model_is_cached() -> bool:
    """Whether the sentence-transformer weights are already on disk.

    Checked before searching because the first `SentenceTransformer(...)` call
    downloads ~500MB, and huggingface_hub retries a stalled transfer for a
    long time before giving up. That turns a cold cache into a request that
    hangs for minutes rather than one that fails - and discovery is on the
    path of a user's question, so a hang is the worse outcome. Reporting
    "not cached" lets `sources.search_papers` fall back to its labelled
    placeholder immediately.

    Warm the cache once, up front, with:

        python -m agents.Literature_Agent.subagents.retrieval_knowledge.warm_cache
    """
    try:
        from huggingface_hub import try_to_load_from_cache

        from .finalagenttt import EMBEDDING_MODEL

        name = EMBEDDING_MODEL
        if "/" not in name:
            name = f"sentence-transformers/{name}"

        # Check the specific files the loader needs, not the whole repo. A
        # repo-level check (snapshot_download(local_files_only=True)) demands
        # every file the repo publishes - including the onnx and openvino
        # variants and the duplicate pytorch_model.bin - so a cache warmed by
        # any normal means looks "missing" and retrieval degrades to the
        # placeholder for no reason.
        if not isinstance(try_to_load_from_cache(name, "config.json"), str):
            return False
        return any(
            isinstance(try_to_load_from_cache(name, weights), str)
            for weights in ("model.safetensors", "pytorch_model.bin")
        )
    except Exception:  # noqa: BLE001
        return False


def _citation_line(result: dict[str, Any]) -> str:
    """One human-readable reference line, in the shape `papers` promises.

    This is what the writing subagent is allowed to cite, so it carries the
    identifier a reader can follow - DOI when the engine found one, otherwise
    the source URL.
    """
    title = (result.get("title") or "Untitled").strip()
    year = result.get("year")
    authors = result.get("authors")
    doi = (result.get("doi") or "").strip()
    url = (result.get("url") or "").strip()

    parts = [title]
    if authors:
        parts.insert(0, authors if isinstance(authors, str) else ", ".join(authors))
    if year and year != "N/A":
        parts.append(f"({year})")
    if doi:
        parts.append(f"https://doi.org/{doi}")
    elif url:
        parts.append(url)
    return " ".join(str(p) for p in parts)


def _to_record(result: dict[str, Any]) -> dict[str, Any]:
    """One structured evidence record.

    `pmid` is always None. The engine searches Firecrawl and OpenAlex, never
    PubMed (`Config.PUBMED_*` is declared but unused), so no PubMed identifier
    exists for these hits. The Trait Discovery Agent skips records without a
    pmid, which is the intended outcome: it writes the pmid it is handed
    straight into Neo4j as evidence for a trait->gene edge, and a fabricated
    one would put an invented citation into shared, persisted scientific
    state. The DOI is carried alongside so the record stays useful to
    everything that does not require a pmid.
    """
    return {
        "pmid": None,
        "doi": (result.get("doi") or "").strip() or None,
        "title": (result.get("title") or "").strip(),
        "year": result.get("year"),
        "short_summary": (result.get("abstract") or "")[:500],
        "url": (result.get("url") or "").strip() or None,
        "source": result.get("source"),
    }


def search_literature(query: str, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    """Run the retrieval engine and return the discovery payload.

    Raises nothing: a failure comes back as an empty result the caller can
    detect via `records`/`papers` being empty, so `sources.search_papers` can
    fall back to its labelled placeholder instead of the agent erroring out.
    """
    if not embedding_model_is_cached():
        _logger.warning(
            "embedding model is not in the local cache; skipping retrieval rather "
            "than blocking the request on a ~500MB download"
        )
        return _empty("retrieval_model_not_cached")

    try:
        from .finalagenttt import UnifiedSearchWithLLM
    except Exception as exc:  # noqa: BLE001
        _logger.warning("retrieval engine unavailable (%s): %s", type(exc).__name__, exc)
        return _empty("retrieval_unavailable")

    # The engine narrates every step with emoji-laden print() calls - it began
    # life as a Colab notebook. Two problems as a library: the chatter does not
    # belong in a service log, and on Windows a redirected stdout is cp1252, so
    # the first emoji raises UnicodeEncodeError and takes the whole search with
    # it. Capturing into a StringIO sidesteps the codec entirely (it accepts any
    # str) and keeps the trace available at DEBUG.
    chatter = io.StringIO()
    try:
        with contextlib.redirect_stdout(chatter):
            raw = UnifiedSearchWithLLM().search(query, limit=limit)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("retrieval search failed (%s): %s", type(exc).__name__, exc)
        _logger.debug("engine output before failure:\n%s", chatter.getvalue())
        return _empty("retrieval_failed")
    finally:
        _logger.debug("engine output:\n%s", chatter.getvalue())

    results = (raw or {}).get("results") or []
    return {
        "papers": [_citation_line(r) for r in results],
        "records": [_to_record(r) for r in results],
        "total_found": raw.get("total_found", len(results)),
        "source": "retrieval_knowledge",
    }


def _empty(source: str) -> dict[str, Any]:
    return {"papers": [], "records": [], "total_found": 0, "source": source}
