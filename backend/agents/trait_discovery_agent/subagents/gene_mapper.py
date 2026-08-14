from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from schemas.inputs import GeneMapperInput
from schemas.outputs import GeneMapperOutput, GOAnnotation
from schemas.common import AgentStatus
from kb.qdrant_store import get_cached, upsert_point
from kb.sources.go_client import (
    fetch_go_annotation,
    _list_go_candidates_raw as list_go_candidates,
    _resolve_go_term_name_raw as resolve_go_term_name,
)
from workflows.llm import invoke_json_with_fallback

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 1

# Cap concurrent QuickGO name lookups — this is also what was tripping
# QuickGO's own 429 rate limit when several names were resolved back-to-back
# inside the LLM tool loop.
_NAME_RESOLVE_CONCURRENCY = 3


async def _resolve_missing_go_names(candidates: list[dict]) -> list[dict]:
    """Fill in go_name for every candidate up front.

    list_go_candidates always returns go_name=None (§ go_client — QuickGO's
    search endpoint doesn't include names), so without this the LLM has to
    burn a tool-call turn per candidate just to find out what each GO id
    means before it can decide anything. Resolving them here means the model
    sees a fully-named list on turn 1 and — in the common case — doesn't
    need to call a tool at all.
    """
    sem = asyncio.Semaphore(_NAME_RESOLVE_CONCURRENCY)

    async def _fill(c: dict) -> dict:
        if c.get("go_name"):
            return c
        async with sem:
            name = await resolve_go_term_name(c["go_id"])
        return {**c, "go_name": name}

    return await asyncio.gather(*(_fill(c) for c in candidates))

_SYSTEM_PROMPT = (
    "You are the Gene Mapper agent. You're given a trait, a species, and a list of genes "
    "already confirmed relevant to that trait — your job is to attach the correct GO "
    "biological-process annotation to each one, choosing from a candidate list that has "
    "already been resolved to human-readable names.\n\n"
    "If there's more than one candidate:\n"
    "- Pick the one whose name is most plausibly related to the trait as described.\n"
    "- If none of the candidates look trait-relevant, still pick the closest one but say "
    "so plainly in your reasoning — do not invent a better-sounding GO term.\n"
    "- Only report a go_id and go_name that appear in the candidate list below — never "
    "one you recall from elsewhere.\n\n"
    "Reply with ONLY a JSON object with exactly these keys (no markdown fences, no extra "
    'text): "go_id", "go_name", "reasoning".'
)


async def _llm_pick_candidate(
    trait_name: str,
    gene_symbol: str,
    candidates: list[dict],
) -> tuple[str, str, str]:
    """
    Ask the LLM to pick the best candidate. Candidates are pre-resolved (every
    go_name already filled in by _resolve_missing_go_names), so this is a
    single tool-free JSON completion — no bind_tools loop, no turn limit, no
    risk of the model re-fetching unnamed data it doesn't need.

    Returns (go_id, go_name, reasoning). Raises on any failure — the caller is
    responsible for the deterministic fallback (§9).
    """
    candidates_text = "\n".join(
        f"- {c['go_id']}: {c.get('go_name') or '(name unavailable)'}"
        for c in candidates
    )

    human_prompt = (
        f"Trait: {trait_name}\n"
        f"Gene: {gene_symbol}\n\n"
        f"Candidates (this is the complete list — do not assume others exist):\n"
        f"{candidates_text}\n\n"
        "Reply with ONLY the final JSON object."
    )

    parsed = await invoke_json_with_fallback(
        _SYSTEM_PROMPT,
        human_prompt,
        temperature=0.1,
    )

    picked_id = parsed.get("go_id")
    picked_name = parsed.get("go_name") or "unknown"
    reasoning = parsed.get("reasoning", "")

    if not picked_id:
        raise RuntimeError(f"Model did not return a go_id: {parsed!r}")

    return picked_id, picked_name, reasoning


async def gene_mapper_agent(input: GeneMapperInput) -> GeneMapperOutput:
    annotations: list[GOAnnotation] = []
    unmatched: list[str] = []

    for gene in input.gene_list:
        uniprot_accession = input.context.get("uniprot_accessions", {}).get(gene)
        if not uniprot_accession:
            # §8: nothing to disambiguate — never reaches the LLM.
            unmatched.append(gene)
            continue

        # ---- fetch candidates from QuickGO (with one retry, §9) ----
        candidates: list[dict] = []
        try:
            candidates = await list_go_candidates(gene, uniprot_accession)
        except Exception as exc:
            logger.warning("QuickGO error for %s, retrying once: %s", gene, exc)
            try:
                candidates = await list_go_candidates(gene, uniprot_accession)
            except Exception as exc2:
                logger.error("QuickGO failed for %s after retry: %s", gene, exc2)
                unmatched.append(gene)
                continue

        if not candidates:
            unmatched.append(gene)
            continue

        # ---- single candidate: straight through, no LLM (§8) ----
        if len(candidates) == 1:
            go_id = candidates[0]["go_id"]
            go_name = candidates[0].get("go_name") or await resolve_go_term_name(go_id)
            if not go_name:
                unmatched.append(gene)
                continue
            entry = GOAnnotation(gene_symbol=gene, go_id=go_id, go_name=go_name)

        # ---- multi-candidate: LLM pick via bind_tools (§8) ----
        else:
            candidates = await _resolve_missing_go_names(candidates)
            try:
                go_id, go_name, reasoning = await _llm_pick_candidate(
                    input.trait_name, gene, candidates
                )
                # ---- grounding rule (§0.1): validate against actual tool output ----
                valid_ids = {c["go_id"] for c in candidates}
                if go_id not in valid_ids:
                    raise RuntimeError(
                        f"LLM picked invalid go_id {go_id} not in {valid_ids}"
                    )
                logger.info("LLM picked %s (%s) for %s: %s", go_id, go_name, gene, reasoning)
                entry = GOAnnotation(gene_symbol=gene, go_id=go_id, go_name=go_name)
            except Exception as exc:
                # ---- LLM/NIM unavailable or invalid → deterministic fallback (§9) ----
                logger.warning(
                    "LLM pick failed for %s, falling back to first candidate: %s", gene, exc
                )
                fallback = await fetch_go_annotation(gene, uniprot_accession)
                if fallback is None:
                    unmatched.append(gene)
                    continue
                entry = fallback

        # ---- cache layer (§6) ----
        dedup_key = f"go:{entry.go_id}:{entry.gene_symbol}"
        cached = await get_cached("go_annotations", dedup_key)
        if cached:
            annotations.append(entry)
            continue

        await upsert_point(
            "go_annotations",
            dedup_key,
            text_to_embed=entry.go_name,
            payload={
                "gene_symbol": entry.gene_symbol,
                "go_id": entry.go_id,
                "go_name": entry.go_name,
                "source": "GO REST API (QuickGO)",
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "schema_version": SCHEMA_VERSION,
            },
        )
        annotations.append(entry)

    # §9: FAILED if no annotations resolved OR any gene is unmatched
    status = AgentStatus.FAILED if (not annotations or unmatched) else AgentStatus.COMPLETED
    return GeneMapperOutput(
        status=status,
        go_annotations=annotations,
        unmatched_genes=unmatched,
    )


# --------------------------------------------------------------------------- #
#  Mock kept for offline / CI tests (unchanged contract)
# --------------------------------------------------------------------------- #
_MOCK_GO_DB = {
    "FGF5": GOAnnotation(gene_symbol="FGF5", go_id="GO:0031069", go_name="hair follicle development"),
    "KRT71": GOAnnotation(gene_symbol="KRT71", go_id="GO:0031069", go_name="hair follicle development"),
    "HR": GOAnnotation(gene_symbol="HR", go_id="GO:0042633", go_name="hair cycle"),
    "TRPV3": GOAnnotation(gene_symbol="TRPV3", go_id="GO:0050977", go_name="sensory perception of touch"),
    "UCP1": GOAnnotation(gene_symbol="UCP1", go_id="GO:0009408", go_name="response to heat"),
}

async def mock_gene_mapper(input: GeneMapperInput) -> GeneMapperOutput:
    annotations, unmatched = [], []
    for gene in input.gene_list:
        if gene in _MOCK_GO_DB:
            annotations.append(_MOCK_GO_DB[gene])
        else:
            unmatched.append(gene)

    status = AgentStatus.FAILED if (not annotations or unmatched) else AgentStatus.COMPLETED
    return GeneMapperOutput(status=status, go_annotations=annotations, unmatched_genes=unmatched) 