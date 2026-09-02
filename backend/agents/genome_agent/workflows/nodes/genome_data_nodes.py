from __future__ import annotations

import logging
from typing import Any

from ...subagents.genome_metadata import get_genome_metadata
from ...subagents.gene_annotation import get_gene_annotation
from ..state import GenomeAgentState

logger = logging.getLogger(__name__)

# Assembly levels that indicate gaps / unresolved regions requiring reconstruction.
_INCOMPLETE_LEVELS = {"scaffold", "contig"}


async def get_genome_metadata_node(state: GenomeAgentState) -> dict[str, Any]:
    assembly_id = state.assembly_id
    logger.info("[get_genome_metadata] fetching metadata for assembly=%r", assembly_id)

    # Logged before the call, not after: a case that simulates a timeout on
    # this exact call still needs check_tool_selection to see the tool was
    # attempted (see test_cases.yaml: tool_failure_ncbi_timeout expects
    # ncbi_assembly_stats in expected_tool_calls even though it's the tool
    # being made to fail).
    tool_calls = [{"tool": "ncbi_assembly_stats", "args": {"assembly_id": assembly_id}}]

    try:
        result = await get_genome_metadata(assembly_id)
    except Exception as exc:
        return {
            "errors": [*state.errors, f"get_genome_metadata raised an exception: {exc}"],
            "metadata": None,
            "_metadata_done": True,
            "tool_calls_log": tool_calls,
        }

    if result.get("genome_size_bp") is None:
        return {
            "errors": [
                *state.errors,
                f"Genome metadata returned empty for assembly '{assembly_id}'.",
            ],
            "metadata": None,
            "_metadata_done": True,
            "tool_calls_log": tool_calls,
        }

    # ── Detect gaps / unresolved regions ──────────────────────────────
    # This check runs unconditionally — even when needs_metadata is False —
    # because assembly_level is the only signal that tells us whether the
    # genome needs reconstruction.  needs_metadata only gates whether the
    # full metadata dict is surfaced to the user; it must not gate the
    # safety check that decides which path the graph takes.
    reconstruction_need = None
    level = (result.get("assembly_level") or "").lower()
    if level in _INCOMPLETE_LEVELS:
        reconstruction_need = {
            "status": "NEEDS_AGENT",
            "target_agent": None,
            # Carried here (not just left inside state.metadata) so it's
            # available to the adapter/reconstruction handoff even when the
            # caller didn't ask for metadata to be surfaced (needs_metadata
            # gates state.metadata, not this flag).
            "assembly_level": result["assembly_level"],
            "prompt_to_target_agent": (
                f"Genome assembly {assembly_id} is at '{result['assembly_level']}' level "
                f"with gaps/unresolved regions. Reconstruct the complete genome sequence."
            ),
        }
        logger.info(
            "[get_genome_metadata] assembly %s is incomplete (%s) — flagging for reconstruction",
            assembly_id,
            result["assembly_level"],
        )
    # ──────────────────────────────────────────────────────────────────

    # Only populate state.metadata when the caller explicitly asked for it.
    # reconstruction_need is written regardless so the router can always act on it.
    metadata_out = result if state.needs_metadata else None

    return {
        "metadata": metadata_out,
        "_metadata_done": True,
        "reconstruction_need": reconstruction_need,
        "tool_calls_log": tool_calls,
    }


async def get_gene_annotation_node(state: GenomeAgentState) -> dict[str, Any]:
    if not state.needs_annotation:
        return {"_annotation_done": True}

    assembly_id = state.assembly_id
    user_question = state.user_question
    logger.info(
        "[get_gene_annotation] fetching annotation for assembly=%r, question=%r",
        assembly_id,
        user_question,
    )

    # Logged before the call for the same reason as ncbi_assembly_stats
    # above — tool_failure_gene_annotation_timeout expects ncbi_gene_list to
    # show up as attempted even though it's the call being made to fail.
    tool_calls = [{"tool": "ncbi_gene_list", "args": {"assembly_id": assembly_id}}]

    try:
        result = await get_gene_annotation(assembly_id, user_question=user_question)
    except Exception as exc:
        return {
            "errors": [
                *state.errors,
                f"get_gene_annotation raised an exception: {exc}",
            ],
            "annotation": None,
            "_annotation_done": True,
            "tool_calls_log": tool_calls,
        }

    if not result.get("gene_list"):
        return {
            "errors": [
                *state.errors,
                f"Gene annotation returned no genes for assembly '{assembly_id}'.",
            ],
            "annotation": result,
            "_annotation_done": True,
            "tool_calls_log": tool_calls,
        }

    return {
        "annotation": result,
        "_annotation_done": True,
        "tool_calls_log": tool_calls,
    }