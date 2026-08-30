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

    tool_log = [{"tool": "ncbi_assembly_stats", "args": {"assembly_id": assembly_id}}]

    try:
        result = await get_genome_metadata(assembly_id)
    except Exception as exc:
        return {
            "errors": [*state.errors, f"get_genome_metadata raised an exception: {exc}"],
            "metadata": None,
            "_metadata_done": True,
            "node_sequence": ["get_genome_metadata"],
            "tool_calls_log": tool_log,
        }

    if result.get("genome_size_bp") is None:
        return {
            "errors": [
                *state.errors,
                f"Genome metadata returned empty for assembly '{assembly_id}'.",
            ],
            "metadata": None,
            "_metadata_done": True,
            "node_sequence": ["get_genome_metadata"],
            "tool_calls_log": tool_log,
        }

    reconstruction_need = None
    level = (result.get("assembly_level") or "").lower()
    if level in _INCOMPLETE_LEVELS:
        reconstruction_need = {
            "status": "NEEDS_AGENT",
            "target_agent": None,
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

    metadata_out = result if state.needs_metadata else None

    return {
        "metadata": metadata_out,
        "_metadata_done": True,
        "reconstruction_need": reconstruction_need,
        "node_sequence": ["get_genome_metadata"],
        "tool_calls_log": tool_log,
    }


async def get_gene_annotation_node(state: GenomeAgentState) -> dict[str, Any]:
    if not state.needs_annotation:
        return {"_annotation_done": True, "node_sequence": ["get_gene_annotation"]}

    assembly_id = state.assembly_id
    user_question = state.user_question
    logger.info(
        "[get_gene_annotation] fetching annotation for assembly=%r, question=%r",
        assembly_id,
        user_question,
    )

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
            "node_sequence": ["get_gene_annotation"],
        }

    if not result.get("gene_list"):
        return {
            "errors": [
                *state.errors,
                f"Gene annotation returned no genes for assembly '{assembly_id}'.",
            ],
            "annotation": result,
            "_annotation_done": True,
            "node_sequence": ["get_gene_annotation"],
        }

    return {
        "annotation": result,
        "_annotation_done": True,
        "node_sequence": ["get_gene_annotation"],
        "tool_calls_log": [{"tool": "ncbi_gene_list", "args": {"assembly_id": assembly_id}}],
    }