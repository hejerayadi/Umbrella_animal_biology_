from __future__ import annotations

import logging
from typing import Any

from ...subagents.genome_metadata import get_genome_metadata
from ...subagents.gene_annotation import get_gene_annotation
from ..state import GenomeAgentState

logger = logging.getLogger(__name__)


async def get_genome_metadata_node(state: GenomeAgentState) -> dict[str, Any]:
    if not state.needs_metadata:
        return {"_metadata_done": True}

    assembly_id = state.assembly_id
    logger.info("[get_genome_metadata] fetching metadata for assembly=%r", assembly_id)

    try:
        result = await get_genome_metadata(assembly_id)
    except Exception as exc:
        return {
            "errors": [*state.errors, f"get_genome_metadata raised an exception: {exc}"],
            "metadata": None,
            "_metadata_done": True,
        }

    if result.get("genome_size_bp") is None:
        return {
            "errors": [
                *state.errors,
                f"Genome metadata returned empty for assembly '{assembly_id}'.",
            ],
            "metadata": None,
            "_metadata_done": True,
        }

    return {
        "metadata": result,
        "_metadata_done": True,
    }


async def get_gene_annotation_node(state: GenomeAgentState) -> dict[str, Any]:
    if not state.needs_annotation:
        return {"_annotation_done": True}

    assembly_id = state.assembly_id
    logger.info("[get_gene_annotation] fetching annotation for assembly=%r", assembly_id)

    try:
        result = await get_gene_annotation(assembly_id)
    except Exception as exc:
        return {
            "errors": [
                *state.errors,
                f"get_gene_annotation raised an exception: {exc}",
            ],
            "annotation": None,
            "_annotation_done": True,
        }

    if not result.get("gene_list"):
        return {
            "errors": [
                *state.errors,
                f"Gene annotation returned no genes for assembly '{assembly_id}'.",
            ],
            "annotation": result,
            "_annotation_done": True,
        }

    return {
        "annotation": result,
        "_annotation_done": True,
    }