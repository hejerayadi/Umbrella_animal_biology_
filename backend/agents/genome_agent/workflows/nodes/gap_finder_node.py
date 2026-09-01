from __future__ import annotations

import logging
from typing import Any

from ...subagents.gap_finder import find_target_gaps
from ..state import GenomeAgentState

logger = logging.getLogger(__name__)


async def find_target_gaps_node(state: GenomeAgentState) -> dict[str, Any]:
    """Populate state.target_gaps / state.sequence_accession for the handoff.

    Only reached when `join_parallel` routed here because
    `reconstruction_need.status == "NEEDS_AGENT"` (see orchestrator.py). A
    failure here is not fatal to the run: the reconstruction handoff can
    still proceed with an empty `target_gaps` list (and errors surfaced as
    warnings) rather than dropping the escalation entirely — the
    Reconstruction Agent finding out there's nothing concrete to act on is
    better than the whole turn failing.
    """
    assembly_id = state.assembly_id
    logger.info("[find_target_gaps] locating gaps for assembly=%r", assembly_id)

    try:
        result = await find_target_gaps(assembly_id)
    except Exception as exc:
        return {
            "errors": [
                f"find_target_gaps raised an exception: {exc}",
            ],
            "sequence_accession": None,
            "target_gaps": [],
            "gap_selection": None,
        }

    return {
        "sequence_accession": result.get("sequence_accession"),
        "target_gaps": result.get("target_gaps") or [],
        # Kept together with the gaps so the handoff can say how many were
        # found versus how many are being sent.
        "gap_selection": {
            "gaps_found": result.get("gaps_found"),
            "gaps_over_floor": result.get("gaps_over_floor"),
            "gaps_selected": result.get("gaps_selected"),
            "selection_policy": result.get("selection_policy"),
        },
    }
