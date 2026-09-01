from __future__ import annotations

import logging
from typing import Any

from ...subagents.genome_metadata import get_genome_metadata
from ...subagents.gene_annotation import get_gene_annotation
from ..state import GenomeAgentState

logger = logging.getLogger(__name__)

# Assembly levels that indicate gaps / unresolved regions requiring reconstruction.
#
# Necessary but not sufficient: an assembly organised into chromosomes is not
# thereby finished, and screening on level alone means a Chromosome-level
# assembly with real unresolved regions is never offered for reconstruction
# (measured: Mus musculus NT_111906.3 holds a 50,000-base run of N). The
# gap-count check below is what catches those, so this set only has to catch
# the assemblies whose level says gaps are expected.
_INCOMPLETE_LEVELS = {"scaffold", "contig"}

# How much of an assembly has to be unresolved before its gap count alone is
# worth escalating over, as a fraction of total length.
#
# A bare `gap_bases > 0` is far too loose: nearly every real assembly reports
# some N. Measured across six assemblies, the spread is not continuous - there
# is roughly a thousandfold gap between the finished and the genuinely
# fragmented, and any threshold inside it separates them:
#
#     Panthera tigris  GCF_018350195.1  Chromosome        10,100 bp   0.0004%
#     Ursus maritimus  GCF_017311325.1  Scaffold       9,414,993 bp   0.404%
#     Mus musculus     GCF_000001635.27 Chromosome    73,600,614 bp   2.698%
#     Homo sapiens     GCF_000001405.40 Chromosome   161,611,139 bp   4.900%
#     Homo sapiens     GCF_009914755.1  Complete               0 bp   0%
#     C. elegans       GCF_000002985.6  Complete               0 bp   0%
#
# The tiger is the case this exists for. Ten kilobases of N in 2.4 Gb is an
# assembly with a few uncallable bases, not one with unresolved regions worth
# sending to reconstruction - and it was the control case in the handoff audit
# precisely because it should complete without escalating. 0.1% sits in the
# empty middle of that spread, far from every measured value, so it is not
# tuned to any single one of them.
_MIN_GAP_FRACTION = 0.001


async def get_genome_metadata_node(state: GenomeAgentState) -> dict[str, Any]:
    assembly_id = state.assembly_id
    logger.info("[get_genome_metadata] fetching metadata for assembly=%r", assembly_id)

    try:
        result = await get_genome_metadata(assembly_id)
    except Exception as exc:
        return {
            "errors": [f"get_genome_metadata raised an exception: {exc}"],
            "metadata": None,
            "_metadata_done": True,
        }

    if result.get("genome_size_bp") is None:
        return {
            "errors": [
                f"Genome metadata returned empty for assembly '{assembly_id}'.",
            ],
            "metadata": None,
            "_metadata_done": True,
        }

    # ── Detect gaps / unresolved regions ──────────────────────────────
    # This check runs unconditionally — even when needs_metadata is False —
    # because assembly_level is the only signal that tells us whether the
    # genome needs reconstruction.  needs_metadata only gates whether the
    # full metadata dict is surfaced to the user; it must not gate the
    # safety check that decides which path the graph takes.
    reconstruction_need = None
    level = (result.get("assembly_level") or "").lower()

    # Two independent reasons to escalate, because either one on its own
    # misses real cases. `assembly_level` is a statement about organisation
    # ("has this been arranged into chromosomes?"), while `gap_bases_bp` is a
    # direct count of unresolved bases from NCBI's own stats - a Scaffold-level
    # assembly may report no gap bases, and a Chromosome-level one may report
    # plenty. `gap_bases_bp` is None when it could not be measured, which is
    # deliberately not treated as zero.
    gap_bases = result.get("gap_bases_bp")
    genome_size = result.get("genome_size_bp")
    gap_fraction = (
        gap_bases / genome_size
        if isinstance(gap_bases, int) and isinstance(genome_size, int) and genome_size > 0
        else None
    )
    # Enough of the genome missing to be worth a reconstruction run. An
    # unmeasurable fraction is not an escalation: `None` means "we did not
    # find out", and guessing in either direction costs more than waiting for
    # the level check to decide.
    has_gap_bases = gap_fraction is not None and gap_fraction >= _MIN_GAP_FRACTION

    if level in _INCOMPLETE_LEVELS or has_gap_bases:
        # The whole clause, not a fragment: the two reasons need different
        # wording, and "Chromosome level with gaps/unresolved regions" would
        # be describing the level as the problem when the count is.
        reason = (
            f"is at '{result['assembly_level']}' level with gaps/unresolved regions"
            if level in _INCOMPLETE_LEVELS
            else f"is at '{result['assembly_level']}' level but reports "
            f"{gap_bases:,} unresolved bases ({gap_fraction:.1%} of the assembly)"
        )
        reconstruction_need = {
            "status": "NEEDS_AGENT",
            "target_agent": None,
            # Carried here (not just left inside state.metadata) so it's
            # available to the adapter/reconstruction handoff even when the
            # caller didn't ask for metadata to be surfaced (needs_metadata
            # gates state.metadata, not this flag).
            "assembly_level": result["assembly_level"],
            # Carried for the same reason as assembly_level: it is the evidence
            # behind the escalation, and the consumer cannot re-derive it.
            "gap_bases_bp": gap_bases,
            "gap_fraction": gap_fraction,
            "prompt_to_target_agent": (
                f"Genome assembly {assembly_id} {reason}. "
                f"Reconstruct the complete genome sequence."
            ),
        }
        logger.info(
            "[get_genome_metadata] assembly %s is incomplete "
            "(level=%s, gap_bases=%s, fraction=%s) — flagging for reconstruction",
            assembly_id,
            result["assembly_level"],
            gap_bases,
            "unknown" if gap_fraction is None else f"{gap_fraction:.4%}",
        )
    # ──────────────────────────────────────────────────────────────────

    # Only populate state.metadata when the caller explicitly asked for it.
    # reconstruction_need is written regardless so the router can always act on it.
    metadata_out = result if state.needs_metadata else None

    return {
        "metadata": metadata_out,
        "_metadata_done": True,
        "reconstruction_need": reconstruction_need,
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

    try:
        result = await get_gene_annotation(assembly_id, user_question=user_question)
    except Exception as exc:
        return {
            "errors": [
                f"get_gene_annotation raised an exception: {exc}",
            ],
            "annotation": None,
            "_annotation_done": True,
        }

    if not result.get("gene_list"):
        return {
            "errors": [
                f"Gene annotation returned no genes for assembly '{assembly_id}'.",
            ],
            "annotation": result,
            "_annotation_done": True,
        }

    return {
        "annotation": result,
        "_annotation_done": True,
    }