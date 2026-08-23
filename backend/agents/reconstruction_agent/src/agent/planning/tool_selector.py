"""Turns a plan step into a concrete, validated tool invocation.

The planner names a tool and a gap; this builds the typed payload that tool
needs from the current state. Keeping that translation here means the planner
never has to know a tool's input schema.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from agent.planning.planner import PlanStep
from agent.state.state import ReconstructionState, attempt_key
from configuration.logging import get_logger
from domain.models import GapContext, Reference
from domain.policies.reference_quality import ReferenceQualityPolicy
from domain.services import ReferenceRanker
from domain.services.alignment_window import window_residues
from domain.services.organism_affinity import organism_affinity
from tools.blast.schemas import BlastSearchInput
from tools.evo.schemas import EvolutionaryContextInput, PlausibilityInput
from tools.mafft.schemas import AlignmentInput
from tools.ncbi.schemas import NCBISearchInput

_log = get_logger(__name__)

# How many references to carry into an alignment. MAFFT slows sharply with row
# count, and past roughly this many the consensus stops changing.
_MAX_ALIGNMENT_REFERENCES = 8


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    """A tool name paired with the payload to run it on."""

    tool: str
    payload: Any
    gap_id: str | None = None
    #: True when this payload was widened after an earlier attempt found
    #: nothing. Carried into the observation so a retry is visible as one.
    relaxed: bool = False


#: What a second attempt widens. The first pass is deliberately strict - a
#: permissive e-value returns chance similarity, which costs an alignment round
#: to discover. Only once strict has failed is it worth trading precision for
#: reach.
_RELAXED_EXPECT = 1e-3
_RELAXED_MAX_HITS = 100
#: A broader ENA division: the strict default is vertebrate coding sequence, so
#: a gap in a non-coding or poorly-annotated region finds nothing there.
_RELAXED_DATABASE = "em_rel_vrt"

#: Annotation-based judgement, shared by every selection.
_QUALITY = ReferenceQualityPolicy()


class ToolSelector:
    """Builds tool payloads from plan steps and state."""

    def __init__(self, ranker: ReferenceRanker | None = None) -> None:
        self._ranker = ranker or ReferenceRanker()

    def build(self, step: PlanStep, state: ReconstructionState) -> ToolInvocation | None:
        """The invocation for one step, or None when it cannot be run.

        Returning None rather than raising: a step whose preconditions are no
        longer met (its gap got skipped, its references never arrived) should
        be quietly dropped so the rest of the plan still executes.
        """
        context = self._context_for(step.gap_id, state)
        # A second attempt at the same tool/gap pair widens its search rather
        # than repeating the first byte for byte, which could only ever return
        # the same answer.
        relaxed = (state.get("attempts") or {}).get(
            attempt_key(step.tool, step.gap_id), 0
        ) > 0

        if step.tool == "blast_search":
            return self._blast(step, context, relaxed)
        if step.tool == "mafft_align":
            return self._mafft(step, context, state, relaxed)
        if step.tool == "ncbi_search":
            return self._ncbi(step, context, state, relaxed)
        if step.tool == "evolutionary_context":
            return self._evo(step, state)
        if step.tool == "evo2_plausibility":
            return self._plausibility(step, context, state)

        _log.warning("tool_payload_unbuildable", tool=step.tool)
        return None

    # --- per-tool builders --------------------------------------------------

    def _blast(
        self, step: PlanStep, context: GapContext | None, relaxed: bool = False
    ) -> ToolInvocation | None:
        if context is None or not context.has_usable_flanks:
            return None

        arguments = _allowed(step.arguments, {"database", "program", "max_hits", "expect"})
        if relaxed:
            # An explicit argument from the planner still wins: it asked for
            # something specific, and overriding it would make the plan a lie.
            arguments.setdefault("expect", _RELAXED_EXPECT)
            arguments.setdefault("max_hits", _RELAXED_MAX_HITS)
            arguments.setdefault("database", _RELAXED_DATABASE)
            _log.info(
                "blast_retry_relaxed",
                gap_id=context.identifier,
                expect=arguments.get("expect"),
                database=arguments.get("database"),
            )

        return ToolInvocation(
            tool="blast_search",
            gap_id=context.identifier,
            relaxed=relaxed,
            payload=BlastSearchInput(
                sequence=context.query_sequence(),
                gap_id=context.identifier,
                # So the tool knows how far past the HSPs to reach when it
                # retrieves the subject region: the missing segment sits
                # between the flanks and is in no HSP.
                gap_length=context.gap.length,
                # And where the flanks meet, so each hit can be measured on
                # whether it carries bases across the gap at all.
                left_flank_length=len(context.left_flank),
                **arguments,
            ),
        )

    def _mafft(
        self,
        step: PlanStep,
        context: GapContext | None,
        state: ReconstructionState,
        relaxed: bool = False,
    ) -> ToolInvocation | None:
        if context is None:
            return None

        available = (state.get("references") or {}).get(context.identifier, [])

        # A retry admits the references the first pass filtered out as too
        # divergent. They are weaker evidence, but a second identical alignment
        # of the same rows cannot produce a different answer - the only thing
        # left to change is which rows go in.
        candidates = [
            reference
            for reference in (available if relaxed else self._ranker.filter_usable(available))
            # A reference with no residues cannot be aligned, however well it
            # scored on the BLAST metadata alone. A pseudogene can be aligned
            # and should not be: it diverges from the functional copy at
            # exactly the bases being reconstructed.
            if reference.has_sequence
            and _QUALITY.is_usable(reference, expected_length=context.gap.length)
        ]
        if not candidates:
            return None

        # Keep only the references that actually carry bases across the gap,
        # when any do. This is not a preference - it is what makes the
        # alignment work at all.
        #
        # MAFFT builds one multiple alignment over every row it is given. A
        # reference homologous to the flanks but stopping short of the gap has
        # no bases to place there, and it pulls the alignment towards the shape
        # with no inserted columns at the junction. The target row then has no
        # gap columns, the mapper reports `spans_gap=False`, and the run ends
        # "no reference contributed bases across the gap" - while the donors
        # sat in the very same input, carrying the correct fill.
        #
        # Measured, not guessed: on a 21-base gap in human mtDNA COI, one hit
        # covering a single flank at 0.979 identity ranked top and broke the
        # alignment for all seven rows. Dropping it recovered the true bases
        # exactly. High identity over half the query is precisely the profile
        # that both ranks well and carries nothing.
        #
        # Applied before the limit, not after: the limit keeps the best few
        # rows, and filtering afterwards would let a handful of non-carriers at
        # the top starve the aligner of the donors further down.
        carriers = [reference for reference in candidates if reference.carries_gap]
        if carriers:
            candidates = carriers
        elif any(reference.gap_bases is not None for reference in candidates):
            # Every hit was measured and none reaches across the gap. The
            # alignment is still worth running - it is what tells the critic
            # the evidence is absent rather than untried - but say so, because
            # this is the difference between "no homologues" and "homologues
            # that stop at the gap", and the two need different retries.
            _log.info(
                "no_reference_carries_gap",
                gap_id=context.identifier,
                references=len(candidates),
                detail="Every reference stops short of the gap; alignment cannot fill it.",
            )

        candidates = self._with_organism_affinity(candidates, state.get("organism"))
        usable = self._ranker.rank(candidates, limit=_MAX_ALIGNMENT_REFERENCES)

        if relaxed:
            _log.info(
                "mafft_retry_relaxed",
                gap_id=context.identifier,
                references=len(usable),
                detail="Including references below the usability floor.",
            )

        # The end of the evidence path, and the number the earlier counts exist
        # to explain: how many references actually reached the aligner.
        _log.info(
            "references_sent_to_mafft",
            gap_id=context.identifier,
            references_sent_to_mafft=len(usable),
            available=len(available),
            with_sequence=sum(1 for reference in available if reference.has_sequence),
            carrying_gap=sum(1 for reference in available if reference.carries_gap),
        )

        return ToolInvocation(
            tool="mafft_align",
            gap_id=context.identifier,
            relaxed=relaxed,
            payload=AlignmentInput(
                gap_id=context.identifier,
                target_sequence=context.query_sequence(),
                left_flank_length=len(context.left_flank),
                references=self._alignment_references(usable, context),
            ),
        )

    def _with_organism_affinity(
        self, references: list[Reference], target: str | None
    ) -> list[Reference]:
        """Fill in relatedness from the organism names when nothing measured it.

        `ReferenceRanker` weights relatedness at 0.3, but its only writer is
        `evolutionary_context`, which in practice returns no usable evidence -
        so the weight was multiplying zero for every reference and the ranking
        ran on identity and coverage alone. A cross-species hit that aligned
        marginally better than a conspecific one therefore won, and on a human
        mtDNA gap the consensus drifted onto the mammalian sequence: 19 of 21
        bases, reported at 0.82 confidence.

        A measured relatedness is a real phylogenetic distance and is never
        overwritten here; this only fills the gap where there is none. When the
        organism is unknown on either side, `organism_affinity` returns None and
        the reference is passed through untouched - the ranking is then exactly
        what it was.
        """
        if not target:
            return references

        adjusted: list[Reference] = []
        boosted = 0
        for reference in references:
            if reference.relatedness is not None:
                adjusted.append(reference)
                continue
            affinity = organism_affinity(target, reference.organism)
            if affinity is None:
                adjusted.append(reference)
                continue
            boosted += 1
            adjusted.append(replace(reference, relatedness=affinity))

        if boosted:
            _log.info(
                "organism_affinity_applied",
                target_organism=target,
                references_scored=boosted,
                total=len(references),
            )
        return adjusted

    def _alignment_references(
        self, usable: list[Reference], context: GapContext
    ) -> dict[str, str]:
        """Residues for each reference, cut to the region worth aligning.

        A BLAST-derived reference arrives already short - the tool fetches only
        the subject span of its hit - but an `ncbi_search` record is the whole
        published sequence, ~16,000 bases for a mitogenome. MAFFT is a polled
        job with roughly 75 seconds of slice to finish in, and a 400-base target
        against several whole genomes does not: it is killed on every slice the
        run is granted, which reads downstream as "no usable reference evidence"
        even though the evidence was there. Measured on one prompt: 8.9 s when
        the planner opened with BLAST, never finishing when it opened with
        `ncbi_search`.

        Only the region homologous to the flanks can say anything about what
        lies between them, so the discarded sequence costs no information - as
        long as the window is genuinely on that region. `locate_window` refuses
        to guess: a reference it cannot anchor is passed through whole and left
        slow rather than being cut at an arbitrary offset, which would align the
        target against unrelated sequence and produce a confident wrong fill.
        """
        references: dict[str, str] = {}
        trimmed = 0
        for reference in usable:
            residues = reference.residues or ""
            windowed, anchor = window_residues(
                residues,
                metadata=reference.metadata,
                left_flank=context.left_flank,
                right_flank=context.right_flank,
            )
            if anchor != "none":
                trimmed += 1
                _log.info(
                    "alignment_reference_windowed",
                    gap_id=context.identifier,
                    accession=reference.accession,
                    anchor=anchor,
                    original_length=len(residues),
                    windowed_length=len(windowed),
                )
            references[reference.accession] = windowed

        if trimmed:
            _log.info(
                "alignment_references_windowed",
                gap_id=context.identifier,
                windowed=trimmed,
                total=len(usable),
            )
        return references

    def _ncbi(
        self,
        step: PlanStep,
        context: GapContext | None,
        state: ReconstructionState,
        relaxed: bool = False,
    ) -> ToolInvocation:
        organisms = step.arguments.get("organisms") or state.get("requested_organisms") or []
        return ToolInvocation(
            tool="ncbi_search",
            gap_id=context.identifier if context else None,
            relaxed=relaxed,
            payload=NCBISearchInput(
                term=str(step.arguments.get("term") or state.get("organism") or ""),
                organisms=[str(organism) for organism in organisms],
                **_allowed(step.arguments, {"database", "limit", "fetch_sequences"}),
            ),
        )


    def _plausibility(
        self, step: PlanStep, context: GapContext | None, state: ReconstructionState
    ) -> ToolInvocation | None:
        """Ask Evo 2 to arbitrate between the fills the alignment left open.

        Only worth a call when there is genuinely something to arbitrate. With
        a single candidate the model has nothing to compare it against, and its
        opinion would either rubber-stamp the consensus or contradict it on no
        evidence - neither of which is a reason to spend a generation.

        Until this branch existed the step was built by nothing and silently
        dropped with `tool_payload_unbuildable`, so the tool was unreachable
        however often the planner selected it.
        """
        if context is None:
            return None

        candidates = (state.get("candidates") or {}).get(context.identifier) or []
        proposals = {
            f"candidate_{index}": candidate.sequence
            for index, candidate in enumerate(candidates)
            if candidate.sequence
        }
        if len(proposals) < 2:
            _log.info(
                "evo2_skipped",
                gap_id=context.identifier,
                candidates=len(proposals),
                detail="Nothing to arbitrate; a lone candidate needs no tie-break.",
            )
            return None

        if not context.left_flank:
            return None

        return ToolInvocation(
            tool="evo2_plausibility",
            gap_id=context.identifier,
            payload=PlausibilityInput(
                gap_id=context.identifier,
                left_flank=context.left_flank,
                candidates=proposals,
            ),
        )

    def _evo(self, step: PlanStep, state: ReconstructionState) -> ToolInvocation:
        organisms = {
            reference.organism
            for references in (state.get("references") or {}).values()
            for reference in references
            if reference.organism
        }
        return ToolInvocation(
            tool="evolutionary_context",
            payload=EvolutionaryContextInput(
                target_organism=state.get("organism") or "",
                candidate_organisms=sorted(organisms),
            ),
        )

    # --- helpers ------------------------------------------------------------

    @staticmethod
    def _context_for(gap_id: str | None, state: ReconstructionState) -> GapContext | None:
        if not gap_id:
            return None
        for context in state.get("gap_contexts") or []:
            if context.identifier == gap_id:
                return context
        return None

    @staticmethod
    def apply_relatedness(
        references: list[Reference], relatedness: dict[str, float]
    ) -> list[Reference]:
        """Attach relatedness scores so the ranker can weigh them.

        Returns new objects because `Reference` is frozen - the originals stay
        valid for anything already holding them.
        """
        return [
            Reference(
                accession=reference.accession,
                organism=reference.organism,
                description=reference.description,
                residues=reference.residues,
                identity=reference.identity,
                coverage=reference.coverage,
                e_value=reference.e_value,
                bit_score=reference.bit_score,
                relatedness=relatedness.get(reference.organism or "", reference.relatedness),
                source=reference.source,
                metadata=dict(reference.metadata),
            )
            for reference in references
        ]


def _allowed(arguments: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    """Keep only the argument keys a given tool schema accepts.

    The planner is a language model and will occasionally invent parameters;
    filtering here turns that into a no-op instead of a validation error.
    """
    return {key: value for key, value in arguments.items() if key in keys}
