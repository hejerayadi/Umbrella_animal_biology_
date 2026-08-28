"""Turning the next planned action into a concrete, typed call.

This is where "the plan says search for homologues" becomes "call
`search_homologs` with these flanks, this profile, this time budget and these
scopes already excluded". It is deliberately deterministic and model-free.

The reason is worth stating plainly: arguments decide *what gets measured*. A
model that chose an e-value, a scope or a hit count would be choosing the
evidence, and the whole design rests on the evidence being gathered by rule and
only the *order* of gathering being open to judgement. The model plans; this
module executes.

Two things do vary, and both are measured rather than guessed: the replanner's
overrides, and the phase budget left on the clock.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from reconstruction_agent.agent.state import AgentContext, GapState
from reconstruction_agent.domain.enums import ToolName, UnresolvedReason
from reconstruction_agent.domain.models.evidence import EvidenceBundle
from reconstruction_agent.tools.alignment.align_homologs import AlignHomologsInput
from reconstruction_agent.tools.alignment.analyze_alignment import AnalyzeAlignmentInput
from reconstruction_agent.tools.candidate.generate_candidates import GenerateCandidatesInput
from reconstruction_agent.tools.candidate.score_candidate import ScoreCandidateInput
from reconstruction_agent.tools.finalize.finalize_result import FinalizeResultInput
from reconstruction_agent.tools.homology.get_homolog_sequences import HomologSequencesInput
from reconstruction_agent.tools.homology.search_homologs import SearchHomologsInput
from reconstruction_agent.tools.plausibility.evaluate_with_evo2 import Evo2Input
from reconstruction_agent.tools.reconstruction.reconstruct_gap import ReconstructGapInput
from reconstruction_agent.tools.sequence.get_assembly_metadata import AssemblyMetadataInput
from reconstruction_agent.tools.sequence.get_sequence_context import SequenceContextInput
from reconstruction_agent.tools.validation.validate_candidate import ValidateCandidateInput


class MissingPrerequisite(Exception):
    """The plan reached an action whose input was never produced.

    Raised rather than returned because it means the plan and the state have
    diverged, which is a bug in planning rather than a scientific outcome. The
    graph converts it into a failed observation so the run still finalises.
    """


def build_arguments(action: ToolName, state: GapState, context: AgentContext) -> BaseModel:
    """The typed request for `action`, read off the evidence gathered so far."""
    overrides = state.get("overrides", {}).get(action.value, {})
    builder = _BUILDERS.get(action)
    if builder is None:
        raise MissingPrerequisite(f"{action.value} has no argument builder.")
    built: BaseModel = builder(state, context, overrides)
    return built


def _require[T](value: T | None, action: ToolName, what: str) -> T:
    if value is None:
        raise MissingPrerequisite(f"{action.value} needs {what}, which no earlier action produced.")
    return value


def _sequence_context(
    state: GapState, context: AgentContext, overrides: dict[str, Any]
) -> BaseModel:
    gap = _require(state.get("gap"), ToolName.GET_SEQUENCE_CONTEXT, "the gap coordinates")
    return SequenceContextInput(
        gap_id=state["gap_id"],
        gap=gap,
        accession=state.get("accession"),
        residues=state.get("residues"),
        **overrides,
    )


def _assembly_metadata(
    state: GapState, context: AgentContext, overrides: dict[str, Any]
) -> BaseModel:
    record = _require(state.get("record"), ToolName.GET_ASSEMBLY_METADATA, "the target record")
    return AssemblyMetadataInput(
        record=record, scientific_name=state.get("scientific_name"), **overrides
    )


def _search_homologs(
    state: GapState, context: AgentContext, overrides: dict[str, Any]
) -> BaseModel:
    gap_context = _require(state.get("context"), ToolName.SEARCH_HOMOLOGS, "the gap's flanks")
    profile = _require(state.get("profile"), ToolName.SEARCH_HOMOLOGS, "the target profile")
    return SearchHomologsInput(
        gap_id=state["gap_id"],
        context=gap_context,
        profile=profile,
        deadline=context.deadline,
        budget=context.budget,
        # The homology phase's share of the clock, not the whole window: a
        # search allowed to consume everything leaves hits with no time to be
        # fetched or aligned.
        time_budget=context.phases.for_homology(context.deadline),
        exclude_scopes=state.get("exhausted_scopes", frozenset()),
        **overrides,
    )


def _homolog_sequences(
    state: GapState, context: AgentContext, overrides: dict[str, Any]
) -> BaseModel:
    hits = state.get("hits", ())
    if not hits:
        raise MissingPrerequisite("get_homolog_sequences needs hits, and the search found none.")
    return HomologSequencesInput(
        gap_id=state["gap_id"], hits=hits, deadline=context.deadline, **overrides
    )


def _align_homologs(state: GapState, context: AgentContext, overrides: dict[str, Any]) -> BaseModel:
    gap_context = _require(state.get("context"), ToolName.ALIGN_HOMOLOGS, "the gap's flanks")
    hits = state.get("hits", ())
    if not hits:
        raise MissingPrerequisite(
            "align_homologs needs homologue sequences, and none were fetched."
        )
    return AlignHomologsInput(
        gap_id=state["gap_id"],
        context=gap_context,
        hits=hits,
        deadline=context.deadline,
        time_budget=context.phases.for_alignment(context.deadline),
        **overrides,
    )


def _analyze_alignment(
    state: GapState, context: AgentContext, overrides: dict[str, Any]
) -> BaseModel:
    alignment = _require(state.get("alignment"), ToolName.ANALYZE_ALIGNMENT, "an alignment")
    gap_context = _require(state.get("context"), ToolName.ANALYZE_ALIGNMENT, "the gap's flanks")
    return AnalyzeAlignmentInput(
        gap_id=state["gap_id"],
        alignment=alignment,
        context=gap_context,
        hits=state.get("hits", ()),
        **overrides,
    )


def _generate_candidates(
    state: GapState, context: AgentContext, overrides: dict[str, Any]
) -> BaseModel:
    support = _require(state.get("support"), ToolName.GENERATE_CANDIDATES, "the alignment analysis")
    gap_context = _require(state.get("context"), ToolName.GENERATE_CANDIDATES, "the gap's flanks")
    profile = _require(state.get("profile"), ToolName.GENERATE_CANDIDATES, "the target profile")
    return GenerateCandidatesInput(
        gap_id=state["gap_id"],
        support=support,
        context=gap_context,
        profile=profile,
        hits=state.get("hits", ()),
        **overrides,
    )


def _finalize(state: GapState, context: AgentContext, overrides: dict[str, Any]) -> BaseModel:
    # Finalisation deliberately has no prerequisite beyond the gap itself. It
    # must be reachable from every state, including one where nothing was
    # measured, because refusing with a stated reason is a result.
    gap = _require(state.get("gap"), ToolName.FINALIZE_RESULT, "the gap coordinates")
    hint, explanation = _why_nothing_was_produced(state)
    # The audit trail, assembled from what each tool reported measuring rather
    # than re-derived here. A returned sequence that cannot say where it came
    # from is indistinguishable from a fabricated one.
    bundle = state.get("evidence") or EvidenceBundle()
    return FinalizeResultInput(
        gap_id=state["gap_id"],
        gap=gap,
        candidates=state.get("candidates", ()),
        evidence=bundle.to_gap_evidence(),
        provenance=bundle.to_provenance(tool_calls=context.budget.tool_calls),
        unresolved_hint=hint,
        hint_explanation=explanation,
        **overrides,
    )


def _why_nothing_was_produced(state: GapState) -> tuple[UnresolvedReason | None, str]:
    """The honest reason for an empty result, when the run already knows it.

    A search that timed out and a search that returned nothing look identical
    from the candidate list, and calling the first one an absence of homologues
    is a claim about biology the run never measured.
    """
    scopes = state.get("measured_scopes", ())
    if scopes and all(item.error for item in scopes):
        return (
            UnresolvedReason.DEADLINE_EXCEEDED,
            "No homology search completed within the run deadline, so no evidence "
            "was gathered for this region.",
        )
    return None, ""


def _evaluate_with_evo2(
    state: GapState, context: AgentContext, overrides: dict[str, Any]
) -> BaseModel:
    gap = _require(state.get("gap"), ToolName.EVALUATE_WITH_EVO2, "the gap coordinates")
    gap_context = _require(state.get("context"), ToolName.EVALUATE_WITH_EVO2, "the gap's flanks")
    # Deliberately no candidate prerequisite. A gap no homologue spans is the
    # case this tool exists for: requiring candidates would skip it exactly
    # where it is the only remaining source of an answer. Observed doing so on
    # NW_007907101 - the search returned 150 hits, none spanning, and the model
    # was never asked.
    candidates = state.get("candidates", ())
    return Evo2Input(
        gap_id=state["gap_id"],
        candidates=candidates,
        context=gap_context,
        left_flank=gap_context.left_flank,
        gap_length=gap.length,
        **overrides,
    )


def _score_candidate(
    state: GapState, context: AgentContext, overrides: dict[str, Any]
) -> BaseModel:
    candidates = state.get("candidates", ())
    if not candidates:
        raise MissingPrerequisite("score_candidate needs candidates, and none were generated.")
    return ScoreCandidateInput(
        gap_id=state["gap_id"],
        candidates=candidates,
        # Read back from state rather than passed along from the arbitration
        # call: this is the hop where an Evo 2 result is most easily lost, and
        # routing it through state is what makes the loss testable.
        evo2_agreement=state.get("evo2_agreement", {}),
        **overrides,
    )


def _validate_candidate(
    state: GapState, context: AgentContext, overrides: dict[str, Any]
) -> BaseModel:
    gap_context = _require(state.get("context"), ToolName.VALIDATE_CANDIDATE, "the gap's flanks")
    candidates = state.get("candidates", ())
    if not candidates:
        raise MissingPrerequisite("validate_candidate needs candidates, and none exist.")
    return ValidateCandidateInput(
        gap_id=state["gap_id"], candidates=candidates, context=gap_context, **overrides
    )


def _reconstruct_gap(
    state: GapState, context: AgentContext, overrides: dict[str, Any]
) -> BaseModel:
    gap = _require(state.get("gap"), ToolName.RECONSTRUCT_GAP, "the gap coordinates")
    record = _require(state.get("record"), ToolName.RECONSTRUCT_GAP, "the target record")
    candidates = state.get("candidates", ())
    if not candidates:
        raise MissingPrerequisite("reconstruct_gap needs a candidate to write into the record.")
    return ReconstructGapInput(
        gap_id=state["gap_id"],
        gap=gap,
        candidate=candidates[0],
        residues=record.residues,
        **overrides,
    )


_BUILDERS: dict[ToolName, Callable[[GapState, AgentContext, dict[str, Any]], BaseModel]] = {
    ToolName.GET_SEQUENCE_CONTEXT: _sequence_context,
    ToolName.GET_ASSEMBLY_METADATA: _assembly_metadata,
    ToolName.SEARCH_HOMOLOGS: _search_homologs,
    ToolName.GET_HOMOLOG_SEQUENCES: _homolog_sequences,
    ToolName.ALIGN_HOMOLOGS: _align_homologs,
    ToolName.ANALYZE_ALIGNMENT: _analyze_alignment,
    ToolName.GENERATE_CANDIDATES: _generate_candidates,
    ToolName.EVALUATE_WITH_EVO2: _evaluate_with_evo2,
    ToolName.SCORE_CANDIDATE: _score_candidate,
    ToolName.VALIDATE_CANDIDATE: _validate_candidate,
    ToolName.RECONSTRUCT_GAP: _reconstruct_gap,
    ToolName.FINALIZE_RESULT: _finalize,
}
