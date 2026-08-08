"""State carried between LangGraph nodes.

One dataclass, passed node to node, each node returning only the fields it
owns. LangGraph merges those updates - which is why nodes return dicts rather
than mutating: an update that is not returned did not happen, and that makes
each node testable in isolation.

`normalized` holds the image bytes and the shared context, and both are excluded
from that model's repr and serialization. This state object is never logged,
never serialized and never returned - only the values the finalize node
explicitly copies out of it ever leave the workflow. No checkpointer is
attached, so nothing here is persisted between requests either.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..domain.models import (
    NormalizedRecognitionInput,
    RecognitionDecision,
    RetrievedReference,
    SpeciesCandidate,
    TextEvidence,
)

# Which shared-context key each helper agent writes when it completes. Read on
# resume: if the key is present, the dependency has already run and this agent
# must finish rather than escalate again.
#
# Taken from the agents' own card.json `output` blocks - these are the live
# cross-agent contract, not names chosen here.
HELPER_OUTPUT_KEYS: dict[str, str] = {
    "Evolution": "evolution_analysis",
    "Genome": "genome",
    "Biodiversity": "biodiversity_report",
    "Trait": "traits",
    "Literature": "papers",
    "Protein": "protein_structure",
}


@dataclass
class RecognitionState:
    """Everything the workflow knows about one request, as it learns it."""

    # -- the request, as handed to the graph --
    instruction: Any = None
    context: Any = None

    # -- validation --
    normalized: NormalizedRecognitionInput | None = None

    # -- text analysis and planning --
    text_evidence: TextEvidence | None = None
    resolved_hint_species_id: str | None = None
    plan: Any = None
    plan_source: str = "deterministic"
    plan_rejected: bool = False

    # What the reasoning model actually did on THIS request. Reported verbatim
    # in provenance; `used` is True only when a result was genuinely accepted.
    llm_plan_calls: int = 0
    llm_explain_calls: int = 0
    reasoning_llm_calls: int = 0
    reasoning_llm_used: bool = False
    explanation_source: str = "deterministic"

    # -- embedding --
    query_vector: list[float] | None = None

    # -- retrieval --
    references: list[RetrievedReference] = field(default_factory=list)
    retrieval_provider: str | None = None
    retrieval_mode: str | None = None
    retrieval_collection: str | None = None
    retrieval_dataset_version: str | None = None
    rejected_payloads: int = 0

    # -- aggregation, taxonomy, fusion, confidence --
    candidates: list[SpeciesCandidate] = field(default_factory=list)
    margin: float | None = None
    taxonomy_degraded: bool = False
    taxonomy_report: dict[str, Any] = field(default_factory=dict)
    visual_evidence_sufficient: bool = True
    decision: RecognitionDecision | None = None

    # -- delegation --
    delegate_to: str | None = None
    delegation_prompt: str | None = None

    # -- controlled failure --
    error_code: str | None = None
    error_message: str | None = None

    # Non-fatal notes for the response: a degraded taxonomy, dropped payloads.
    # Never contains request data.
    warnings: list[str] = field(default_factory=list)

    # Config values the finalize node needs for provenance. Snapshotted into the
    # state as plain data so the state carries no provider references and stays
    # serializable - LangGraph only keeps declared fields, so this must be one.
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    # -- the single terminal product --
    agent_result: Any = None

    @property
    def shared_context(self) -> dict[str, Any]:
        """The caller's context, minus the image. Empty before validation runs."""
        return self.normalized.context if self.normalized else {}
