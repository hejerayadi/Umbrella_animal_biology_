"""Deciding the order of actions for one gap.

The deterministic plan is the specification, not the fallback. Evidence
gathering for a gap has a fixed dependency order - you cannot align sequences
you have not fetched - so there is exactly one sensible sequence, and it is
written here in Python.

What a model adds is the ability to *shorten* it: skipping taxonomy resolution
when the organism is already known, or stopping early when the evidence in hand
already answers the question. So the model's plan is accepted only after being
validated against the same dependency rules, and any plan that fails validation
is replaced by the deterministic one rather than repaired. A half-valid plan
executed hopefully is worse than a correct plan executed plainly.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from reconstruction_agent.agent.prompts.loader import load_prompt
from reconstruction_agent.domain.enums import ToolName
from reconstruction_agent.integrations.llm.base import LLMClient
from reconstruction_agent.observability.logger import get_logger

_log = get_logger(__name__)

#: The dependency order. Every plan is a subsequence of this, and validation is
#: exactly that check - which is why the order is data rather than control flow.
CANONICAL_PLAN: tuple[ToolName, ...] = (
    ToolName.GET_SEQUENCE_CONTEXT,
    ToolName.GET_ASSEMBLY_METADATA,
    ToolName.SEARCH_HOMOLOGS,
    ToolName.GET_HOMOLOG_SEQUENCES,
    ToolName.ALIGN_HOMOLOGS,
    ToolName.ANALYZE_ALIGNMENT,
    ToolName.GENERATE_CANDIDATES,
    # Arbitration and rescoring sit here rather than earlier because both act
    # on candidates. Neither is in the default plan: `evaluate_with_evo2`
    # declines unless there is a tie, so planning it every time would cost a
    # tool call to be told there was nothing to break.
    ToolName.EVALUATE_WITH_EVO2,
    ToolName.SCORE_CANDIDATE,
    ToolName.VALIDATE_CANDIDATE,
    ToolName.RECONSTRUCT_GAP,
    ToolName.FINALIZE_RESULT,
)

#: What a run does when nothing has gone wrong yet. A subsequence of the
#: canonical order: revalidation and writing the fill back are replan responses
#: to a measured deficit, while Evo 2 and the rescoring that reads it are part
#: of every run - the model is the only source of an answer for a gap homology
#: cannot span.
DEFAULT_PLAN: tuple[ToolName, ...] = (
    ToolName.GET_SEQUENCE_CONTEXT,
    ToolName.GET_ASSEMBLY_METADATA,
    ToolName.SEARCH_HOMOLOGS,
    ToolName.GET_HOMOLOG_SEQUENCES,
    ToolName.ALIGN_HOMOLOGS,
    ToolName.ANALYZE_ALIGNMENT,
    ToolName.GENERATE_CANDIDATES,
    # Always planned, cheap when it declines. Two reasons it earns a slot in
    # every run rather than only in a replan: a gap no homologue spans has no
    # other source of an answer, and a gap that several homologues disagree
    # about is settled here rather than one iteration later. The tool itself
    # decides which of those applies and returns immediately when neither does.
    ToolName.EVALUATE_WITH_EVO2,
    ToolName.SCORE_CANDIDATE,
    ToolName.FINALIZE_RESULT,
)

#: Actions no plan may omit, whoever wrote it.
#:
#: Without the first there is no query. Without the last a gap is never
#: committed to or refused, and the run reports nothing.
#:
#: Evo 2 and the rescoring that reads it are required for a different reason: a
#: gap no homologue spans has no other source of an answer, and a model-authored
#: plan that drops them - which is legal as a subsequence, and was observed
#: happening - silently removes the only way that gap ever gets filled. The tool
#: returns immediately when it has nothing to add, so there is no plan in which
#: omitting it is an improvement.
REQUIRED: frozenset[ToolName] = frozenset(
    {
        ToolName.GET_SEQUENCE_CONTEXT,
        ToolName.EVALUATE_WITH_EVO2,
        ToolName.SCORE_CANDIDATE,
        ToolName.FINALIZE_RESULT,
    }
)


class ProposedPlan(BaseModel):
    """The model's answer, before validation."""

    actions: list[str] = Field(default_factory=list)
    rationale: str = ""


class Plan(BaseModel):
    """An ordered, validated sequence of actions for one gap."""

    actions: tuple[ToolName, ...]
    rationale: str = ""
    #: True when the actions came from a model rather than the canonical order.
    model_authored: bool = False


class Planner:
    """Produces the action sequence for one gap."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def plan(
        self, *, gap_id: str, gap_length: int, target: str = "", known: str = ""
    ) -> Plan:
        """The plan to execute, model-authored when one is valid."""
        if not self._llm.available:
            return self._canonical("No model configured; using the standard evidence order.")

        prompt = load_prompt("planner")
        proposed = await self._llm.structured(
            system=prompt.system,
            user=prompt.render_user(
                gap_id=gap_id, gap_length=gap_length, target=target, known=known
            ),
            schema=ProposedPlan,
        )
        if proposed is None:
            return self._canonical("The model returned no usable plan; using the standard order.")

        actions = validate(proposed.actions)
        if actions is None:
            _log.info("plan_rejected", gap_id=gap_id, proposed=proposed.actions)
            return self._canonical(
                "The proposed plan violated the action dependency order; using the standard order."
            )

        return Plan(actions=actions, rationale=proposed.rationale, model_authored=True)

    def _canonical(self, rationale: str) -> Plan:
        return Plan(actions=DEFAULT_PLAN, rationale=rationale)


def validate(names: list[str]) -> tuple[ToolName, ...] | None:
    """The plan as tool names, or None if it is not executable.

    A plan is executable when it is a subsequence of the canonical order, has no
    repeats, and includes the two actions no run can omit. Anything else would
    dispatch a tool whose input does not exist yet.
    """
    try:
        actions = tuple(ToolName(name) for name in names)
    except ValueError:
        return None

    if not actions or len(set(actions)) != len(actions):
        return None
    if not REQUIRED.issubset(actions):
        return None
    if not _is_subsequence(actions, CANONICAL_PLAN):
        return None
    return actions


def _is_subsequence(actions: tuple[ToolName, ...], order: tuple[ToolName, ...]) -> bool:
    """Whether `actions` appears in `order`, in order, possibly with gaps."""
    remaining = iter(order)
    return all(action in remaining for action in actions)
