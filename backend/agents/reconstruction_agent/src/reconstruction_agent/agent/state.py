"""What the graph carries between nodes.

Two rules govern everything in this module.

**Values are frozen models and plain data - never open clients.** A node
receives state, not a service. Clients, limiters and caches live in the
`AgentContext` handed to the graph at construction, which is deliberately *not*
part of the state: putting a live `httpx` client in a channel would make the
state unserialisable, unloggable and impossible to snapshot in a test.

**Reducers are declared, not implied.** Every accumulating channel carries an
`Annotated` reducer, so two nodes that both write `tool_history` append rather
than overwrite. The failure this prevents is specific and was worth a rule: a
result computed by one node and silently dropped during a merge looks exactly
like a result that was never computed, and no test catches it unless the merge
itself is tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from reconstruction_agent.domain.enums import CriticDeficit, EvaluationDecision
from reconstruction_agent.domain.models.alignment import Alignment, AlignmentSupport
from reconstruction_agent.domain.models.candidate import Candidate
from reconstruction_agent.domain.models.evidence import EvidenceBundle, merge_evidence
from reconstruction_agent.domain.models.homology import HomologHit, HomologySearchOutcome
from reconstruction_agent.domain.models.result import GapReconstruction
from reconstruction_agent.domain.models.sequence import Gap, GapContext, SequenceRecord
from reconstruction_agent.domain.models.taxonomy import TargetProfile
from reconstruction_agent.integrations.llm.base import LLMClient
from reconstruction_agent.orchestration.budget import BudgetLedger
from reconstruction_agent.orchestration.deadline import Deadline, PhaseBudget
from reconstruction_agent.tools.base import ToolRecord
from reconstruction_agent.tools.registry import ToolRegistry


def append[T](left: tuple[T, ...], right: tuple[T, ...]) -> tuple[T, ...]:
    """Accumulate. The reducer for history, observations and errors."""
    return tuple(left) + tuple(right)


def replace[T](left: T, right: T) -> T:
    """Last write wins. The reducer for a channel holding a current value.

    Declared explicitly even though it is LangGraph's default, because a
    channel whose reducer is obvious to its author is the one most likely to be
    given the wrong one by the next person to add a node.
    """
    return right


@dataclass(frozen=True)
class AgentContext:
    """Everything the nodes need that is not state.

    Held outside the state on purpose: these are live objects with sockets,
    locks and caches. `Deadline` and `BudgetLedger` are here rather than in
    state because both are mutable ledgers read and advanced by every node -
    copying them through a reducer would let two concurrent branches each see a
    budget the other had already spent.
    """

    registry: ToolRegistry
    llm: LLMClient
    deadline: Deadline
    budget: BudgetLedger
    phases: PhaseBudget


class GapState(TypedDict, total=False):
    """The graph's working memory for one gap.

    One gap per graph run. Multi-gap runs execute this graph once per gap, so a
    slow or failing gap cannot corrupt the evidence gathered for another.
    """

    # --- Identity, written once by `initialize` -----------------------------
    gap_id: str
    gap: Gap
    request_id: str
    accession: str | None
    residues: str | None
    scientific_name: str | None

    # --- Evidence, accumulated as tools return ------------------------------
    record: Annotated[SequenceRecord | None, replace]
    profile: Annotated[TargetProfile | None, replace]
    context: Annotated[GapContext | None, replace]
    hits: Annotated[tuple[HomologHit, ...], replace]
    measured_scopes: Annotated[tuple[HomologySearchOutcome, ...], append]
    alignment: Annotated[Alignment | None, replace]
    support: Annotated[AlignmentSupport | None, replace]
    candidates: Annotated[tuple[Candidate, ...], replace]
    #: Agreement per candidate id, as arbitration reported it. Held in its
    #: own channel rather than folded straight into the candidates: this is
    #: the hop where an Evo 2 result is most easily lost, and routing it
    #: through state is what makes the loss observable in a test.
    evo2_agreement: Annotated[dict[str, float], replace]
    #: Merged rather than replaced: a tool result must never overwrite what
    #: an earlier tool measured, which is how a run once reported an exact
    #: reconstruction beside zero gap-spanning hits and no provenance.
    evidence: Annotated[EvidenceBundle, merge_evidence]

    # --- The plan and what has been done about it ---------------------------
    #: Remaining actions, head first. Consumed by `act`, refilled by `replan`.
    plan: Annotated[tuple[str, ...], replace]
    #: Arguments overriding a tool's defaults, keyed by tool name. This is how
    #: the replanner widens flanks or relaxes an e-value without any node
    #: needing to know why.
    overrides: Annotated[dict[str, dict[str, Any]], replace]
    #: Scopes already measured as weak, never re-searched.
    exhausted_scopes: Annotated[frozenset[str], replace]

    tool_history: Annotated[tuple[ToolRecord, ...], append]
    observations: Annotated[tuple[str, ...], append]
    errors: Annotated[tuple[str, ...], append]

    # --- Control -----------------------------------------------------------
    decision: Annotated[EvaluationDecision | None, replace]
    deficit: Annotated[CriticDeficit | None, replace]
    #: Scientific replans only. Transport retries are counted by the retry
    #: policy inside the HTTP layer and never reach this number - conflating
    #: them would let a flaky network exhaust the agent's thinking budget.
    replan_count: Annotated[int, replace]
    result: Annotated[GapReconstruction | None, replace]

    # --- Handed to the routing predicates ----------------------------------
    #: The live ledgers. Present in state so `agent/router.py` can stay a set
    #: of pure functions over a mapping, testable against a literal dict
    #: without constructing a graph. Nothing mutates them through state.
    deadline: Deadline | None
    budget: BudgetLedger | None
    max_replans: int

    # --- Transient, one hop from `act` to `observe` -------------------------
    #: Declared because LangGraph rejects an update naming a channel the state
    #: does not have, and cleared by `observe` so a stale outcome can never be
    #: merged twice.
    pending_outcome: Annotated[Any | None, replace]
    pending_action: Annotated[Any | None, replace]


def initial_state(
    *,
    gap: Gap,
    request_id: str,
    accession: str | None,
    residues: str | None,
    scientific_name: str | None,
) -> GapState:
    """A state with every accumulating channel present and empty.

    Explicit rather than relying on `total=False`: a node reading a channel
    that was never written gets a `KeyError` at runtime, and the places that
    happens are exactly the paths a happy-path test does not take.
    """
    return GapState(
        gap_id=gap.gap_id,
        gap=gap,
        request_id=request_id,
        accession=accession,
        residues=residues,
        scientific_name=scientific_name,
        record=None,
        profile=None,
        context=None,
        hits=(),
        measured_scopes=(),
        alignment=None,
        support=None,
        candidates=(),
        evo2_agreement={},
        evidence=EvidenceBundle(),
        plan=(),
        overrides={},
        exhausted_scopes=frozenset(),
        tool_history=(),
        observations=(),
        errors=(),
        decision=None,
        deficit=None,
        replan_count=0,
        result=None,
        pending_outcome=None,
        pending_action=None,
    )
