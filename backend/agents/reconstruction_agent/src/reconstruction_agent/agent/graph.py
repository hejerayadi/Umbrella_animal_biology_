"""The reconstruction loop, as a LangGraph state machine.

    initialize -> plan -> reason -> guard -> act -> observe -> evaluate
                            ^                                     |
                            |                                     v
                            +--- replan <- critic <---------+  finalize

One graph run handles one gap. Multi-gap requests run the graph once per gap so
that a gap which stalls, fails or exhausts its replans cannot take the evidence
gathered for another gap down with it.

Node responsibilities are kept narrow on purpose:

- `plan`     decides the order of actions and nothing else.
- `reason`   turns the next action into typed arguments; touches no network.
- `guard`    is the only place that may refuse to dispatch. Every stop
             condition - deadline, budget, empty plan - is checked here, so
             there is one answer to "why did this run stop".
- `act`      dispatches exactly one tool.
- `observe`  merges the outcome into state. It is the only writer of evidence
             channels, which is what makes a lost merge a testable event.
- `evaluate` reads the evidence and returns a decision.
- `critic`   names the deficit; `replan` turns it into a new plan.
- `finalize` commits or refuses, always reachable, never skipped.
"""

from __future__ import annotations

from typing import Any, cast

from langgraph.graph import END, StateGraph

from reconstruction_agent.agent import router
from reconstruction_agent.agent.critic import Critic
from reconstruction_agent.agent.evaluator import Evaluator
from reconstruction_agent.agent.planner import Planner
from reconstruction_agent.agent.reasoner import MissingPrerequisite, build_arguments
from reconstruction_agent.agent.replanner import Replanner
from reconstruction_agent.agent.state import AgentContext, GapState
from reconstruction_agent.domain.enums import ToolName
from reconstruction_agent.domain.models.evidence import EvidenceBundle
from reconstruction_agent.observability.logger import get_logger
from reconstruction_agent.tools.base import ToolOutcome

_log = get_logger(__name__)

#: How many scientific replans one gap may attempt. Small: each costs a full
#: search round, and a deficit that survives two changes of strategy is a
#: property of the evidence rather than of the plan.
DEFAULT_MAX_REPLANS = 2


class ReconstructionGraph:
    """The compiled agent loop for one gap."""

    def __init__(
        self,
        context: AgentContext,
        *,
        planner: Planner,
        evaluator: Evaluator,
        critic: Critic,
        replanner: Replanner | None = None,
        max_replans: int = DEFAULT_MAX_REPLANS,
    ) -> None:
        self._context = context
        self._planner = planner
        self._evaluator = evaluator
        self._critic_service = critic
        self._replanner = replanner or Replanner()
        self._max_replans = max_replans
        self._compiled = self._build()

    async def run(self, state: GapState) -> GapState:
        """Execute the loop for one gap and return its final state."""
        seeded = dict(state)
        # Routing predicates read these off the mapping they are given, which
        # is what lets them be tested against a literal dict.
        seeded["deadline"] = self._context.deadline
        seeded["budget"] = self._context.budget
        seeded["max_replans"] = self._max_replans
        final: dict[str, Any] = await self._compiled.ainvoke(seeded)
        # Cast rather than reconstruct: LangGraph returns the channels it was
        # given, and re-validating them here would only re-state the TypedDict.
        return cast(GapState, {k: v for k, v in final.items() if k in GapState.__annotations__})

    # --- Nodes -------------------------------------------------------------

    async def _initialize(self, state: GapState) -> dict[str, Any]:
        _log.info(
            "gap_run_started",
            gap_id=state.get("gap_id"),
            request_id=state.get("request_id"),
            gap_length=state["gap"].length if state.get("gap") else 0,
        )
        return {"observations": ("run started",)}

    async def _plan(self, state: GapState) -> dict[str, Any]:
        gap = state.get("gap")
        plan = await self._planner.plan(
            gap_id=state.get("gap_id", ""),
            gap_length=gap.length if gap else 0,
            target=state.get("scientific_name") or "",
        )
        _log.info(
            "plan_selected",
            gap_id=state.get("gap_id"),
            actions=[action.value for action in plan.actions],
            model_authored=plan.model_authored,
            rationale=plan.rationale,
        )
        return {
            "plan": tuple(action.value for action in plan.actions),
            "observations": (f"plan: {plan.rationale}",),
        }

    async def _reason(self, state: GapState) -> dict[str, Any]:
        """A no-op on the happy path: argument building happens in `act`.

        The node exists because the loop re-enters here after a replan, and a
        named re-entry point is clearer than an edge back into `guard`.
        """
        return {}

    async def _guard(self, state: GapState) -> dict[str, Any]:
        """Stop conditions, all in one place."""
        if self._context.deadline.expired():
            return {"observations": ("the working window closed before the next action",)}
        if self._context.budget.exhausted:
            return {"observations": ("the tool budget was exhausted",)}
        return {}

    async def _act(self, state: GapState) -> dict[str, Any]:
        """Dispatch the head of the plan."""
        plan = state.get("plan", ())
        if not plan:
            return {}

        action = ToolName(plan[0])
        try:
            arguments = build_arguments(action, state, self._context)
        except MissingPrerequisite as error:
            # The plan and the state diverged. Drop the action and let the
            # evaluator decide from the evidence actually in hand.
            _log.info("action_skipped", gap_id=state.get("gap_id"), action=action.value)
            return {"plan": plan[1:], "errors": (str(error),)}

        outcome, record = await self._context.registry.invoke(
            action,
            arguments,
            budget=self._context.budget,
            gap_id=state.get("gap_id", ""),
        )
        _log.info(
            "tool_executed",
            gap_id=state.get("gap_id"),
            tool=action.value,
            ok=record.ok,
            seconds=record.duration_seconds,
            **record.summary,
        )
        return {
            "plan": plan[1:],
            "tool_history": (record,),
            "pending_outcome": outcome,
            "pending_action": action,
        }

    async def _observe(self, state: GapState) -> dict[str, Any]:
        """Merge one tool outcome into the evidence channels.

        The only writer of evidence. Concentrating it here is what makes a
        result computed and then lost during a merge a testable event rather
        than an invisible one.
        """
        outcome: ToolOutcome[Any] | None = state.get("pending_outcome")
        if outcome is None:
            return {}

        update: dict[str, Any] = {"pending_outcome": None, "pending_action": None}
        # Merged before the ok/failed branch below: a failed call has often
        # still measured something, and that measurement is what tells a
        # scoping problem from an absence.
        if not outcome.evidence.is_empty:
            update["evidence"] = EvidenceBundle().merged_with(outcome.evidence)
        if not outcome.ok and outcome.reason:
            update["observations"] = (f"{outcome.tool.value}: {outcome.reason}",)
        if outcome.transport_error and outcome.reason:
            update["errors"] = (outcome.reason,)

        data = outcome.data
        if data is None:
            return update

        # Written even when the call reported failure: a search that found hits
        # but none spanning the gap has still measured something the critic
        # needs in order to tell a scoping problem from an absence.
        for field, channel in _EVIDENCE_CHANNELS.items():
            value = getattr(data, field, None)
            if value is not None:
                update[channel] = value

        measured = getattr(data, "measured", None)
        if measured:
            update["measured_scopes"] = tuple(measured)

        # Arbitration's result, kept explicitly. A computed Evo 2 score that
        # never reaches `score_candidate` is a call spent on a number nobody
        # reads, and it looks exactly like a call that was never made.
        agreement = getattr(data, "agreement", None)
        if agreement:
            update["evo2_agreement"] = dict(agreement)

        # The model's own proposal, added alongside the homology candidates
        # rather than replacing them. It carries `origin=MODEL` and a capped
        # confidence, so it wins only where nothing observed competes - which
        # is the case it was generated for.
        generated = getattr(data, "generated_candidate", None)
        if generated is not None:
            existing = tuple(
                item
                for item in state.get("candidates", ())
                if item.candidate_id != generated.candidate_id
            )
            merged = (*existing, generated)
            update["candidates"] = tuple(sorted(merged, key=lambda item: -item.final_confidence))

        return update

    async def _evaluate(self, state: GapState) -> dict[str, Any]:
        decision = self._evaluator.evaluate(
            state, deadline=self._context.deadline, budget=self._context.budget
        )
        return {"decision": decision}

    async def _critic_node(self, state: GapState) -> dict[str, Any]:
        deficit = await self._critic_service.diagnose(state)
        _log.info(
            "deficit_diagnosed",
            gap_id=state.get("gap_id"),
            deficit=deficit.value if deficit else None,
        )
        return {"deficit": deficit}

    async def _replan(self, state: GapState) -> dict[str, Any]:
        deficit = state.get("deficit")
        if deficit is None:
            return {"plan": ()}

        strategy = self._replanner.plan_for(deficit, state)
        _log.info(
            "replan",
            gap_id=state.get("gap_id"),
            deficit=deficit.value,
            actionable=strategy.actionable,
            rationale=strategy.rationale,
            attempt=state.get("replan_count", 0) + 1,
        )
        if not strategy.actionable:
            return {"plan": (), "observations": (f"replan declined: {strategy.rationale}",)}

        merged = {**state.get("overrides", {})}
        for tool, values in strategy.overrides.items():
            merged[tool] = {**merged.get(tool, {}), **values}

        return {
            "plan": tuple(action.value for action in strategy.actions),
            "overrides": merged,
            "exhausted_scopes": state.get("exhausted_scopes", frozenset())
            | strategy.exhaust_scopes,
            "replan_count": state.get("replan_count", 0) + 1,
            "deficit": None,
            "observations": (f"replan: {strategy.rationale}",),
        }

    async def _finalize(self, state: GapState) -> dict[str, Any]:
        """Commit or refuse. Reachable from every state, including an empty one."""
        gap = state.get("gap")
        if gap is None:
            return {}

        arguments = build_arguments(ToolName.FINALIZE_RESULT, state, self._context)
        outcome, record = await self._context.registry.invoke(
            ToolName.FINALIZE_RESULT,
            arguments,
            budget=self._context.budget,
            gap_id=state.get("gap_id", ""),
        )
        result = getattr(outcome.data, "reconstruction", None)
        _log.info(
            "gap_run_finished",
            gap_id=state.get("gap_id"),
            status=result.status.value if result else "no result",
            replans=state.get("replan_count", 0),
        )
        return {"result": result, "tool_history": (record,)}

    # --- Wiring ------------------------------------------------------------

    def _build(self) -> Any:
        graph: StateGraph[Any, Any, Any] = StateGraph(GapState)
        graph.add_node("initialize", self._initialize)
        graph.add_node("plan", self._plan)
        graph.add_node("reason", self._reason)
        graph.add_node("guard", self._guard)
        graph.add_node("act", self._act)
        graph.add_node("observe", self._observe)
        graph.add_node("evaluate", self._evaluate)
        graph.add_node("critic", self._critic_node)
        graph.add_node("replan", self._replan)
        graph.add_node("finalize", self._finalize)

        graph.set_entry_point("initialize")
        graph.add_edge("initialize", "plan")
        graph.add_edge("plan", "reason")
        graph.add_edge("reason", "guard")
        graph.add_conditional_edges(
            "guard", router.after_guard, {router.ACT: "act", router.FINALIZE: "finalize"}
        )
        graph.add_edge("act", "observe")
        graph.add_edge("observe", "evaluate")
        graph.add_conditional_edges(
            "evaluate",
            _route_after_evaluate,
            {
                router.ACT: "guard",
                router.CRITIC: "critic",
                router.FINALIZE: "finalize",
            },
        )
        graph.add_conditional_edges(
            "critic",
            router.after_critic,
            {router.REPLAN: "replan", router.FINALIZE: "finalize"},
        )
        graph.add_conditional_edges(
            "replan",
            router.after_replan,
            {router.REASON: "reason", router.FINALIZE: "finalize"},
        )
        graph.add_edge("finalize", END)
        return graph.compile()


def _route_after_evaluate(state: dict[str, Any]) -> str:
    """Finish the plan before deciding anything about it.

    A plan with actions left has not yet produced the evidence it was written
    to produce. Criticising it now would diagnose a deficit the next action was
    already going to fix - and, worse, *accepting* it now abandons the actions
    still queued.

    That second case was a live bug and is the reason this predicate no longer
    looks at the decision at all. The evaluator declared a candidate ready as
    soon as it cleared the confidence floor, the run finalised, and the
    `evaluate_with_evo2` and `score_candidate` steps the planner had explicitly
    scheduled were silently dropped. A gap answered by homology lost nothing
    visible; a gap that needed the model would have lost its only answer.

    So the plan runs to completion, and the evaluator's verdict decides only
    what happens once there is nothing left to run. The guard still refuses
    individual dispatches on time and budget, which is what keeps this bounded.
    """
    plan = state.get("plan")
    if plan and plan[0] != router.FINALIZE_ACTION:
        return router.ACT
    return router.after_evaluate(state)


#: Which output field lands in which state channel. A table rather than a chain
#: of `isinstance` checks, so adding a tool means adding a row.
_EVIDENCE_CHANNELS: dict[str, str] = {
    "record": "record",
    "profile": "profile",
    "context": "context",
    "hits": "hits",
    "alignment": "alignment",
    "support": "support",
    "candidates": "candidates",
}
