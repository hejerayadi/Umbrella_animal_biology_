"""The graph's nodes: one function per step of the agent loop.

Each takes the whole state and returns only what it changed. Nodes hold no run
state of their own - `ReconstructionNodes` carries the collaborators, not run
data - so the same instance serves every concurrent run and every slice.

The loop is plan -> select -> act -> observe -> reason -> validate -> critique
-> decide, and the two steps that make it an agent rather than a pipeline are
`decide` (which can send it round again) and `observe` (which is what `decide`
reasons over).
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace
from typing import Any, cast

from agent.graph.conditions import has_work
from agent.planning.planner import Planner, PlanStep
from agent.planning.stop_policy import StopPolicy, StopReason
from agent.planning.tool_selector import ToolInvocation, ToolSelector
from agent.reasoning.critic import Critic, Verdict
from agent.reasoning.reasoner import Reasoner
from agent.reasoning.revision import diagnose
from agent.state import transitions
from agent.state.reducers import accumulate_references
from agent.state.state import (
    ReconstructionState,
    attempt_key,
    final_gap_outcomes,
    unreconstructed_gap_ids,
)
from configuration.logging import get_logger
from contracts.events import EventType
from contracts.observation import Observation, ObservationStatus
from contracts.output import ReconstructionStatus
from domain.models import Alignment, Reference
from domain.policies.budget_policy import BudgetPolicy, BudgetUsage
from domain.policies.validation_policy import ValidationPolicy
from domain.services import ContextExtractor, GapDetector, ReconstructionValidator
from observability.events import EventEmitter
from tools.contracts import ToolOutput
from tools.registry import ToolRegistry

_log = get_logger(__name__)

#: A tool that produced nothing usable is retried once with relaxed parameters
#: before being abandoned. Two attempts, not more: a third rarely differs and
#: every attempt is charged to the budget.
_MAX_ATTEMPTS_PER_TOOL = 2


@dataclass(slots=True)
class ReconstructionNodes:
    """The collaborators every node needs, wired once at startup."""

    detector: GapDetector
    extractor: ContextExtractor
    policy: ValidationPolicy
    planner: Planner
    selector: ToolSelector
    registry: ToolRegistry
    reasoner: Reasoner
    validator: ReconstructionValidator
    critic: Critic
    stop_policy: StopPolicy
    budgets: BudgetPolicy
    events: EventEmitter

    # --- 0. Resume or start -------------------------------------------------

    async def load_or_init(self, state: ReconstructionState) -> ReconstructionState:
        """Open a slice.

        The graph is re-entered on every CONTINUE retry with the checkpointed
        state already loaded, so this does not restore anything itself - it
        marks the new slice and restarts the wall clock, which is per-slice
        because the 120 s timeout it guards is per HTTP call.
        """
        slice_index = state.get("slice_index", 0)
        resumed = bool(state.get("gap_contexts"))

        if resumed:
            _log.info(
                "slice_resumed",
                slice_index=slice_index,
                resolved=len(state.get("reconstructions") or {}),
                tool_calls=state.get("budget_tool_calls", 0),
            )

        return ReconstructionState(
            slice_index=slice_index,
            # The wall clock is per-slice and lives in state, so every node
            # reads the same start point and it survives a checkpoint.
            slice_started_at=time.monotonic(),
            yielded=False,
            # A slice that inherited a stop flag from the previous one would
            # finish before doing any work.
            should_continue=True,
            stop_reason=None,
        )

    # --- 1. Find the work ---------------------------------------------------

    async def detect_gaps(self, state: ReconstructionState) -> ReconstructionState:
        """Locate the gaps and decide which are worth attempting.

        Skipped on resume: the gaps were found in the first slice and the
        sequence has not changed, so redoing it would only overwrite context
        the later slices have been building on.
        """
        if state.get("gap_contexts"):
            return ReconstructionState()

        target = state["target"]
        gaps = self.detector.detect(target)
        contexts = self.extractor.extract_all(target, gaps)

        update: dict[str, Any] = dict(transitions.with_gaps(contexts))
        skipped: dict[str, str] = {}

        for context in contexts:
            attempt, reason = self.policy.should_attempt(context)
            if not attempt and reason:
                skipped[context.identifier] = reason

        if skipped:
            update["skipped"] = skipped

        self.events.emit(
            EventType.GAPS_DETECTED,
            state["run_id"],
            f"Found {len(gaps)} gap(s); {len(skipped)} will not be attempted.",
            {"gap_count": len(gaps), "skipped": len(skipped)},
        )
        return cast(ReconstructionState, update)

    # --- 2. Decide what to do ----------------------------------------------

    async def plan(self, state: ReconstructionState) -> ReconstructionState:
        """Choose this iteration's tool calls."""
        steps = await self.planner.plan(state, self.registry.catalogue())

        self.events.emit(
            EventType.PLAN_CREATED,
            state["run_id"],
            f"Planned {len(steps)} step(s).",
            {"steps": [step.as_dict() for step in steps]},
        )
        update = dict(transitions.with_plan([step.as_dict() for step in steps]))
        update["budget_llm_tokens"] = state.get("budget_llm_tokens", 0) + self.planner.take_tokens()
        return cast(ReconstructionState, update)

    # --- 3. Turn the plan into runnable calls ------------------------------

    async def select_tools(self, state: ReconstructionState) -> ReconstructionState:
        """Resolve plan steps into concrete invocations, within budget.

        Its own node rather than a side effect of execution, so that what the
        agent chose to run is inspectable - and so a step refused for budget
        is recorded as a SKIPPED observation instead of silently vanishing.
        """
        usage = self._usage(state)
        invocations: list[dict[str, Any]] = []
        refusals: list[Observation] = []

        for raw in state.get("plan") or []:
            step = PlanStep(
                tool=raw["tool"],
                gap_id=raw.get("gap_id"),
                reason=raw.get("reason", ""),
                arguments=raw.get("arguments") or {},
            )

            allowed, kind = self.budgets.may_run(step.tool, usage)
            if not allowed:
                refusals.append(
                    Observation(
                        tool=step.tool,
                        gap_id=step.gap_id,
                        status=ObservationStatus.SKIPPED,
                        iteration=state.get("iteration", 0),
                        detail=f"Budget exhausted: {kind.value if kind else 'unknown'}.",
                    )
                )
                continue

            attempts = state.get("attempts") or {}
            if attempts.get(attempt_key(step.tool, step.gap_id), 0) >= _MAX_ATTEMPTS_PER_TOOL:
                refusals.append(
                    Observation(
                        tool=step.tool,
                        gap_id=step.gap_id,
                        status=ObservationStatus.SKIPPED,
                        iteration=state.get("iteration", 0),
                        detail="Already attempted twice; not retried again.",
                    )
                )
                continue

            invocation = self.selector.build(step, state)
            if invocation is None:
                continue

            invocations.append(
                {
                    "tool": invocation.tool,
                    "gap_id": invocation.gap_id,
                    "payload": invocation.payload,
                    "relaxed": invocation.relaxed,
                }
            )

        update: dict[str, Any] = {"pending_invocations": invocations}
        if refusals:
            update["observations"] = refusals
        return cast(ReconstructionState, update)

    # --- 4. Act -------------------------------------------------------------

    async def execute_tools(self, state: ReconstructionState) -> ReconstructionState:
        """Run the selected invocations, concurrently where independent.

        Steps for different gaps do not interact, so they are gathered rather
        than awaited in sequence - the difference is minutes when several gaps
        each need a BLAST job.
        """
        pending = state.get("pending_invocations") or []
        if not pending:
            return transitions.with_warning("The plan produced no runnable tool calls.")

        results = await asyncio.gather(
            *(
                self._run_one(
                    ToolInvocation(
                        tool=item["tool"],
                        payload=item["payload"],
                        gap_id=item["gap_id"],
                        # Carried through so the observation records that this
                        # was a widened retry, not a first attempt.
                        relaxed=bool(item.get("relaxed")),
                    ),
                    state,
                )
                for item in pending
            ),
            return_exceptions=True,
        )

        merged: dict[str, Any] = {
            "references": {},
            "alignments": {},
            "tool_calls": [],
            "errors": [],
            "observations": [],
            "attempts": {},
        }
        for result in results:
            if isinstance(result, BaseException):
                merged["errors"].append(str(result))
                continue
            for key, value in result.items():
                if key == "references":
                    # Two tools in the same round can answer for one gap -
                    # BLAST measuring homology, NCBI supplying residues. A
                    # plain overwrite kept whichever finished last.
                    merged[key] = accumulate_references(merged[key], value)
                elif key in ("alignments", "attempts"):
                    merged[key].update(value)
                elif key in ("tool_calls", "errors", "observations"):
                    merged[key].extend(value)

        merged["pending_invocations"] = []
        return cast(
            ReconstructionState,
            {key: value for key, value in merged.items() if value or key == "pending_invocations"},
        )

    async def _run_one(
        self, invocation: ToolInvocation, state: ReconstructionState
    ) -> dict[str, Any]:
        """Execute one tool and fold its output into a partial state."""
        run_id = state["run_id"]
        started = time.monotonic()
        key = attempt_key(invocation.tool, invocation.gap_id)
        attempt = (state.get("attempts") or {}).get(key, 0) + 1
        # The round this observation belongs to. `decide` increments the
        # counter at the end of the round, so during execution it still names
        # the round now running.
        iteration = state.get("iteration", 0)

        self.events.emit(
            EventType.TOOL_STARTED,
            run_id,
            f"Running {invocation.tool}.",
            {"tool": invocation.tool, "gap_id": invocation.gap_id, "attempt": attempt},
        )

        update: dict[str, Any] = {
            "attempts": {key: attempt},
            "tool_calls": [
                {"tool": invocation.tool, "gap_id": invocation.gap_id, "succeeded": True}
            ],
            "errors": [],
            "observations": [],
        }

        def failure(detail: str, diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
            """One failed attempt, recorded identically however it failed."""
            self.events.emit(
                EventType.TOOL_FAILED, run_id, detail, {"tool": invocation.tool}
            )
            update["tool_calls"] = [
                {"tool": invocation.tool, "gap_id": invocation.gap_id, "succeeded": False}
            ]
            update["errors"] = [f"{invocation.tool}: {detail}"]
            update["observations"] = [
                Observation(
                    tool=invocation.tool,
                    gap_id=invocation.gap_id,
                    status=ObservationStatus.FAILED,
                    attempt=attempt,
                    iteration=iteration,
                    relaxed=invocation.relaxed,
                    duration_seconds=round(time.monotonic() - started, 3),
                    detail=detail,
                    diagnostics=diagnostics or {},
                )
            ]
            return update

        def deadline_abort(remaining: float) -> dict[str, Any]:
            """The slice ran out of wall clock while this tool was still going.

            Recorded as its own kind of failure, not as a semantic one: the
            tool did not answer "nothing found", it never answered at all. So
            `attempts` is deliberately left untouched - the next slice must be
            free to make the same call again as a first attempt, rather than
            inheriting a relaxed retry it never earned.
            """
            detail = f"aborted after {remaining:.0f}s: the slice ran out of wall clock"
            self.events.emit(EventType.TOOL_FAILED, run_id, detail, {"tool": invocation.tool})
            return {
                "tool_calls": [
                    {"tool": invocation.tool, "gap_id": invocation.gap_id, "succeeded": False}
                ],
                "errors": [f"{invocation.tool}: {detail}"],
                "observations": [
                    Observation(
                        tool=invocation.tool,
                        gap_id=invocation.gap_id,
                        status=ObservationStatus.FAILED,
                        attempt=attempt,
                        iteration=iteration,
                        relaxed=invocation.relaxed,
                        duration_seconds=round(time.monotonic() - started, 3),
                        detail=detail,
                        diagnostics={"slice_deadline": True},
                    )
                ],
            }

        # Whatever is left of this slice is all this call may take. Zero means
        # the slice is already over, so the call is not started at all.
        remaining = self.budgets.remaining_seconds(self._usage(state))

        try:
            output = await asyncio.wait_for(
                self.registry.run(invocation.tool, invocation.payload), timeout=remaining
            )
        except TimeoutError:
            return deadline_abort(remaining)
        except Exception as error:  # noqa: BLE001 - recorded, never fatal
            return failure(str(error))

        # A tool that answers with the wrong type is a bug in that tool, and it
        # must not be indistinguishable from a legitimate empty result. Reading
        # it with `getattr(..., default)` would silently record "ran cleanly,
        # found nothing" for what is really a broken contract.
        if not isinstance(output, ToolOutput):
            return failure(
                f"returned {type(output).__name__}, which is not a ToolOutput",
                {
                    "contract_violation": "tool_output_type",
                    "returned_type": type(output).__name__,
                },
            )

        references: list[Reference] = list(getattr(output, "references", []) or [])
        alignment = getattr(output, "alignment", None)
        relatedness: dict[str, float] = dict(getattr(output, "relatedness", {}) or {})

        evidence_added = len(references)
        if isinstance(alignment, Alignment) and alignment.spans_gap:
            evidence_added += alignment.reference_count
        # Relatedness is evidence too - it changes how references rank, even
        # though it adds none. Counting only references would record a
        # successful `evolutionary_context` call as EMPTY.
        evidence_added += len(getattr(output, "relatedness", {}) or {})
        evidence_added += len(getattr(output, "scores", {}) or {})

        if not output.succeeded:
            return failure(str(output.error or "failed"), dict(output.diagnostics or {}))

        if evidence_added == 0:
            # Ran cleanly, found nothing. Not an error - a real answer about
            # this gap - but it must be visible so the critic can weigh it.
            status = ObservationStatus.EMPTY
            detail: str | None = "Completed without producing usable evidence."
        else:
            status = ObservationStatus.OK
            detail = None

        scores: dict[str, float] = dict(getattr(output, "scores", {}) or {})
        if scores and invocation.gap_id:
            # Evo 2's verdict on the fills the alignment left open. Recorded
            # per gap so `reason` can re-decide with it, and so a resumed slice
            # does not pay for the same generation twice.
            update["plausibility"] = {
                invocation.gap_id: {
                    "scores": scores,
                    "best": getattr(output, "best_candidate", None),
                    "model_confidence": getattr(output, "model_confidence", 0.0),
                }
            }

        if references and invocation.gap_id:
            # Which round and attempt produced this evidence. Once references
            # accumulate, a gap's pool mixes rounds, and "where did this come
            # from" stops being answerable from the run log alone.
            update["references"] = {
                invocation.gap_id: [
                    _with_provenance(reference, iteration=iteration, attempt=attempt)
                    for reference in references
                ]
            }
        if isinstance(alignment, Alignment) and invocation.gap_id:
            update["alignments"] = {invocation.gap_id: alignment}

        if relatedness:
            # Scores are applied to every gap's references, not just this
            # call's: relatedness is a property of the organism, and the same
            # organism can back several gaps. Without writing them back, the
            # planner would see unscored references again next round and ask
            # for the same answer forever.
            existing = state.get("references") or {}
            update["references"] = {
                gap_id: ToolSelector.apply_relatedness(gap_references, relatedness)
                for gap_id, gap_references in existing.items()
            }

        update["observations"] = [
            Observation(
                tool=invocation.tool,
                gap_id=invocation.gap_id,
                status=status,
                attempt=attempt,
                iteration=iteration,
                relaxed=invocation.relaxed,
                duration_seconds=round(time.monotonic() - started, 3),
                evidence_added=evidence_added,
                detail=detail,
                diagnostics=dict(output.diagnostics or {}),
            )
        ]

        self.events.emit(
            EventType.TOOL_COMPLETED,
            run_id,
            f"{invocation.tool} {status.value}.",
            {"tool": invocation.tool, "evidence_added": evidence_added},
        )
        return update

    # --- 5. Observe ---------------------------------------------------------

    async def observe(self, state: ReconstructionState) -> ReconstructionState:
        """Charge the tool calls made so far to the budget.

        The observations themselves were recorded by `execute_tools`; this is
        where they become spend. Recomputed from the full observation list
        rather than incremented, so a replayed slice cannot double-charge -
        which matters because the graph is re-entered on every CONTINUE.

        SKIPPED observations are excluded: a call refused for budget was never
        made and must not consume the budget that refused it.
        """
        charged = [
            observation
            for observation in (state.get("observations") or [])
            if observation.status is not ObservationStatus.SKIPPED
        ]

        by_name: dict[str, int] = {}
        for observation in charged:
            by_name[observation.tool] = by_name.get(observation.tool, 0) + 1

        return ReconstructionState(
            budget_tool_calls=len(charged),
            budget_tool_calls_by_name=by_name,
        )

    # --- 6. Make sense of it ------------------------------------------------

    async def reason(self, state: ReconstructionState) -> ReconstructionState:
        """Build and score candidates from every alignment gathered so far."""
        alignments = state.get("alignments") or {}
        references = state.get("references") or {}
        # Not `reconstructions`: a LOW_CONFIDENCE or UNRESOLVED entry there is
        # this round's best answer, not a final one, and skipping on its mere
        # presence is what made every REVISE unable to change an outcome.
        settled = set(final_gap_outcomes(state))

        candidates: dict[str, Any] = {}

        for context in state.get("gap_contexts") or []:
            gap_id = context.identifier
            if gap_id in settled or gap_id not in alignments:
                continue

            built = self.reasoner.build_candidates(
                context, alignments[gap_id], references.get(gap_id, [])
            )
            # Where the alignment left two fills open and Evo 2 has since
            # judged them, its verdict decides. This is the whole point of
            # carrying an alternative: the vote could not settle the column,
            # and an independent read of the sequence context can.
            arbitration = (state.get("plausibility") or {}).get(gap_id)
            if arbitration and len(built) > 1:
                built = self.reasoner.arbitrate(
                    context,
                    built,
                    alignments[gap_id],
                    references.get(gap_id, []),
                    scores=dict(arbitration.get("scores") or {}),
                )
                self.events.emit(
                    EventType.CANDIDATE_PROPOSED,
                    state["run_id"],
                    f"Evo 2 settled a contested reconstruction for {gap_id}.",
                    {
                        "gap_id": gap_id,
                        "scores": arbitration.get("scores"),
                        "chosen": built[0].sequence[:40] if built else None,
                    },
                )

            candidates[gap_id] = built

            self.events.emit(
                EventType.CANDIDATE_PROPOSED,
                state["run_id"],
                f"Proposed a reconstruction for {gap_id}.",
                {"gap_id": gap_id, "candidates": len(built)},
            )

        return ReconstructionState(candidates=candidates) if candidates else ReconstructionState()

    # --- 7. Validate --------------------------------------------------------

    async def validate(self, state: ReconstructionState) -> ReconstructionState:
        """Turn scored candidates into reportable outcomes.

        Its own node so a validation failure is a fact the critic can act on -
        it used to be buried inside `Reasoner.finalise`, where a rejected
        candidate silently became UNRESOLVED and the loop never learned why.
        """
        candidates = state.get("candidates") or {}
        # Not `reconstructions`: a LOW_CONFIDENCE or UNRESOLVED entry there is
        # this round's best answer, not a final one, and skipping on its mere
        # presence is what made every REVISE unable to change an outcome.
        settled = set(final_gap_outcomes(state))
        contexts = {context.identifier: context for context in state.get("gap_contexts") or []}

        reconstructions: dict[str, Any] = {}
        warnings: list[str] = []

        for gap_id, gap_candidates in candidates.items():
            if gap_id in settled:
                continue
            context = contexts.get(gap_id)
            if context is None:
                continue

            outcome = self.reasoner.finalise(context, gap_candidates)
            reconstructions[gap_id] = outcome

            if outcome.status is ReconstructionStatus.UNRESOLVED and gap_candidates:
                warnings.append(
                    f"{gap_id}: a candidate was produced but did not pass validation."
                )

        update: dict[str, Any] = {}
        if reconstructions:
            update["reconstructions"] = reconstructions
        if warnings:
            update["warnings"] = warnings
        return cast(ReconstructionState, update)

    # --- 8. Check our work --------------------------------------------------

    async def critique(self, state: ReconstructionState) -> ReconstructionState:
        """Review this iteration's reconstructions and record a verdict each."""
        contexts = {context.identifier: context for context in state.get("gap_contexts") or []}
        usage = self._usage(state)

        # Whether re-planning could still help: with budget or slices spent,
        # a REVISE would be a promise the loop cannot keep, so the critic is
        # told to abstain instead.
        more_possible = (
            self.budgets.exhausted(usage) is None
            and self.budgets.remaining_tool_calls(usage) > 0
            and state.get("iteration", 0) + 1 < state.get("max_iterations", 6)
        )

        notes: list[str] = []
        verdicts: dict[str, str] = {}
        revisions: dict[str, str] = {}

        for gap_id, reconstruction in (state.get("reconstructions") or {}).items():
            context = contexts.get(gap_id)
            if context is None:
                continue
            critique = await self.critic.review(
                context, reconstruction, more_evidence_possible=more_possible
            )
            verdicts[gap_id] = critique.verdict.value
            if critique.verdict is not Verdict.ACCEPT:
                notes.append(critique.as_note())
            if critique.verdict is Verdict.REVISE:
                # What the next plan should do differently. Derived from state
                # rather than from the critic's prose so it is available with
                # no LLM configured, and so it means the same thing every time.
                reason = diagnose(state, gap_id)
                if reason is not None:
                    revisions[gap_id] = reason.value

        if notes:
            self.events.emit(
                EventType.CRITIQUE_ISSUED,
                state["run_id"],
                f"{len(notes)} reconstruction(s) were not accepted.",
                {"critiques": notes},
            )

        update: dict[str, Any] = {"verdicts": verdicts} if verdicts else {}
        if notes:
            update["critiques"] = notes
        if revisions:
            update["revision_reasons"] = revisions
            self.events.emit(
                EventType.CRITIQUE_ISSUED,
                state["run_id"],
                "Diagnosed what to change on the next round.",
                {"revision_reasons": revisions},
            )
        return cast(ReconstructionState, update)

    # --- 9. Decide ----------------------------------------------------------

    async def decide(self, state: ReconstructionState) -> ReconstructionState:
        """ACCEPT / REVISE / ABSTAIN, folded together with budgets and time.

        The one place that answers "does the loop go round again?". Everything
        else only records facts; this weighs them.
        """
        iteration = state.get("iteration", 0) + 1
        usage = self._usage(state)

        self.events.emit(
            EventType.ITERATION_COMPLETED, state["run_id"], f"Iteration {iteration} complete."
        )

        update: dict[str, Any] = {
            "iteration": iteration,
            "budget_llm_tokens": state.get("budget_llm_tokens", 0) + self.critic.take_tokens(),
        }

        probe = cast(ReconstructionState, {**state, **update})
        should_stop, reason = self.stop_policy.should_stop(
            probe, budgets=self.budgets, usage=usage
        )

        if should_stop and reason is not None:
            update["should_continue"] = False
            update["stop_reason"] = reason.value
            if reason is StopReason.YIELDED:
                update["yielded"] = True
                update["slice_index"] = state.get("slice_index", 0) + 1
                _log.info(
                    "slice_yielded",
                    elapsed=round(usage.elapsed_seconds, 1),
                    next_slice=update["slice_index"],
                )
            else:
                _log.info("loop_stopped", reason=reason.value, iterations=iteration)

        return cast(ReconstructionState, update)

    # --- 10. Finalise -------------------------------------------------------

    async def finalize(self, state: ReconstructionState) -> ReconstructionState:
        """Last stop before the result is built.

        Records a stop reason when none of the branches above set one - a run
        that ended without saying why is the hardest kind to debug - and
        decides whether the run should escalate to the Evolution Agent.
        """
        update: dict[str, Any] = {}

        escalation = self._evolution_escalation(state)
        if escalation is not None:
            # Deliberately NOT suppressed on the last slice. NEEDS_AGENT is
            # routed by the orchestrator's capability resolver and does not
            # consume a CONTINUE retry - only status="continue" does
            # (worker_node.route_after_worker). Gating it on the slice budget
            # conflated two unrelated budgets and withheld the request for help
            # from exactly the runs that had exhausted everything else.
            target, prompt = escalation
            update["needs_agent"] = target
            update["prompt_to_target_agent"] = prompt
            update["stop_reason"] = StopReason.DELEGATED.value
            _log.info("escalating", target=target, slice_index=state.get("slice_index", 0))
            return cast(ReconstructionState, update)

        if not state.get("stop_reason"):
            # Reached only when the graph skipped `decide` entirely - the
            # gapless path, where detect_gaps routes straight here. Anything
            # that ran the loop has already recorded why it stopped.
            update["stop_reason"] = (
                StopReason.NOTHING_TO_DO.value
                if not has_work(state)
                else StopReason.ALL_RESOLVED.value
            )

        return cast(ReconstructionState, update)

    @staticmethod
    def _evolution_escalation(state: ReconstructionState) -> tuple[str, str] | None:
        """Whether phylogeny is the blocker, and what to ask for.

        `EvolutionaryContextTool` reports when its genus-level heuristic cannot
        separate candidates. Acting on that is what turns the note into the
        NEEDS_AGENT the orchestrator knows how to route.

        The blocked gaps are every gap this agent failed to reconstruct -
        including ones the critic has already abandoned. Two earlier versions
        of this were wrong in opposite directions: keying on
        `reconstructions` missed gaps that never produced a candidate at all,
        and keying on *open* gaps let the critic's abstain silently cancel the
        request for help on exactly the gaps that needed it.
        """
        recommended = any(
            observation.diagnostics.get("delegation_recommended")
            for observation in state.get("observations") or []
        )
        if not recommended:
            return None

        # Already asked on an earlier slice; the orchestrator either routed it
        # or could not. Asking again with nothing new is what
        # `escalation_signatures` force-fails as a loop.
        if state.get("needs_agent"):
            return None

        blocked = sorted(unreconstructed_gap_ids(state))
        if not blocked:
            return None

        organism = state.get("organism") or "the target organism"
        return (
            "Evolution",
            f"Rank these reference organisms by phylogenetic proximity to {organism}. "
            f"The reconstruction of {', '.join(blocked)} is blocked because a "
            "genus-level name heuristic cannot tell which relatives are informative.",
        )

    # --- helpers ------------------------------------------------------------

    def _usage(self, state: ReconstructionState) -> BudgetUsage:
        """The budget tally for this run, as a view over checkpointed state.

        Built fresh on every call: nodes are shared between concurrent runs, so
        holding the tally on the node would leak one run's spend into another.
        """
        return BudgetUsage(
            tool_calls=state.get("budget_tool_calls", 0),
            tool_calls_by_name=dict(state.get("budget_tool_calls_by_name") or {}),
            llm_tokens=state.get("budget_llm_tokens", 0),
            slice_started_at=state.get("slice_started_at"),
        )


def _with_provenance(reference: Reference, *, iteration: int, attempt: int) -> Reference:
    """`reference` tagged with the round and attempt that produced it."""
    return replace(reference, iteration=iteration, attempt=attempt)
