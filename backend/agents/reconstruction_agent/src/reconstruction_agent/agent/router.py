"""The graph's conditional edges, as pure functions over plain dicts.

Every predicate here takes a mapping and returns a string. Nothing imports
LangGraph, nothing touches a client, nothing awaits. That is the point: routing
is where an agent loops forever, finalises without evidence, or replans past
its budget, and those are the behaviours most worth testing directly. Expressed
this way each one is a table-driven unit test over a literal dict, with no
graph to compile and no services to fake.

The node functions in `graph.py` are the only callers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reconstruction_agent.domain.enums import EvaluationDecision

#: Destinations. Strings rather than an enum because LangGraph matches edge
#: names by value, and a mismatch between an enum member and a registered node
#: name is a silent routing bug.
ACT = "act"
CRITIC = "critic"
FINALIZE = "finalize"
REASON = "reason"
REPLAN = "replan"


#: The action the `finalize` node owns. Every plan ends with it, but `act`
#: must never dispatch it: the node runs it too, and a gap finalised twice
#: reports the second answer while the history records both.
FINALIZE_ACTION = "finalize_result"


def after_guard(state: Mapping[str, Any]) -> str:
    """Whether the next planned action may run at all.

    The guard is what stops a run from starting work it cannot finish. An empty
    plan means the reasoner has nothing left to do; an expired deadline or an
    exhausted budget means finalising now with the evidence in hand beats
    dispatching a call whose answer arrives after the response has been sent.
    """
    plan = state.get("plan")
    if not plan:
        return FINALIZE
    if plan[0] == FINALIZE_ACTION:
        return FINALIZE
    if _out_of_time(state) or _out_of_budget(state):
        return FINALIZE
    return ACT


def after_evaluate(state: Mapping[str, Any]) -> str:
    """Where an evaluated state goes next.

    `MORE_EVIDENCE_REQUIRED` is the only decision that keeps the loop open, and
    even then only while a replan is still affordable. Everything else - a
    ready candidate, a resolved gap, an exhausted budget, a failure - ends in
    finalisation, because each is a state the agent can honestly report.
    """
    decision = state.get("decision")

    if decision is EvaluationDecision.MORE_EVIDENCE_REQUIRED:
        if _out_of_time(state) or _out_of_budget(state):
            return FINALIZE
        if not _replans_left(state):
            return FINALIZE
        return CRITIC

    return FINALIZE


def after_critic(state: Mapping[str, Any]) -> str:
    """Whether a named deficit is worth acting on.

    A critic that names no deficit has nothing for the replanner to act on, and
    replanning anyway would re-run the strategy that just produced insufficient
    evidence. That is the loop this branch exists to prevent.
    """
    if state.get("deficit") is None:
        return FINALIZE
    return REPLAN


def after_replan(state: Mapping[str, Any]) -> str:
    """Whether the replan produced anything new to do."""
    if not state.get("plan"):
        return FINALIZE
    return REASON


def _out_of_time(state: Mapping[str, Any]) -> bool:
    """True once the working window has closed.

    Read from the deadline itself rather than from a flag in state: a flag is
    a snapshot of a clock that has kept moving, and the gap between the two is
    exactly where an over-deadline call gets dispatched.
    """
    deadline = state.get("deadline")
    return bool(deadline is not None and deadline.expired())


def _out_of_budget(state: Mapping[str, Any]) -> bool:
    budget = state.get("budget")
    return bool(budget is not None and budget.exhausted)


def _replans_left(state: Mapping[str, Any]) -> bool:
    limit = state.get("max_replans")
    if limit is None:
        return True
    return int(state.get("replan_count", 0)) < int(limit)
