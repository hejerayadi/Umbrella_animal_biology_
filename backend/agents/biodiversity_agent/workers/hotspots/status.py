"""M3's outcome vocabulary, mapped onto the platform's four statuses.

``Biodiversity Hotspots.pdf`` Table 14 defines six outcomes for this module, two
of which the platform contract does not have: **PARTIAL** (real work was produced
but a piece failed) and **NEEDS_CLARIFICATION** (the region could not be
resolved, so the module asks rather than guessing).

The platform's ``AgentStatus`` in ``schema.py`` has four members, and that file is
shared with the Global Orchestrator - which parses every agent's reply into its
own identically-named enum. Adding members there would mean the orchestrator
receiving a status string it cannot parse, so instead M3 keeps its richer
vocabulary here and narrows it at the boundary:

    M3 outcome              AgentResult.status     why
    ------------------      ------------------     -----------------------------
    COMPLETED               COMPLETED              nothing to translate
    PARTIAL                 COMPLETED              a grid and figures were
                                                   produced; the caller is told
                                                   what is missing in warnings
    NEEDS_CLARIFICATION     FAILED                 M3 cannot proceed; the
                                                   Responder explains honestly
                                                   and lists the known regions
    FAILED                  FAILED                 nothing to translate

Nothing is lost in the narrowing: the original outcome always travels in the
payload as ``status_detail``, so a caller that understands M3 can act on the
distinction while a caller that only knows the four platform statuses still gets
a correct one.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from ...schema import AgentResult, AgentStatus


class M3Outcome(str, Enum):
    """What M3 can conclude, per the design document's Table 14."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    NEEDS_CLARIFICATION = "needs_clarification"
    FAILED = "failed"


# The narrowing. PARTIAL counts as success because a grid plus zero clusters is a
# legitimate, useful answer; NEEDS_CLARIFICATION counts as failure because the
# module genuinely cannot answer as asked.
PLATFORM_STATUS: dict[M3Outcome, AgentStatus] = {
    M3Outcome.COMPLETED: AgentStatus.COMPLETED,
    M3Outcome.PARTIAL: AgentStatus.COMPLETED,
    M3Outcome.NEEDS_CLARIFICATION: AgentStatus.FAILED,
    M3Outcome.FAILED: AgentStatus.FAILED,
}


def is_success(outcome: M3Outcome) -> bool:
    return outcome in (M3Outcome.COMPLETED, M3Outcome.PARTIAL)


def build_result(outcome: M3Outcome, output: Any = None, **fields: Any
                 ) -> AgentResult:
    """An ``AgentResult`` carrying a platform status and the M3 detail.

    ``status_detail`` is attached to the payload: as a key when it is a dict, as
    an attribute when it is one of M3's dataclasses (they declare the field), and
    by wrapping otherwise. A caller never has to guess which outcome produced a
    COMPLETED.
    """

    detail = outcome.value

    if isinstance(output, dict):
        output = {**output, "status_detail": detail}
    elif output is not None and hasattr(output, "status_detail"):
        output.status_detail = detail
    elif output is not None and not isinstance(output, (str, int, float)):
        # An unknown object: leave it alone rather than mangling it.
        pass

    return AgentResult(status=PLATFORM_STATUS[outcome], output=output, **fields)


def outcome_of(result: AgentResult) -> M3Outcome:
    """Recover M3's outcome from a result, for a caller that wants the detail."""

    payload = result.output
    detail = None
    if isinstance(payload, dict):
        detail = payload.get("status_detail")
    elif payload is not None:
        detail = getattr(payload, "status_detail", None)

    if detail:
        try:
            return M3Outcome(detail)
        except ValueError:
            pass

    return (M3Outcome.COMPLETED if result.status is AgentStatus.COMPLETED
            else M3Outcome.FAILED)
