from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScheduleDecision:
    """The scheduler's decision about which agent, if any, should run next."""

    next_agent: str | None


class ExecutionScheduler:
    """Translate an agent result into the next orchestration action.

    The scheduler does not decide *what* a result means; it only interprets
    the status emitted by an agent and selects the next step using the planner
    or capability resolver when needed.
    """

    def schedule(
        self,
        result,
        planner,
        capability_resolver,
        state,
    ) -> ScheduleDecision:

        # The workflow currently receives statuses as plain strings, but this
        # normalization also accepts enum-like objects that expose `.name`.
        status = getattr(result, "status", None)
        status_name = getattr(status, "name", status)

        # A completed task ends the workflow immediately.
        if status_name == "COMPLETED":
            return ScheduleDecision(next_agent=None)

        # Some agents ask for a specific capability instead of naming the next
        # agent directly. In that case we delegate to the capability resolver.
        if status_name == "NEEDS_CAPABILITY":

            requested_capability = getattr(result, "requested_capability", None)

            agent = capability_resolver.resolve(
                requested_capability
            )

            return ScheduleDecision(next_agent=agent)

        # CONTINUE means the current agent wants the planner to pick the next
        # step in the predefined workflow chain.
        if status_name == "CONTINUE":

            agent = planner.next_agent(state)

            return ScheduleDecision(next_agent=agent)

        # Unknown or unexpected statuses are treated conservatively as terminal
        # so the orchestrator does not continue on a bad assumption.
        return ScheduleDecision(next_agent=None)