"""Feature-to-worker routing for the Evolution Orchestrator.

The router is intentionally the smallest piece of the orchestrator: it
takes an ``AgentRequest``, reads the ``feature`` field, and returns the
worker callable that should handle it. All planning / aggregation logic
lives elsewhere.

Two selection strategies are exposed:

- ``pick``      — single feature → single worker.
- ``pick_many`` — a set of features (e.g. when the user asks for both a
                  tree and a divergence time in one turn) → ordered list
                  of workers for parallel dispatch.
"""

from __future__ import annotations

from typing import Iterable, Protocol

from ..schema import AgentRequest, AgentResult, EvolutionaryFeature


class Worker(Protocol):
    def run(self, request: AgentRequest) -> AgentResult: ...


class Router:
    def __init__(self, workers: dict[EvolutionaryFeature, Worker]) -> None:
        missing = set(EvolutionaryFeature) - set(workers)
        if missing:
            names = ", ".join(f.value for f in missing)
            raise ValueError(f"Router missing worker(s) for: {names}")
        self._workers = workers

    def pick(self, request: AgentRequest) -> tuple[EvolutionaryFeature, Worker]:
        """Return the worker for ``request.feature``.

        Raises ``ValueError`` if the feature field is missing or unknown
        so the orchestrator can convert it into a well-formed FAILED result.
        """
        if not request.feature:
            raise ValueError(
                "AgentRequest.feature is required for the Evolution Agent. "
                "Expected one of: "
                + ", ".join(f.value for f in EvolutionaryFeature)
            )
        try:
            feature = EvolutionaryFeature(request.feature)
        except ValueError as exc:
            raise ValueError(
                f"Unknown evolutionary feature: {request.feature!r}"
            ) from exc
        return feature, self._workers[feature]

    def pick_many(
        self,
        features: Iterable[str | EvolutionaryFeature],
    ) -> list[tuple[EvolutionaryFeature, Worker]]:
        """Return an ordered list of (feature, worker) for parallel dispatch."""
        chosen: list[tuple[EvolutionaryFeature, Worker]] = []
        for f in features:
            feature = EvolutionaryFeature(f) if isinstance(f, str) else f
            chosen.append((feature, self._workers[feature]))
        return chosen
