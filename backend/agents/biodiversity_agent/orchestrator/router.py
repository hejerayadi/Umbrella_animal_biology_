"""Feature-to-worker routing for the Biodiversity Orchestrator.

The router is intentionally the smallest piece of the orchestrator:
it takes an ``AgentRequest``, reads the ``feature`` field, and returns
the worker callable that should handle it. All planning / aggregation
logic lives elsewhere.

Two selection strategies are exposed:

- ``pick`` - single feature -> single worker.
- ``pick_many`` - a set of features (e.g. when the user asks for
  distribution AND migration in one turn) -> ordered list of workers,
  so the orchestrator can dispatch them in parallel.
"""

from __future__ import annotations

from typing import Iterable, Protocol

from ..schema import AgentRequest, AgentResult, BiodiversityFeature


class Worker(Protocol):
    def run(self, request: AgentRequest) -> AgentResult: ...


class Router:
    def __init__(self, workers: dict[BiodiversityFeature, Worker]) -> None:
        missing = set(BiodiversityFeature) - set(workers)
        if missing:
            names = ", ".join(f.value for f in missing)
            raise ValueError(f"Router missing worker(s) for: {names}")
        self._workers = workers

    def pick(self, request: AgentRequest) -> tuple[BiodiversityFeature, Worker]:
        """Return the worker for ``request.feature``.

        Raises ``ValueError`` if the feature field is missing or unknown so
        the orchestrator can convert it into a well-formed FAILED result.
        """

        if not request.feature:
            raise ValueError(
                "AgentRequest.feature is required for the Biodiversity Agent. "
                "Expected one of: "
                + ", ".join(f.value for f in BiodiversityFeature)
            )
        try:
            feature = BiodiversityFeature(request.feature)
        except ValueError as exc:
            raise ValueError(f"Unknown biodiversity feature: {request.feature}") from exc
        return feature, self._workers[feature]

    def pick_many(
        self, features: Iterable[str | BiodiversityFeature]
    ) -> list[tuple[BiodiversityFeature, Worker]]:
        chosen: list[tuple[BiodiversityFeature, Worker]] = []
        for f in features:
            feature = BiodiversityFeature(f) if isinstance(f, str) else f
            chosen.append((feature, self._workers[feature]))
        return chosen
