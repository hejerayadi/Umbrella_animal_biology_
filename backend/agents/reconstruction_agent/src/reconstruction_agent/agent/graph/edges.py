"""The graph's topology, described independently of LangGraph.

Having the shape as data means the wiring can be asserted in tests and drawn
in documentation without compiling a graph or installing LangGraph.
"""
from __future__ import annotations

from dataclasses import dataclass

from .conditions import (
    CRITIQUE,
    DETECT_GAPS,
    EXECUTE_TOOLS,
    FINISH,
    PLAN,
    REASON,
    route_after_critique,
    route_after_detection,
)


@dataclass(frozen=True, slots=True)
class Edge:
    """A direct, unconditional transition."""

    source: str
    target: str


@dataclass(frozen=True, slots=True)
class ConditionalEdge:
    """A branch, with the mapping from predicate result to next node."""

    source: str
    condition: object
    targets: dict[str, str]


ENTRY_POINT = DETECT_GAPS

#: The linear body of one iteration.
EDGES: tuple[Edge, ...] = (
    Edge(PLAN, EXECUTE_TOOLS),
    Edge(EXECUTE_TOOLS, REASON),
    Edge(REASON, CRITIQUE),
)

#: The two decision points: whether to start at all, and whether to loop.
CONDITIONAL_EDGES: tuple[ConditionalEdge, ...] = (
    ConditionalEdge(
        source=DETECT_GAPS,
        condition=route_after_detection,
        targets={PLAN: PLAN, FINISH: FINISH},
    ),
    ConditionalEdge(
        source=CRITIQUE,
        condition=route_after_critique,
        targets={PLAN: PLAN, FINISH: FINISH},
    ),
)
