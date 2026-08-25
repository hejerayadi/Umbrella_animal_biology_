"""The graph's topology, described independently of LangGraph.

Having the shape as data means the wiring can be asserted in tests and drawn
in documentation without compiling a graph or installing LangGraph.

    load_or_init -> detect_gaps -> (work?) --no--> finalize
                                      | yes
                                      v
        +--------------> plan -> select_tools -> (runnable?) --no--> critique
        |                              | yes                            ^
        |                              v                                |
        |                     execute_tools -> observe -> reason -> validate
        |                                                                |
        |                                                                v
        +---- REVISE ------------------------------------------------ decide
                                       |
                    ACCEPT / ABSTAIN / budget / yield
                                       v
                                   finalize
"""
from __future__ import annotations

from dataclasses import dataclass

from agent.graph.conditions import (
    CRITIQUE,
    DECIDE,
    DETECT_GAPS,
    EXECUTE_TOOLS,
    FINALIZE,
    LOAD_OR_INIT,
    OBSERVE,
    PLAN,
    REASON,
    SELECT_TOOLS,
    VALIDATE,
    route_after_decision,
    route_after_detection,
    route_after_selection,
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


ENTRY_POINT = LOAD_OR_INIT

#: The linear body of one iteration.
EDGES: tuple[Edge, ...] = (
    Edge(LOAD_OR_INIT, DETECT_GAPS),
    Edge(PLAN, SELECT_TOOLS),
    Edge(EXECUTE_TOOLS, OBSERVE),
    Edge(OBSERVE, REASON),
    Edge(REASON, VALIDATE),
    Edge(VALIDATE, CRITIQUE),
    Edge(CRITIQUE, DECIDE),
    Edge(FINALIZE, "__end__"),
)

#: The three decision points: whether to start, whether the plan produced
#: anything runnable, and whether to loop.
CONDITIONAL_EDGES: tuple[ConditionalEdge, ...] = (
    ConditionalEdge(
        source=DETECT_GAPS,
        condition=route_after_detection,
        targets={PLAN: PLAN, FINALIZE: FINALIZE},
    ),
    ConditionalEdge(
        source=SELECT_TOOLS,
        condition=route_after_selection,
        targets={EXECUTE_TOOLS: EXECUTE_TOOLS, CRITIQUE: CRITIQUE},
    ),
    ConditionalEdge(
        source=DECIDE,
        condition=route_after_decision,
        targets={PLAN: PLAN, FINALIZE: FINALIZE},
    ),
)
