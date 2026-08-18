"""Compiles the agent's LangGraph.

The graph is a plan/act/reason/critique loop:

    detect_gaps ──> (work?) ──no──> END
         │yes
         v
       plan ──> execute_tools ──> reason ──> critique ──> (continue?) ──> plan
                                                              │no
                                                              v
                                                             END

The loop is what makes the agent adaptive: the critic can reject its own
reconstruction and send the planner back for stronger evidence, which a
straight-line pipeline could not do.
"""
from __future__ import annotations

from typing import Any

from ...configuration.logging import get_logger
from ...configuration.settings import Settings
from ...domain.policies.confidence_policy import ConfidencePolicy
from ...domain.policies.validation_policy import ValidationPolicy
from ...domain.services import (
    CandidateRanker,
    ContextExtractor,
    GapDetector,
    ReconstructionValidator,
    ReferenceRanker,
)
from ...infrastructure.llm.factory import build_llm_client
from ...infrastructure.persistence.checkpoints import build_checkpointer
from ...observability.events import EventEmitter
from ...tools.registry import ToolRegistry
from ..planning.planner import Planner
from ..planning.stop_policy import StopPolicy
from ..planning.tool_selector import ToolSelector
from ..reasoning.critic import Critic
from ..reasoning.reasoner import Reasoner
from ..state.state import ReconstructionState
from . import conditions, edges
from .nodes import ReconstructionNodes

_log = get_logger(__name__)


def build_nodes(
    settings: Settings, registry: ToolRegistry, events: EventEmitter | None = None
) -> ReconstructionNodes:
    """Wire the node collaborators from settings.

    Exposed separately from `build_graph` so tests can drive individual nodes
    without compiling a graph or installing LangGraph.
    """
    llm = build_llm_client(settings.llm)
    confidence = ConfidencePolicy(minimum_confidence=settings.min_confidence)

    return ReconstructionNodes(
        detector=GapDetector(),
        extractor=ContextExtractor(),
        policy=ValidationPolicy(max_gap_length=settings.max_gap_length),
        planner=Planner(llm, registry.names),
        selector=ToolSelector(ReferenceRanker()),
        registry=registry,
        reasoner=Reasoner(
            ranker=CandidateRanker(),
            validator=ReconstructionValidator(),
            confidence=confidence,
        ),
        critic=Critic(llm),
        stop_policy=StopPolicy(),
        events=events or EventEmitter(),
    )


def build_graph(
    settings: Settings,
    registry: ToolRegistry,
    events: EventEmitter | None = None,
    *,
    checkpointer: str = "memory",
) -> Any:
    """The compiled agent graph.

    Raises ImportError when LangGraph is absent - unlike the optional LLM, the
    graph has no meaningful fallback, and `run_agent` is explicit about
    requiring it.
    """
    from langgraph.graph import END, StateGraph

    nodes = build_nodes(settings, registry, events)
    graph: Any = StateGraph(ReconstructionState)

    graph.add_node(conditions.DETECT_GAPS, nodes.detect_gaps)
    graph.add_node(conditions.PLAN, nodes.plan)
    graph.add_node(conditions.EXECUTE_TOOLS, nodes.execute_tools)
    graph.add_node(conditions.REASON, nodes.reason)
    graph.add_node(conditions.CRITIQUE, nodes.critique)

    graph.set_entry_point(edges.ENTRY_POINT)

    for edge in edges.EDGES:
        graph.add_edge(edge.source, edge.target)

    # `conditions.FINISH` is the sentinel used in the topology description;
    # LangGraph wants its own END constant in the compiled mapping.
    for conditional in edges.CONDITIONAL_EDGES:
        graph.add_conditional_edges(
            conditional.source,
            conditional.condition,
            {
                key: (END if target == conditions.FINISH else target)
                for key, target in conditional.targets.items()
            },
        )

    compiled = graph.compile(checkpointer=build_checkpointer(checkpointer))
    _log.info("Reconstruction graph compiled with %d tools.", len(registry.names))
    return compiled
