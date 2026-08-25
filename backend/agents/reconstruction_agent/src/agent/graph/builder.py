"""Compiles the agent's LangGraph.

The graph is a genuine agentic loop, not a pipeline: `decide` can send it round
again with the critic's objection attached, and `select_tools` chooses what to
run from the catalogue rather than replaying a fixed order.

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

The graph is compiled with a checkpointer and re-entered on every CONTINUE
retry, keyed by the orchestrator's trace id - see `application/run_agent.py`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent.graph import conditions, edges
from agent.graph.nodes import ReconstructionNodes
from agent.planning.planner import Planner
from agent.planning.stop_policy import StopPolicy
from agent.planning.tool_selector import ToolSelector
from agent.reasoning.critic import Critic
from agent.reasoning.reasoner import Reasoner
from agent.state.state import ReconstructionState
from configuration.logging import get_logger
from configuration.settings import Settings
from domain.policies.budget_policy import BudgetPolicy, Budgets
from domain.policies.confidence_policy import ConfidencePolicy
from domain.policies.validation_policy import ValidationPolicy
from domain.services import (
    CandidateRanker,
    ContextExtractor,
    GapDetector,
    ReconstructionValidator,
    ReferenceRanker,
)
from infrastructure.llm.factory import build_llm_client
from observability.events import EventEmitter
from tools.registry import ToolRegistry

if TYPE_CHECKING:
    from tools.blast.advisor import DatabaseAdvisor

_log = get_logger(__name__)


def build_budget_policy(settings: Settings) -> BudgetPolicy:
    """The allowance one run may spend, from configuration."""
    return BudgetPolicy(
        Budgets(
            max_tool_calls=settings.budgets.max_tool_calls,
            per_tool=settings.budgets.per_tool(),
            max_llm_tokens=settings.budgets.max_llm_tokens,
            yield_after_seconds=settings.continuation.effective_yield_after_seconds,
        )
    )


def build_nodes(
    settings: Settings,
    registry: ToolRegistry,
    events: EventEmitter | None = None,
    *,
    databases: DatabaseAdvisor | None = None,
) -> ReconstructionNodes:
    """Wire the node collaborators from settings.

    Exposed separately from `build_graph` so tests can drive individual nodes
    without compiling a graph or installing LangGraph.
    """
    llm = build_llm_client(settings.llm, settings.azure)
    confidence = ConfidencePolicy(minimum_confidence=settings.min_confidence)
    validator = ReconstructionValidator()

    return ReconstructionNodes(
        detector=GapDetector(),
        extractor=ContextExtractor(),
        policy=ValidationPolicy(max_gap_length=settings.max_gap_length),
        planner=Planner(llm, registry.names),
        selector=ToolSelector(ReferenceRanker()),
        registry=registry,
        reasoner=Reasoner(
            ranker=CandidateRanker(),
            validator=validator,
            confidence=confidence,
        ),
        validator=validator,
        critic=Critic(llm),
        stop_policy=StopPolicy(),
        budgets=build_budget_policy(settings),
        events=events or EventEmitter(),
        # Optional: with no advisor the planner's own database choice stands
        # alone, which is what keeps the offline smoke test running with an
        # empty registry and no network.
        databases=databases,
        max_gaps_per_run=settings.max_gaps_per_run,
    )


def build_graph(
    settings: Settings,
    registry: ToolRegistry,
    events: EventEmitter | None = None,
    *,
    checkpointer: Any | None = None,
    nodes: ReconstructionNodes | None = None,
    databases: DatabaseAdvisor | None = None,
) -> Any:
    """The compiled agent graph.

    Raises ImportError when LangGraph is absent - unlike the optional LLM, the
    graph has no meaningful fallback, and `run_agent` is explicit about
    requiring it.

    `checkpointer` is passed in rather than built here because it owns a
    database connection whose lifetime belongs to the application, not to a
    graph construction call.

    `nodes` is passed in for the same reason. `AgentRunner` needs the same
    instance for its no-LangGraph fallback path, and building a second set here
    would give the process two LLM clients, two of every collaborator, and two
    different objects a test could patch - only one of which the graph would
    actually call.
    """
    from langgraph.graph import END, StateGraph

    nodes = nodes or build_nodes(settings, registry, events, databases=databases)
    graph: Any = StateGraph(ReconstructionState)

    for name, handler in (
        (conditions.LOAD_OR_INIT, nodes.load_or_init),
        (conditions.DETECT_GAPS, nodes.detect_gaps),
        (conditions.PLAN, nodes.plan),
        (conditions.SELECT_TOOLS, nodes.select_tools),
        (conditions.EXECUTE_TOOLS, nodes.execute_tools),
        (conditions.OBSERVE, nodes.observe),
        (conditions.REASON, nodes.reason),
        (conditions.VALIDATE, nodes.validate),
        (conditions.CRITIQUE, nodes.critique),
        (conditions.DECIDE, nodes.decide),
        (conditions.FINALIZE, nodes.finalize),
    ):
        graph.add_node(name, handler)

    graph.set_entry_point(edges.ENTRY_POINT)

    for edge in edges.EDGES:
        if edge.target == "__end__":
            graph.add_edge(edge.source, END)
        else:
            graph.add_edge(edge.source, edge.target)

    for conditional in edges.CONDITIONAL_EDGES:
        graph.add_conditional_edges(
            conditional.source,
            conditional.condition,
            dict(conditional.targets),
        )

    compiled = graph.compile(checkpointer=checkpointer)
    _log.info(
        "graph_compiled",
        tool_count=len(registry.names),
        checkpointer=type(checkpointer).__name__ if checkpointer else "none",
    )
    return compiled
