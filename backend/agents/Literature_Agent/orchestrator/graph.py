"""The Literature Agent's top-level LangGraph.

    router ─┬─ discovery ─┬─ writing ─┐
            │             └───────────┤
            └─ writing ───────────────┴─ aggregate ─ END

The router classifies the instruction into one of four routes (see
`routing/router.py`); `orchestrator/router.py` turns that route into the nodes
that run next. `both_sequential` is the only route where writing waits for
discovery, so that the draft can cite the papers that were actually found.
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from ..routing.router import classify_route_llm
from ..schema import AgentRequest, AgentResult, AgentStatus
from ..subagents.discovery import KnowledgeDiscoveryOrchestrator
from ..subagents.writing import ScientificWritingOrchestrator
from .router import decide_after_discovery, decide_next_after_routing
from .state import OrchestratorState, initial_state

# The agent's single .env, addressed explicitly. A bare load_dotenv() walks up
# from the working directory, which is the repository root when the service is
# started the documented way - and backend/.env holds none of these keys.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


def aggregate_results(state: OrchestratorState) -> dict:
    """Merge whichever branches ran into one `AgentResult`.

    Both keys are always present, set to None for a branch that did not run,
    so a caller never has to guess which route was taken.
    """
    discovery = state.get("discovery_result")
    writing = state.get("writing_result")

    # FAILED propagates. A branch that could not produce anything must not be
    # reported as COMPLETED with an empty payload: the Responder treats a
    # completed-but-empty finding as something to write around, and fills the
    # gap from its own memory. An explicit failure it explains honestly.
    #
    # This has to be `all`, not `any`. Discovery currently reads from a
    # placeholder that cannot fail, so on either "both" route `any` made the
    # result COMPLETED no matter what writing did - a draft that was never
    # produced was reported to the Global Orchestrator as a success. Both
    # payloads are still returned either way, so nothing is lost by failing.
    ran = [r for r in (discovery, writing) if r]
    status = (
        AgentStatus.COMPLETED
        if ran and all(r.status == AgentStatus.COMPLETED for r in ran)
        else AgentStatus.FAILED
    )

    return {
        "final_result": AgentResult(
            status=status,
            output={
                "discovery": discovery.output if discovery else None,
                "writing": writing.output if writing else None,
            },
        )
    }


class LiteratureOrchestrator:
    def __init__(self) -> None:
        self._discovery = KnowledgeDiscoveryOrchestrator()
        self._writing = ScientificWritingOrchestrator()
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(OrchestratorState)

        graph.add_node("router", self._route)
        graph.add_node("discovery", self._run_discovery)
        graph.add_node("writing", self._run_writing)
        graph.add_node("aggregate", aggregate_results)

        graph.set_entry_point("router")
        graph.add_conditional_edges("router", decide_next_after_routing)
        graph.add_conditional_edges("discovery", decide_after_discovery)
        graph.add_edge("writing", "aggregate")
        graph.add_edge("aggregate", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    def _route(self, state: OrchestratorState) -> dict:
        """Classify the instruction. Never raises - discovery is the fallback."""
        route = classify_route_llm(state["request"].instruction) or "discovery"
        return {"route": route}

    def _run_discovery(self, state: OrchestratorState) -> dict:
        return {"discovery_result": self._discovery.run(state["request"])}

    def _run_writing(self, state: OrchestratorState) -> dict:
        """Draft the text, handing over the search results when there are any.

        On `both_parallel` this node runs in the same superstep as discovery,
        so `discovery_result` is still None and the draft stands on its own.
        That is the difference between the two "both" routes.
        """
        request = state["request"]
        context = dict(request.context or {})

        discovery = state.get("discovery_result")
        if discovery and discovery.status == AgentStatus.COMPLETED:
            context["discovery_output"] = discovery.output

        result = self._writing.run(
            AgentRequest(instruction=request.instruction, context=context)
        )
        return {"writing_result": result}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_request(self, request: AgentRequest) -> AgentResult:
        """Run the graph for an already-built request.

        The way in for the HTTP boundary: `api.py` is handed an `AgentRequest`
        by the Global Orchestrator and must not reach into `_graph` to run it.
        """
        return self._graph.invoke(initial_state(request))["final_result"]

    def run(self, instruction: str, context: dict | None = None) -> AgentResult:
        """Convenience entry point for direct Python and script usage."""
        return self.run_request(
            AgentRequest(instruction=instruction, context=context or {})
        )
