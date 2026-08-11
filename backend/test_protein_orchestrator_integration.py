"""Offline Main Orchestrator round trips for Protein hand-offs.

The production LangGraph and HTTP request/response parsing run unchanged. Only
the LLM decisions and remote HTTP services are scripted, so no provider, Azure,
Qdrant, or embedding model is touched.
"""

from __future__ import annotations

from typing import Any

import pytest

from backend.orchestrator.langgraph.graph import build_orchestrator_graph
from backend.orchestrator.planner import ExecutionPlan
from backend.orchestrator.state import WorkflowState


class _Planner:
    def plan(self, user_query: str, *, has_image: bool = False) -> ExecutionPlan:
        return ExecutionPlan(
            initial_agent="Protein", reasoning="protein structure request"
        )


class _Extractor:
    def extract(self, user_query: str) -> dict[str, Any]:
        return {"species": "Homo sapiens", "gene_name": "TP53", "mutation": "R273H"}


class _Resolver:
    def __init__(self, target: str) -> None:
        self.target = target

    def resolve(self, current_agent: str, prompt_to_target_agent: str) -> str:
        assert current_agent == "Protein"
        assert "missing evidence" in prompt_to_target_agent
        return self.target


class _Responder:
    def answer_directly(self, user_query: str) -> str:
        return "unused"

    def synthesize(
        self,
        user_query: str,
        context: dict[str, Any],
        execution_history: list[str],
        failure: str | None = None,
    ) -> str:
        return failure or str(context["protein_structure"])


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class _HandoffClient:
    def __init__(self, helper: str) -> None:
        self.helper = helper
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.protein_calls = 0

    def post(self, url: str, json: dict[str, Any], **kwargs: Any) -> _Response:
        agent = url.split("//", 1)[-1].split("/", 1)[0].capitalize()
        self.calls.append((agent, json))

        if agent == "Protein":
            self.protein_calls += 1
            if self.protein_calls == 1:
                return _Response(
                    {
                        "status": "needs_agent",
                        "target_agent": self.helper,
                        "prompt_to_target_agent": "Supply missing evidence for TP53",
                        "output": {"protein_warnings": ["evidence incomplete"]},
                    }
                )
            return _Response(
                {
                    "status": "completed",
                    "output": {
                        "protein_structure": "3D structure of TP53: RCSB PDB 1TUP."
                    },
                }
            )

        assert agent == self.helper
        return _Response(
            {
                "status": "completed",
                "output": {"helper_evidence": f"{self.helper} evidence"},
            }
        )


@pytest.mark.parametrize("helper", ["Trait", "Genome", "Literature"])
def test_main_orchestrator_routes_merges_and_resumes_protein(helper: str) -> None:
    client = _HandoffClient(helper)
    endpoints = {
        "Protein": "http://protein",
        "Trait": "http://trait",
        "Genome": "http://genome",
        "Literature": "http://literature",
    }
    graph = build_orchestrator_graph(
        planner=_Planner(),
        extractor=_Extractor(),
        resolver=_Resolver(helper),
        responder=_Responder(),
        agent_cards={},
        agent_endpoints=endpoints,
        worker_client=client,
        sleep=lambda _: None,
    )

    raw = graph.invoke(WorkflowState(user_query="Explain the TP53 R273H structure"))
    state = raw if isinstance(raw, WorkflowState) else WorkflowState(**raw)

    assert [agent for agent, _ in client.calls] == ["Protein", helper, "Protein"]
    assert client.calls[1][1]["instruction"] == "Supply missing evidence for TP53"
    assert client.calls[2][1]["instruction"] == "Explain the TP53 R273H structure"
    assert state.context["mutation"] == "R273H"
    assert state.context["protein_warnings"] == ["evidence incomplete"]
    assert state.context["helper_evidence"] == f"{helper} evidence"
    assert state.context["protein_structure"].endswith("1TUP.")
    assert state.final_answer == "3D structure of TP53: RCSB PDB 1TUP."
