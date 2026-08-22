"""Translation layer between the HTTP boundary and the LangGraph workflow.

`api.py` speaks the orchestrator's schema (`schema.py`: instruction + context in,
AgentResult out). The workflow under `workflows/` speaks its own richer schema
(`schemas/`: trait_name, species_name, gene_list in; annotations, pathways,
proteins, evidence, explanation out). Neither should have to know about the
other, so this module is the single place that knows both.

Everything under `workflows/`, `schemas/` and `subagents/` is deliberately left
untouched by this file - it only imports them. That is what keeps the workflow's
own tests and `python -m workflows` runnable with no API, no orchestrator and no
HTTP, exactly as before.

The two mapping functions are pure and side-effect free so they can be tested
without a network, a NIM key, or a running graph. `WorkflowTraitAgent` is the
thin async wrapper that actually drives the compiled graph.
"""
from __future__ import annotations

import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

# The workflow package imports itself absolutely (`from workflows.state import
# ...`, `from schemas.common import ...`), which only resolves when this agent's
# own directory is on sys.path. That happens for free under `python -m workflows`
# and in tests via conftest.py, but NOT when api.py is imported as
# `backend.agents.trait_discovery_agent.api` from the repository root. Adding it
# here keeps that bootstrap in one place instead of spreading it through the
# workflow, and is a no-op when something else already added it.
_AGENT_ROOT = Path(__file__).resolve().parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from schemas.common import AgentStatus as WorkflowStatus  # noqa: E402
from workflows.state import TraitDiscoveryState  # noqa: E402
from workflows.trait_discovery_graph import build_trait_discovery_graph  # noqa: E402

# This module is imported two ways: as part of the agent package (from api.py,
# where the relative import is correct) and as a top-level module (from tests,
# which put the agent root on sys.path). Supporting both is what lets the mapping
# functions be tested without dragging in FastAPI or the package machinery.
try:  # pragma: no cover - import-style shim, both branches are equivalent
    from .schema import AgentRequest, AgentResult, AgentStatus
except ImportError:  # pragma: no cover
    from schema import AgentRequest, AgentResult, AgentStatus

# Used when the orchestrator did not name a specific trait - a broad question
# like "explain the traits of the African elephant" seeds a species but no trait.
# The workflow needs *some* trait string for its literature lookup and for the
# explanation it writes, and this reads correctly in that sentence.
_DEFAULT_TRAIT = "observable traits"
_DEFAULT_SPECIES = "unspecified species"


def _to_plain(value: Any) -> Any:
    """Recursively turn workflow dataclasses into JSON-serialisable values.

    The workflow returns `GOAnnotation`, `PathwayEntry` and friends. Whatever
    goes into `AgentResult.output` is merged into the orchestrator's shared
    context and later serialised to JSON, so it has to be plain data.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    return value


def to_state(request: AgentRequest) -> TraitDiscoveryState:
    """Map an orchestrator request onto the workflow's input state.

    `trait_name` and `species` are read from the shared context because the
    orchestrator's extractor node puts them there before any agent runs.
    `gene_list` is not read here on purpose - `TraitDiscoveryState.__post_init__`
    already lifts it out of `context`, so passing the context through untouched
    is enough, and duplicating that logic would risk the two disagreeing.
    """
    context = request.context or {}

    return TraitDiscoveryState(
        trait_name=str(context.get("trait_name") or _DEFAULT_TRAIT),
        # The extractor writes "species"; accept "species_name" too so a caller
        # using the workflow's own vocabulary is not silently ignored.
        species_name=str(
            context.get("species") or context.get("species_name") or _DEFAULT_SPECIES
        ),
        instruction=request.instruction,
        context=dict(context),
    )


def _describe_failure(final_state: dict) -> str:
    """Say which step failed, rather than just that something did."""
    if final_state.get("gene_mapper_status") == WorkflowStatus.FAILED:
        genes = list(final_state.get("gene_list") or [])
        # Name a few rather than all of them: a live NCBI query returns ~50
        # symbols, and this string is merged into context and then rendered
        # into the Responder's prompt.
        shown = ", ".join(str(gene) for gene in genes[:8])
        if len(genes) > 8:
            shown += f", ... (+{len(genes) - 8} more)"
        return (
            f"Gene mapping failed: none of the {len(genes)} supplied genes "
            f"({shown}) could be annotated, so there was nothing to build traits "
            "from. The Gene Mapper only fails when NOTHING matches - a partly "
            "matched set is reported in unmatched_genes and the workflow carries "
            "on. Its GO lookup is still the stub in subagents/gene_mapper.py, "
            "which holds five hand-written genes, so real NCBI symbols for any "
            "species will miss every time."
        )
    if final_state.get("functional_evidence_status") == WorkflowStatus.FAILED:
        return "Functional evidence lookup failed: no pathway or protein data was found."
    if final_state.get("literature_status") == WorkflowStatus.FAILED:
        return "Literature support failed: no evidence was found for this trait."
    return "Trait discovery workflow failed."


def to_result(final_state: dict) -> AgentResult:
    """Map the workflow's final state onto the orchestrator's AgentResult.

    The workflow's status values are the same four strings as the orchestrator's,
    but they come from a different `AgentStatus` class, so they are converted by
    value rather than passed through.
    """
    workflow_status = final_state.get("status")
    if workflow_status is None:
        return AgentResult(
            status=AgentStatus.FAILED,
            output="Trait discovery workflow returned no status.",
        )

    status = AgentStatus(workflow_status.value)

    if status is AgentStatus.FAILED:
        return AgentResult(status=status, output=_describe_failure(final_state))

    # Whatever was gathered before the workflow paused or finished. Sent on both
    # the completed and needs_agent paths: an escalation still carries real
    # findings, and the orchestrator merges them so the next agent can use them.
    output: dict[str, Any] = {
        # The ecosystem contract. card.json declares `traits`, and the Protein
        # and ImageGeneration agents both branch on its presence - drop this key
        # and they silently start waiting on something that never arrives.
        "traits": [a.gene_symbol for a in final_state.get("go_annotations", [])],
        "go_annotations": _to_plain(final_state.get("go_annotations", [])),
        "pathways": _to_plain(final_state.get("pathway_data", [])),
        "proteins": _to_plain(final_state.get("protein_data", [])),
        "evidence": _to_plain(final_state.get("evidence", [])),
    }

    explanation = final_state.get("explanation")
    if explanation:
        output["explanation"] = explanation

    if status is AgentStatus.NEEDS_AGENT:
        return AgentResult(
            status=status,
            target_agent=final_state.get("target_agent"),
            prompt_to_target_agent=final_state.get("prompt_to_target_agent"),
            output=output,
        )

    return AgentResult(status=status, output=output)


class WorkflowTraitAgent:
    """Runs the real LangGraph workflow behind the agent's HTTP endpoint.

    Exposes the same `run(request) -> AgentResult` shape as `TraitMock`, except
    that it is async - the graph is async all the way down.
    """

    def __init__(self) -> None:
        # Compiled once at startup, like the mock is constructed once: building
        # the graph per request would rebuild every node and subgraph.
        self._graph = build_trait_discovery_graph()

    async def run(self, request: AgentRequest) -> AgentResult:
        final_state = await self._graph.ainvoke(to_state(request))
        return to_result(final_state)
