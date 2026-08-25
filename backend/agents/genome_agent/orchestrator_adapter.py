"""Adapter between the HTTP boundary and the real Genome orchestrator.

`api.py` speaks the platform contract - `instruction` plus `context` in, an
`AgentResult` out. `GenomeAgentLangGraphOrchestrator.run` takes three separate
arguments (`user_question`, `species_name`, `visualization_scope`) and returns
its own `GenomeAgentState`. This module is the only place that knows both.

No schema translation is needed on the way out: `schemas/common.py` already
defines the platform's `AgentRequest`/`AgentResult`/`AgentStatus`, with the
same four status strings. What this layer does is fill in the arguments the
orchestrator needs and decide which of its findings become shared-context keys.

Nothing under `subagents/`, `workflows/` or `orchestrator.py` is modified -
only imported. That keeps the agent's own tests and scripts working unchanged.

SVG rendering note
------------------
`render_chromosome_map` and `render_size_comparison` return raw `bytes`.
`bytes` is not JSON-serialisable, so the chart cannot leave the agent as-is.
`_visualization_summary` base64-encodes `chart_data` and stores it under
`svg_data` (a plain string).  Downstream consumers decode it with:

    import base64
    svg_bytes = base64.b64decode(visualization["svg_data"])

The global orchestrator's `_RENDER_ONLY_KEYS` in `responder.py` only hides
`protein_viewer` and `image`; `visualization` is rendered as findings text,
so the responder sees `svg_data` as a base64 string and notes it is present.
The frontend reads `context.visualization.svg_data` and renders it directly
via an `<img src="data:image/svg+xml;base64,…">` tag or equivalent.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from .orchestrator import GenomeAgentLangGraphOrchestrator
from .schema import AgentRequest, AgentResult, AgentStatus
from .workflows.state import GenomeAgentState

_logger = logging.getLogger(__name__)

# Let the query router infer what the question needs. Passing "" is not a
# missing value - `run()` treats any non-empty scope as an explicit caller
# request and refuses to override it, so "" is how we say "you decide".
_INFER_SCOPE = ""


def resolve_species_name(request: AgentRequest) -> str:
    """The species to look up, preferring what the platform already established.

    The Global Orchestrator runs a dedicated extraction step before any agent
    is called and writes `context["species"]`. That is a purpose-built reading
    of the question, so it wins. Falling back to the raw instruction lets the
    agent still work when called directly, since NCBI's search tolerates a
    sentence better than an empty string does.
    """
    context = request.context or {}
    return str(
        context.get("species")
        or context.get("species_name")
        or request.instruction
        or ""
    ).strip()


def _summarise(state: GenomeAgentState) -> str:
    """The `genome` context key: a short human-readable line about the genome.

    Every downstream agent tests `"genome" in context` before deciding whether
    to escalate to this one - Trait, Evolution, Protein and Reconstruction all
    do. The key must therefore always be present on a completed run, and it is
    a string because that is what those agents were written to expect.
    """
    species = state.species or {}
    name = species.get("scientific_name") or species.get("common_name") or state.species_name
    parts = [f"Genome of {name}"]

    if state.assembly_id:
        parts.append(f"assembly {state.assembly_id}")
    metadata = state.metadata or {}
    if metadata.get("genome_size_bp"):
        parts.append(f"{metadata['genome_size_bp'] / 1_000_000_000:.2f} Gb")
    if metadata.get("chromosome_count"):
        parts.append(f"{metadata['chromosome_count']} chromosomes")

    return ", ".join(parts) + " (source: NCBI)"


def _visualization_summary(visualization: dict[str, Any]) -> dict[str, Any]:
    """Serialise a rendered chart so it can travel over HTTP as JSON.

    `render_chromosome_map` and `render_size_comparison` return raw SVG
    `bytes`.  `bytes` is not JSON-serialisable, so the bytes are base64-
    encoded and stored under `svg_data`.  The frontend decodes them with:

        <img src="data:image/svg+xml;base64,{svg_data}" />

    or, equivalently:
        const svg = atob(visualization.svg_data);

    `available` is kept for back-compat with any consumer that only checks
    whether a chart exists before rendering it.
    """
    chart_data: bytes | None = visualization.get("chart_data")
    svg_data: str | None = None
    if isinstance(chart_data, bytes) and chart_data:
        svg_data = base64.b64encode(chart_data).decode("ascii")

    summary: dict[str, Any] = {
        "status": visualization.get("status"),
        "format": visualization.get("format"),
        "available": svg_data is not None,
    }
    if svg_data is not None:
        summary["svg_data"] = svg_data
    for key in ("note", "comparisons"):
        if visualization.get(key) is not None:
            summary[key] = visualization[key]
    return summary

def to_result(state: GenomeAgentState) -> AgentResult:
    """Map the orchestrator's final state onto the platform's AgentResult."""

    # No assembly id means species resolution failed, and every later step
    # needs it. Reported with the name that was actually searched, because
    # "NCBI does not know that name" is something the user can act on.
    if state.assembly_id is None:
        _logger.info("[Genome] could not resolve species %r", state.species_name)
        return AgentResult(
            status=AgentStatus.FAILED,
            output=(
                f"No NCBI genome assembly could be found for '{state.species_name}'. "
                f"Check the spelling, or try the scientific name."
            ),
        )

    output: dict[str, Any] = {
        "genome": _summarise(state),
        "assembly_id": state.assembly_id,
    }

    if state.species:
        output["species_record"] = state.species
    if state.metadata:
        output["genome_metadata"] = state.metadata

    annotation = state.annotation or {}
    if annotation.get("gene_list"):
        # The key the Trait Discovery Agent waits for. Real NCBI gene symbols
        # rather than a fixed list - this is the whole point of running the
        # real agent instead of the mock.
        output["gene_list"] = annotation["gene_list"]
    if annotation.get("gene_table"):
        output["gene_table"] = annotation["gene_table"]
    if state.explanation:
        output["explanation"] = state.explanation
    if state.errors:
        # Non-fatal: a partial answer with a note beats no answer at all.
        output["warnings"] = list(state.errors)

    visualization = state.visualization
    if visualization:
        output["visualization"] = _visualization_summary(visualization)

    # Reconstruction handoff takes priority over the visualization handoff
    # below: get_genome_metadata_node (workflows/nodes/genome_data_nodes.py)
    # sets reconstruction_need whenever assembly_level is Scaffold/Contig,
    # and _route_after_join_parallel routes to reconstruction_resolver
    # instead of generate_visualization in that case — so visualization
    # never even runs. Checking reconstruction_need first here keeps this
    # function's precedence consistent with the graph's own routing.
    need = state.reconstruction_need or {}
    if need.get("status") == "NEEDS_AGENT":
        _logger.info(
            "[Genome] assembly %s is incomplete — needs reconstruction agent",
            state.assembly_id,
        )
        return AgentResult(
            status=AgentStatus.NEEDS_AGENT,
            target_agent=need.get("target_agent"),
            prompt_to_target_agent=need.get("prompt_to_target_agent"),
            output=output,
        )

    # Separate handoff: a *requested* protein_structure visualization that
    # this agent can't render itself. Mutually exclusive with the
    # reconstruction_need branch above — the graph only reaches
    # generate_visualization when reconstruction_need was NOT triggered.
    if visualization and visualization.get("status") == "NEEDS_AGENT":
        _logger.info("[Genome] needs another agent for the requested visualization")
        return AgentResult(
            status=AgentStatus.NEEDS_AGENT,
            target_agent=visualization.get("target_agent"),
            prompt_to_target_agent=visualization.get("prompt_to_target_agent"),
            output=output,
        )

    return AgentResult(status=AgentStatus.COMPLETED, output=output)
class OrchestratorGenomeAgent:
    """Serves the real LangGraph orchestrator behind the agent's endpoint.

    `run(request) -> AgentResult` is the whole interface `api.py` depends on.
    It is async because the orchestrator's graph is, and because its subagents
    call NCBI over the network.
    """

    def __init__(self, orchestrator: GenomeAgentLangGraphOrchestrator | None = None) -> None:
        # Built once: constructing it compiles the LangGraph state machine.
        self._orchestrator = orchestrator or GenomeAgentLangGraphOrchestrator()

    async def run(self, request: AgentRequest) -> AgentResult:
        species_name = resolve_species_name(request)
        if not species_name:
            return AgentResult(
                status=AgentStatus.FAILED,
                output="No species was named in the request, so there is nothing to look up.",
            )

        state = await self._orchestrator.run(
            user_question=request.instruction,
            species_name=species_name,
            visualization_scope=_INFER_SCOPE,
        )
        return to_result(state)
