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
"""

from __future__ import annotations

import logging
from typing import Any

from .orchestrator import GenomeAgentLangGraphOrchestrator
from .schema import AgentRequest, AgentResult, AgentStatus
from .subagents.sequence_window import find_largest_genomic_scaffold as _find_scaffold
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


# A rendered chart is a few kilobytes of SVG text (~1.4 KB for a size
# comparison, ~9.5 KB for a fifty-gene chromosome map), so it travels inline
# rather than behind a file endpoint. The ceiling guards the case this cannot
# predict: gene tables are not bounded by anything here, and chat messages are
# persisted to localStorage in the browser, where a runaway payload would evict
# the conversation it belongs to. Over the limit the chart is described only,
# exactly as it used to be.
_MAX_INLINE_CHART_BYTES = 128 * 1024


def _visualization_summary(visualization: dict[str, Any]) -> dict[str, Any]:
    """Describe a chart, and carry it when it is small enough to inline.

    `chart_data` arrives as `bytes` from the renderer, which is not
    JSON-serialisable and would break the response on the way out. It is
    decoded to a `chart_svg` string here so the frontend can render it; the
    numbers behind it still travel in `comparisons`, which is what the
    explanation writer uses either way.
    """
    chart = visualization.get("chart_data")
    summary: dict[str, Any] = {
        "status": visualization.get("status"),
        "format": visualization.get("format"),
        "available": chart is not None,
    }
    for key in ("note", "comparisons"):
        if visualization.get(key) is not None:
            summary[key] = visualization[key]

    if isinstance(chart, (bytes, bytearray)) and len(chart) <= _MAX_INLINE_CHART_BYTES:
        try:
            summary["chart_svg"] = bytes(chart).decode("utf-8")
        except UnicodeDecodeError:
            # A non-text chart format would land here. Better to lose the
            # picture than to fail the whole answer over it.
            _logger.warning("[Genome] chart_data was not valid UTF-8; not inlining")
    elif isinstance(chart, str):
        summary["chart_svg"] = chart
    elif chart is not None:
        _logger.info(
            "[Genome] chart is %d bytes, over the %d inline limit; describing only",
            len(chart), _MAX_INLINE_CHART_BYTES,
        )

    return summary


# Two very different reasons land on the same "no assembly" outcome, and they
# need different answers. "NCBI has never heard of this name" is the user's
# problem, and a spelling hint helps. "This taxon exists, but nothing has ever
# been assembled for it" is not: the woolly mammoth resolves cleanly to taxid
# 37349 and has zero assemblies, so telling that user to check their spelling
# or try `Mammuthus primigenius` sends them to a search that also finds
# nothing.
#
# The resolver does record which case it hit, but only as free-text LLM
# `reasoning` whose wording changes from run to run ("returned empty results
# for all candidates" one time, "failed to find an assembly for that tax_id"
# the next). Matching on that prose is unreliable, so the distinction is
# re-established here with one cheap taxonomy lookup instead. It costs a
# request only on the failure path, which is rare by definition.


def _taxon_is_known_to_ncbi(name: str) -> bool | None:
    """Whether NCBI Taxonomy has an entry for `name`.

    `None` means the question could not be answered - NCBI was unreachable or
    answered with something unexpected - so the caller falls back to the
    generic wording rather than asserting either explanation.
    """
    if not name:
        return None
    try:
        from .subagents._ncbi_client import ncbi_get

        response = ncbi_get(
            {"path": "esearch.fcgi", "db": "taxonomy", "term": name, "retmode": "json"}
        )
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 - diagnosis must never mask the failure
        _logger.info("[Genome] taxonomy check for %r failed: %s", name, exc)
        return None
    return bool(payload.get("esearchresult", {}).get("idlist"))


def _unresolved_message(state: GenomeAgentState) -> str:
    """Explain a missing assembly in terms of why it is actually missing."""

    species = state.species or {}
    named = species.get("scientific_name") or state.species_name

    if _taxon_is_known_to_ncbi(state.species_name) is True:
        return (
            f"NCBI has no genome assembly for '{named}'. The species is in NCBI's "
            f"taxonomy, but no genome has been assembled and deposited for it, so "
            f"there is no assembly, size or gene table to report. That is a gap in "
            f"the public data rather than a lookup error - a different spelling or "
            f"the scientific name will not find one either."
        )

    return (
        f"No NCBI genome assembly could be found for '{state.species_name}'. "
        f"Check the spelling, or try the scientific name."
    )


def to_result(state: GenomeAgentState) -> AgentResult:
    """Map the orchestrator's final state onto the platform's AgentResult."""

    # No assembly id means species resolution failed, and every later step
    # needs it. Reported with the name that was actually searched, because
    # "NCBI does not know that name" is something the user can act on.
    if state.assembly_id is None:
        _logger.info("[Genome] could not resolve species %r", state.species_name)
        return AgentResult(
            status=AgentStatus.FAILED,
            output=_unresolved_message(state),
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
    # instead of generate_visualization in that case - so visualization
    # never even runs. Checking reconstruction_need first here keeps this
    # function's precedence consistent with the graph's own routing.
    need = state.reconstruction_need or {}
    if need.get("status") == "NEEDS_AGENT":
        _logger.info(
            "[Genome] assembly %s is incomplete — needs reconstruction agent",
            state.assembly_id,
        )
        prompt = need.get("prompt_to_target_agent")

        # The Reconstruction Agent repairs one sequence record and refuses to
        # pick one itself - handing it only the assembly accession (GCF_...)
        # got every handoff rejected with "No sequence was given to
        # reconstruct". So the handoff is made satisfiable here: resolve the
        # species' largest RefSeq genomic record and publish its accession
        # under `sequence_accession`, the exact key ReconstructionRequest
        # .from_agent_request() reads. It rides in `output` because the Global
        # Orchestrator merges a worker's output into the shared context that
        # the next agent is called with.
        species = state.species or {}
        organism = species.get("scientific_name") or state.species_name
        scaffold = _find_scaffold(organism)
        if scaffold:
            output["sequence_accession"] = scaffold["accession"]
            prompt = (
                f"Genome assembly {state.assembly_id} for {organism} is at "
                f"'{(state.metadata or {}).get('assembly_level', 'draft')}' level. "
                f"Fill the gaps (runs of N) in {scaffold['accession']}, its largest "
                f"genomic scaffold ({scaffold.get('length_bp') or '?'} bp)."
            )
        else:
            _logger.info(
                "[Genome] no RefSeq genomic record found for %r; handing off without one",
                organism,
            )

        return AgentResult(
            status=AgentStatus.NEEDS_AGENT,
            target_agent=need.get("target_agent"),
            prompt_to_target_agent=prompt,
            output=output,
        )

    # Separate handoff: a *requested* protein_structure visualization that
    # this agent can't render itself. Mutually exclusive with the
    # reconstruction_need branch above - the graph only reaches
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
        result = to_result(state)

        # Break the escalation loop. A scaffold-level assembly always escalates
        # to the Reconstruction Agent - but reconstruction cannot change what
        # NCBI holds, so when the workflow comes back to this agent the level
        # is still "Scaffold" and it would escalate again, forever. The
        # orchestrator merges the Reconstruction Agent's own output keys into
        # the shared context, so their presence is the proof it already ran:
        # in that case the draft-assembly answer this agent computed is the
        # final one, and it is returned as COMPLETED instead of a second
        # identical handoff (which showed up to the user as a spurious
        # "Genome - Failed" after the planner refused the loop).
        context = request.context or {}
        already_reconstructed = any(
            key in context for key in ("reconstruction", "reconstruction_summary")
        )
        if (
            result.status is AgentStatus.NEEDS_AGENT
            and result.target_agent == "Reconstruction Agent"
            and already_reconstructed
        ):
            _logger.info(
                "[Genome] reconstruction already ran for this workflow; "
                "answering from the draft assembly instead of re-escalating"
            )
            output = result.output if isinstance(result.output, dict) else {}
            output.setdefault(
                "reconstruction_note",
                "The Reconstruction Agent already examined this assembly's "
                "largest scaffold in this workflow; figures below are from "
                "the draft assembly.",
            )
            return AgentResult(status=AgentStatus.COMPLETED, output=output)

        return result
