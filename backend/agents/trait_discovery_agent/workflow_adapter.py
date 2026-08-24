"""Serves the real LangGraph trait-discovery workflow behind the HTTP boundary.

`workflows/` was written to be driven from its own CLI (`python -m workflows`),
where the caller already knows the trait, the species and a gene list. The
Global Orchestrator knows none of those as named arguments - it sends one
`instruction` string and the flat `context` dict every agent shares. This
module is the translation between the two, in both directions:

  inbound   instruction + context  ->  TraitDiscoveryState(trait_name,
                                       species_name, instruction, context)
                                       with the identifiers `context_bridge`
                                       resolves

  outbound  final graph state      ->  AgentResult the orchestrator understands

Findings are published under `traits`, which is not a free choice: it is this
agent's declared output key in `card.json`, the key the Image Generation agent
reads for the traits it draws (`orchestrator_logic.TRAIT_CONTEXT_KEYS`), and
the key the Multimodal agent waits for before it will resume
(`workflows/state.HELPER_OUTPUT_KEYS`). It is kept a plain list of readable
labels for exactly that reason; everything structured goes under
`trait_discovery` beside it, so the shared context does not gain generic names
like `status` or `explanation` that eight other agents also write.
"""
from __future__ import annotations

import logging
from typing import Any

from schemas.common import AgentStatus as WorkflowStatus
from workflows.state import TraitDiscoveryState
from workflows.trait_discovery_graph import build_trait_discovery_graph

from .context_bridge import prepare_workflow_context
from .schema import AgentRequest, AgentResult, AgentStatus

_logger = logging.getLogger(__name__)

# The workflow's internal capability resolver picks from `agent_cards/`, whose
# display names ("Genome Agent") are not the names the Global Orchestrator
# registers ("Genome" - see backend/registry.py). Its own resolver re-decides
# from `prompt_to_target_agent` and ignores this field, so a mismatch is not
# fatal, but `AgentResult.target_agent` is part of the published contract and
# other agents do validate against registry keys.
_REGISTRY_NAMES: dict[str, str] = {
    "genome agent": "Genome",
    "genome": "Genome",
    "literature agent": "Literature",
    "literature": "Literature",
    "biodiversity agent": "Biodiversity",
    "evolution agent": "Evolution",
    "protein visualization agent": "Protein",
    "protein structure visualization agent": "Protein",
    "reconstruction agent": "Reconstruction",
}


def _registry_name(target_agent: str | None) -> str | None:
    if not target_agent:
        return None
    return _REGISTRY_NAMES.get(target_agent.strip().lower(), target_agent)


def _species_name(context: dict[str, Any]) -> str:
    """The species this question is about, from whichever agent named it.

    `species` is what the orchestrator's extractor writes from the user's
    sentence and what the Multimodal agent publishes after recognising a photo;
    `species_record` is the Genome Agent's resolved NCBI record.
    """
    named = context.get("species")
    if isinstance(named, str) and named.strip():
        return named.strip()
    if isinstance(named, (list, tuple)) and named:
        return str(named[0]).strip()

    record = context.get("species_record")
    if isinstance(record, dict):
        for key in ("scientific_name", "name", "species", "common_name"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _trait_name(instruction: str, context: dict[str, Any]) -> tuple[str, bool]:
    """The trait under investigation, and whether it was actually named as one.

    `trait_name` is seeded by the orchestrator's extractor, which deliberately
    leaves it empty when the message names no specific characteristic ("what
    trait does FGF5 control?"). The workflow uses this string to judge which GO
    term, pathway and protein entry is most relevant, so an empty value would
    strip every relevance decision of its subject - the question itself is a
    better prompt than nothing.

    The flag matters because the question is only fit for that internal use.
    Published as a trait label it reads as a finding, and the Image Generation
    agent draws from those labels: it was handed "What trait does the FGF5 gene
    control in Mus musculus?" as something to illustrate.
    """
    named = context.get("trait_name")
    if isinstance(named, str) and named.strip():
        return named.strip(), True
    return instruction.strip(), False


def _trait_labels(
    trait_name: str,
    trait_was_named: bool,
    annotations: list[Any],
    pathways: list[Any],
) -> list[str]:
    """Readable one-line labels for the `traits` context key.

    Kept to short noun phrases because the Image Generation agent turns these
    straight into drawing instructions - a JSON blob there produces a caption,
    not a picture.
    """
    labels: list[str] = []
    if trait_name and trait_was_named:
        labels.append(trait_name)

    for annotation in annotations:
        go_name = getattr(annotation, "go_name", "")
        gene = getattr(annotation, "gene_symbol", "")
        if go_name and gene:
            labels.append(f"{go_name} ({gene})")
        elif go_name:
            labels.append(str(go_name))

    for pathway in pathways:
        name = getattr(pathway, "pathway_name", "")
        if name:
            labels.append(f"pathway: {name}")

    return labels


def _findings(
    state: dict[str, Any],
    trait_name: str,
    trait_was_named: bool,
    species_name: str,
    dropped: list[str],
) -> dict[str, Any]:
    """The structured payload published under `trait_discovery`."""
    annotations = state.get("go_annotations") or []
    pathways = state.get("pathway_data") or []
    proteins = state.get("protein_data") or []
    evidence = state.get("evidence") or []

    findings: dict[str, Any] = {
        "status": "completed",
        # Only a trait the caller actually named. When it is empty the run was
        # gene-first ("what does FGF5 do?") and the genes below are the answer.
        "trait_name": trait_name if trait_was_named else "",
        "species": species_name,
        "explanation": state.get("explanation") or "",
        "genes": [
            {
                "gene_symbol": getattr(item, "gene_symbol", ""),
                "go_id": getattr(item, "go_id", ""),
                "go_name": getattr(item, "go_name", ""),
            }
            for item in annotations
        ],
        "pathways": [
            {
                "pathway_id": getattr(item, "pathway_id", ""),
                "pathway_name": getattr(item, "pathway_name", ""),
                "reasoning": getattr(item, "reasoning", ""),
            }
            for item in pathways
        ],
        "proteins": [
            {
                "gene_symbol": getattr(item, "gene_symbol", ""),
                "protein_name": getattr(item, "protein_name", ""),
                "function_summary": getattr(item, "function_summary", ""),
                "source_accession": getattr(item, "source_accession", ""),
            }
            for item in proteins
        ],
        "evidence": [
            {
                "pmid": getattr(item, "pmid", ""),
                "title": getattr(item, "title", ""),
                "year": getattr(item, "year", None),
                "short_summary": getattr(item, "short_summary", ""),
            }
            for item in evidence
        ],
    }

    # Partial-failure detail the workflow deliberately surfaces rather than
    # swallowing (see the §8 notes in workflows/state.py). The Responder reads
    # a `warnings` list on any dict finding and is required to preserve it, so
    # a partial answer reaches the user labelled as partial.
    warnings: list[str] = []
    if state.get("unmatched_genes"):
        warnings.append(
            "No Gene Ontology annotation found for: "
            + ", ".join(str(gene) for gene in state["unmatched_genes"])
        )
    if state.get("missing_genes"):
        warnings.append(
            "No reviewed UniProt entry for: "
            + ", ".join(str(gene) for gene in state["missing_genes"])
        )
    if state.get("malformed_ids"):
        warnings.append(
            "No KEGG pathway link for: "
            + ", ".join(str(gene) for gene in state["malformed_ids"])
        )
    if dropped:
        warnings.append(
            f"{len(dropped)} further candidate gene(s) were not investigated in this run: "
            + ", ".join(dropped)
        )
    if warnings:
        findings["warnings"] = warnings

    return findings


class WorkflowTraitAgent:
    """Runs the compiled trait-discovery graph for one orchestrator request."""

    def __init__(self, graph: Any | None = None) -> None:
        # Compiled once at startup. Building the graph imports every subagent
        # and its LLM client, which is not work to repeat per request.
        self._graph = graph if graph is not None else build_trait_discovery_graph()

    async def run(self, request: AgentRequest) -> AgentResult:
        context = dict(request.context or {})
        instruction = request.instruction or ""

        species_name = _species_name(context)
        trait_name, trait_was_named = _trait_name(instruction, context)

        prepared, dropped = await prepare_workflow_context(context, species_name)

        _logger.info(
            "[Trait] running workflow trait=%r species=%r genes=%s",
            trait_name,
            species_name,
            prepared.get("gene_list"),
        )

        final = await self._graph.ainvoke(
            TraitDiscoveryState(
                trait_name=trait_name,
                species_name=species_name,
                instruction=instruction,
                context=prepared,
            )
        )

        return self._to_agent_result(
            final, trait_name, trait_was_named, species_name, dropped
        )

    def _to_agent_result(
        self,
        state: dict[str, Any],
        trait_name: str,
        trait_was_named: bool,
        species_name: str,
        dropped: list[str],
    ) -> AgentResult:
        status = state.get("status")
        status_value = status.value if isinstance(status, WorkflowStatus) else str(status)

        if status_value == WorkflowStatus.NEEDS_AGENT.value:
            prompt = state.get("prompt_to_target_agent") or (
                f"Resolve the candidate genes associated with '{trait_name}' "
                f"in {species_name or 'the species in question'}."
            )
            target = _registry_name(state.get("target_agent"))
            _logger.info("[Trait] needs %s -> %r", target, prompt)
            return AgentResult(
                status=AgentStatus.NEEDS_AGENT,
                target_agent=target,
                prompt_to_target_agent=prompt,
            )

        if status_value == WorkflowStatus.FAILED.value:
            # The graph reaches `failed` only from Gene Mapper resolving
            # nothing at all, so say which genes it tried rather than reporting
            # a bare failure the user cannot act on.
            unmatched = state.get("unmatched_genes") or []
            detail = (
                f" None of the candidate genes ({', '.join(str(g) for g in unmatched)}) "
                f"could be annotated against Gene Ontology."
                if unmatched
                else " No candidate genes could be annotated against Gene Ontology."
            )
            _logger.info("[Trait] failed: %s", detail.strip())
            return AgentResult(
                status=AgentStatus.FAILED,
                output=(
                    f"Trait discovery could not link '{trait_name}' to any gene in "
                    f"{species_name or 'this species'}.{detail}"
                ),
            )

        findings = _findings(state, trait_name, trait_was_named, species_name, dropped)
        labels = _trait_labels(
            trait_name,
            trait_was_named,
            state.get("go_annotations") or [],
            state.get("pathway_data") or [],
        )
        _logger.info("[Trait] completed with %d trait label(s)", len(labels))

        return AgentResult(
            status=AgentStatus.COMPLETED,
            output={"traits": labels, "trait_discovery": findings},
        )
