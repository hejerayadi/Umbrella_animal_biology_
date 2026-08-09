"""The Grand Orchestrator's entry point into the real protein workflow.

`POST /execute` is the contract every agent in this repository speaks: an
`{instruction, context}` body in, an `{status, target_agent,
prompt_to_target_agent, output}` body out. This module is the only thing
standing between it and `ProteinOrchestrator` - it translates, it does not
decide. Every scientific judgement stays in the LangGraph workflow and in
`result_policy.to_agent_result`.

Two translations are genuinely needed, and neither is guesswork:

* The shared context names a species the way the user wrote it, while every
  provider downstream is keyed by NCBI taxonomy id, so the name is resolved
  against UniProt's taxonomy (`TaxonomyCapability`).
* The workflow answers with the full `ProteinAnalysisResponse`. Only a summary
  goes back into the shared context - see `_shared_context_output`.
"""

import logging
from hashlib import blake2b
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from backend.agents.Protein_visualization.app.api.v1.dependencies import (
    get_orchestrator,
    get_taxonomy_capability,
)
from backend.agents.Protein_visualization.app.capabilities.taxonomy import TaxonomyCapability
from backend.agents.Protein_visualization.app.contracts.agent_result import AgentResult, AgentStatus
from backend.agents.Protein_visualization.app.contracts.agent_task import (
    AgentTask,
    ProteinTaskInput,
    SpeciesContract,
)
from backend.agents.Protein_visualization.app.contracts.protein_response import ProteinAnalysisResponse
from backend.agents.Protein_visualization.app.domain.enums import PreferredSource
from backend.agents.Protein_visualization.app.domain.exceptions import ProteinAgentError
from backend.agents.Protein_visualization.app.domain.models import SpeciesRef
from backend.agents.Protein_visualization.app.observability.context import log_context
from backend.agents.Protein_visualization.app.observability.logging import log_event
from backend.agents.Protein_visualization.app.orchestrators.protein.orchestrator import ProteinOrchestrator
from backend.agents.Protein_visualization.app.orchestrators.protein.result_policy import to_agent_result

router = APIRouter(tags=["inter-agent"])

logger = logging.getLogger("app.execute")

# Context keys this agent reads. Several spellings per fact because the key that
# lands in the shared context depends on who put it there: the orchestrator's
# extractor seeds `species` and `gene_name` from the user's sentence, while an
# upstream agent that resolved an identity writes the explicit ones.
_SPECIES_KEYS = ("species", "scientific_name", "organism")
_GENE_KEYS = ("resolved_gene_id", "gene_name", "gene_symbol", "gene")
_ACCESSION_KEYS = ("uniprot_accession", "accession")
_SEQUENCE_KEYS = ("protein_sequence", "sequence")
_RESIDUE_KEYS = ("residue_position", "residue")
_MUTATION_KEYS = ("mutation", "variant")
_REGION_KEYS = ("requested_regions", "regions", "domains")

# What this agent asks for when it cannot start. The Grand Orchestrator ignores
# `target_agent` and routes on the prompt alone (its Capability Resolver picks
# the helper), so these read as requests for information, not as addresses.
_IDENTITY_PROMPT = (
    "Identify the gene whose protein product should be modelled for {species}, and return its "
    "gene symbol - or, better, the canonical UniProt accession. The Protein Agent cannot search "
    "for a structure until it knows which protein the question is about."
)
_SPECIES_PROMPT = (
    "Name the species this question concerns, as a scientific binomial where possible. The "
    "Protein Agent resolves structures per organism and has no species to work from."
)


class AgentRequest(BaseModel):
    """The body every worker agent receives from the Grand Orchestrator."""

    instruction: str
    context: dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    """The body every worker agent answers with."""

    status: str
    target_agent: str | None = None
    prompt_to_target_agent: str | None = None
    output: Any | None = None


def _first(context: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = context.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _text(context: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    value = _first(context, keys)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _positive_int(context: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    value = _first(context, keys)
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _regions(context: dict[str, Any]) -> list[str]:
    value = _first(context, _REGION_KEYS)
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


def _species_hint(context: dict[str, Any]) -> tuple[str | None, SpeciesRef | None]:
    """The species as the context carries it: a name, or an already-resolved pair.

    An upstream agent that has done its own taxonomy work writes a dict, which
    is worth trusting - re-resolving it could disagree with the id it already
    handed to other agents.
    """
    value = _first(context, _SPECIES_KEYS)
    if isinstance(value, dict):
        name = value.get("scientific_name") or value.get("name")
        taxon_id = value.get("taxon_id") or value.get("taxonId")
        if name and taxon_id:
            try:
                return str(name), SpeciesRef(scientific_name=str(name), taxon_id=int(taxon_id))
            except (TypeError, ValueError):
                return str(name), None
        return (str(name) if name else None), None
    if isinstance(value, str) and value.strip():
        return value.strip(), None
    return None, None


def _idempotency_key(gene: str, taxon_id: int, accession: str | None, residue: int | None) -> str:
    """Stable across retries of the same question, distinct across different ones.

    The Grand Orchestrator re-runs an agent on a `continue` status without
    carrying an idempotency key of its own, so deriving one from the scientific
    inputs is what makes a retry recognisable as the same piece of work.
    """
    material = f"{gene}|{taxon_id}|{accession or ''}|{residue or ''}".casefold()
    return f"protein-{blake2b(material.encode(), digest_size=16).hexdigest()}"


def _needs_agent(prompt: str) -> AgentResult:
    return AgentResult(
        status=AgentStatus.NEEDS_AGENT,
        # The Capability Resolver overrides this; it is recorded so the reason
        # for the hand-off is still legible in a log or a stored result.
        target_agent="Genome",
        prompt_to_target_agent=prompt,
        output={},
    )


def _shared_context_output(response: ProteinAnalysisResponse) -> dict[str, Any]:
    """The summary that goes back into the orchestrator's shared context.

    Deliberately not `response.model_dump()`. Whatever an agent returns under
    `completed` is merged into the context every later agent sees, and the
    Responder renders every context key into an LLM prompt - so the full
    payload (Mol* scene, every annotation, every evidence record) would cost
    thousands of tokens per turn and drown the answer. The complete response
    stays available on `POST /api/v1/protein-structure-analyses`, which is what
    the frontend viewer calls.
    """
    structure = response.selected_structure
    protein = response.protein
    return {
        "protein_structure": {
            "analysis_id": str(response.analysis_id),
            "status": response.status.value,
            "validation_status": response.validation_status.value,
            "uniprot_accession": protein.uniprot_accession if protein else None,
            "gene_symbol": protein.gene_symbol if protein else None,
            "scientific_name": protein.scientific_name if protein else None,
            "structure": (
                {
                    "source": structure.source,
                    "external_id": structure.external_id,
                    "structure_type": structure.structure_type,
                    "experimental_method": structure.experimental_method,
                    "resolution_angstrom": structure.resolution_angstrom,
                    "mean_plddt": structure.mean_plddt,
                    "sequence_coverage": structure.sequence_coverage,
                    "file_url": structure.file_url,
                }
                if structure
                else None
            ),
            "explanation": response.explanation.summary if response.explanation else None,
            "annotation_count": len(response.annotations),
            "warnings": response.warnings,
        }
    }


async def _build_task(request: AgentRequest, taxonomy: TaxonomyCapability) -> AgentTask | AgentResult:
    """Turn the shared context into a task, or say what is missing and why."""
    context = request.context

    species_name, resolved = _species_hint(context)
    if not species_name:
        return _needs_agent(_SPECIES_PROMPT)

    gene = _text(context, _GENE_KEYS)
    accession = _text(context, _ACCESSION_KEYS)
    sequence = _text(context, _SEQUENCE_KEYS)
    if not gene and not accession and not sequence:
        return _needs_agent(_IDENTITY_PROMPT.format(species=species_name))

    species = resolved or await taxonomy.resolve(species_name)

    # `resolved_gene_id` is required by the contract, but an accession or a raw
    # sequence identifies the protein just as well; either one stands in so a
    # caller that already knows the accession is not forced to invent a symbol.
    gene_id = gene or accession or "sequence-supplied"

    return AgentTask(
        task_id=uuid4(),
        trace_id=uuid4(),
        source_agent="umbrella-main-orchestrator",
        idempotency_key=_idempotency_key(
            gene_id, species.taxon_id, accession, _positive_int(context, _RESIDUE_KEYS)
        ),
        input=ProteinTaskInput(
            resolved_gene_id=gene_id,
            species=SpeciesContract(scientific_name=species.scientific_name, taxon_id=species.taxon_id),
            uniprot_accession=accession,
            protein_sequence=sequence,
            requested_regions=_regions(context),
            residue_position=_positive_int(context, _RESIDUE_KEYS),
            mutation=_text(context, _MUTATION_KEYS),
            preferred_source=PreferredSource.auto,
            include_explanation=True,
        ),
    )


@router.post("/execute", response_model=AgentResponse)
async def execute(
    request: AgentRequest,
    orchestrator: Annotated[ProteinOrchestrator, Depends(get_orchestrator)],
    taxonomy: Annotated[TaxonomyCapability, Depends(get_taxonomy_capability)],
) -> AgentResponse:
    """Run the real workflow for one orchestrator request.

    Never raises. The Grand Orchestrator's router expects one schema back every
    time and already knows how to handle a `failed` status; a 500 carrying
    FastAPI's `{"detail": ...}` would break that contract instead of being
    handled.
    """
    try:
        task = await _build_task(request, taxonomy)

        if isinstance(task, AgentResult):
            log_event(logger, "execute.needs_input", prompt=task.prompt_to_target_agent)
            return _respond(task)

        with log_context(task_id=str(task.task_id), trace_id=str(task.trace_id)):
            log_event(
                logger,
                "execute.started",
                gene=task.input.resolved_gene_id,
                taxon_id=task.input.species.taxon_id,
            )
            response = await orchestrator.analyze(task)

        result = to_agent_result(response, task)
        log_event(
            logger,
            "execute.finished",
            agent_status=result.status.value,
            analysis_status=response.status.value,
            validation_status=response.validation_status.value,
        )
        return _respond(result, response)

    except ProteinAgentError as exc:
        log_event(logger, "execute.failed", logging.WARNING, error_code=type(exc).__name__)
        return AgentResponse(status=AgentStatus.FAILED.value, output=f"Protein Agent: {exc}")
    except Exception as exc:  # noqa: BLE001 - the boundary must not leak exceptions
        log_event(logger, "execute.failed", logging.ERROR, exc_info=True, error_code=type(exc).__name__)
        return AgentResponse(
            status=AgentStatus.FAILED.value,
            output=f"Protein Agent error: {type(exc).__name__}: {exc}",
        )


def _respond(result: AgentResult, response: ProteinAnalysisResponse | None = None) -> AgentResponse:
    """Shape one `AgentResult` for the wire.

    `output` carries the summary on the paths where the orchestrator merges it
    into the shared context, and the reason otherwise - a bare `{}` would tell
    the Responder nothing about why the agent stopped.
    """
    if response is not None and result.status is AgentStatus.COMPLETED:
        output: Any = _shared_context_output(response)
    elif response is not None:
        output = {
            **_shared_context_output(response),
            "reason": result.continuation_reason or result.error,
            "retryable": result.retryable,
        }
    else:
        output = {"reason": result.prompt_to_target_agent or result.error}

    return AgentResponse(
        status=result.status.value,
        target_agent=result.target_agent,
        prompt_to_target_agent=result.prompt_to_target_agent,
        output=output,
    )
