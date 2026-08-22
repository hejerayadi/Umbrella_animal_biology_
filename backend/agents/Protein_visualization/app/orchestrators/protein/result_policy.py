"""Map a scientific protein analysis to the inter-agent routing contract."""

from backend.agents.Protein_visualization.app.contracts.agent_result import AgentResult, AgentStatus
from backend.agents.Protein_visualization.app.contracts.protein_request import ProteinAnalysisRequest
from backend.agents.Protein_visualization.app.contracts.protein_response import ProteinAnalysisResponse
from backend.agents.Protein_visualization.app.domain.enums import AnalysisStatus, ValidationStatus


def _has_warning(response: ProteinAnalysisResponse, *prefixes: str) -> bool:
    return any(warning.startswith(prefixes) for warning in response.warnings)


def _identity_prompt(response: ProteinAnalysisResponse, task: ProteinAnalysisRequest | None) -> str:
    gene = (
        response.protein.gene_symbol
        if response.protein
        else task.input.resolved_gene_id
        if task
        else "the requested gene"
    )
    species = (
        response.protein.scientific_name
        if response.protein
        else task.input.species.scientific_name
        if task
        else "the requested species"
    )
    return (
        f"Resolve the canonical protein identity for {gene} in {species}. "
        "Return the canonical UniProt accession or protein sequence, the taxonomy identifier, "
        "and evidence supporting the mapping so the Protein Agent can resume."
    )


def _literature_prompt(response: ProteinAnalysisResponse, task: ProteinAnalysisRequest | None) -> str:
    identifier = (
        response.protein.uniprot_accession
        if response.protein
        else task.input.uniprot_accession or task.input.resolved_gene_id
        if task
        else "the requested protein"
    )
    return (
        f"Search the scientific literature for structural evidence about {identifier}. "
        "Return experimental structure identifiers when available, otherwise report validated "
        "predicted models and cite the supporting sources."
    )


def to_agent_result(
    response: ProteinAnalysisResponse,
    task: ProteinAnalysisRequest | None = None,
) -> AgentResult:
    """Apply deterministic, mutually exclusive hand-off rules.

    Rules, in priority order:

    * invalid or incomplete input -> CONTINUE after caller clarification;
    * temporary UniProt/infrastructure failure -> CONTINUE with the same agent;
    * unresolved identity -> NEEDS_AGENT from ``genome_agent``;
    * no usable structural evidence / scientific abstention -> NEEDS_AGENT from
      ``literature_agent``;
    * usable COMPLETED or PARTIAL result -> COMPLETED at the agent-routing level;
    * any other terminal failure -> FAILED.
    """
    output = response.model_dump(mode="json")

    if response.status is AnalysisStatus.needs_clarification:
        return AgentResult(
            status=AgentStatus.CONTINUE,
            output=output,
            continuation_reason=(
                "The Grand Orchestrator must obtain corrected or missing input from the caller "
                "and then resubmit the Protein Agent task."
            ),
        )

    if _has_warning(response, "UNIPROT_TIMEOUT", "UNIPROT_UNAVAILABLE"):
        return AgentResult(
            status=AgentStatus.CONTINUE,
            output=output,
            continuation_reason=(
                "Protein identity resolution failed temporarily; retry this Protein Agent task "
                "with the same correlation and idempotency context."
            ),
            retryable=True,
        )

    if _has_warning(response, "IDENTITY_UNRESOLVED"):
        return AgentResult(
            status=AgentStatus.NEEDS_AGENT,
            target_agent="genome_agent",
            prompt_to_target_agent=_identity_prompt(response, task),
            output=output,
        )

    if response.selected_structure is None and _has_warning(
        response,
        "PDB_TIMEOUT",
        "PDB_UNAVAILABLE",
        "ALPHAFOLD_TIMEOUT",
        "ALPHAFOLD_UNAVAILABLE",
    ):
        return AgentResult(
            status=AgentStatus.CONTINUE,
            output=output,
            continuation_reason=(
                "A structural provider failed temporarily before a usable model was found; "
                "retry this Protein Agent task before delegating it to another specialist."
            ),
            retryable=True,
        )

    needs_structural_evidence = response.status in {
        AnalysisStatus.no_structure_found,
        AnalysisStatus.needs_agent,
    } or (response.validation_status is ValidationStatus.abstain and response.protein is not None)
    if needs_structural_evidence:
        return AgentResult(
            status=AgentStatus.NEEDS_AGENT,
            target_agent="literature_agent",
            prompt_to_target_agent=_literature_prompt(response, task),
            output=output,
        )

    if (
        response.status in {AnalysisStatus.completed, AnalysisStatus.partial}
        and response.validation_status in {ValidationStatus.accept, ValidationStatus.revise}
        and response.selected_structure is not None
    ):
        return AgentResult(status=AgentStatus.COMPLETED, output=output)

    if response.status in {
        AnalysisStatus.received,
        AnalysisStatus.validating,
        AnalysisStatus.running,
    }:
        return AgentResult(
            status=AgentStatus.CONTINUE,
            output=output,
            continuation_reason="The Protein Agent workflow has not reached a terminal node yet.",
            retryable=True,
        )

    return AgentResult(
        status=AgentStatus.FAILED,
        output=output,
        error="The Protein Agent ended without a usable result or a safe specialist hand-off.",
    )
