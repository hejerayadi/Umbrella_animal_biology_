import asyncio

from schemas.inputs import TraitDiscoveryInput, GeneMapperInput, FunctionalEvidenceInput, LiteratureSupportInput
from schemas.outputs import TraitDiscoveryOutput
from schemas.common import AgentStatus


class TraitDiscoveryOrchestrator:
    def __init__(self, gene_mapper, functional_evidence_orchestrator, literature_support):
        self.gene_mapper = gene_mapper
        self.functional_evidence_orchestrator = functional_evidence_orchestrator
        self.literature_support = literature_support

    async def run(self, input: TraitDiscoveryInput) -> TraitDiscoveryOutput:

        # Step 0 — gene_list is a Genome Agent cross-agent dependency, no longer a
        # direct field on TraitDiscoveryInput. Only the Global Scientific Orchestrator
        # can call the Genome Agent, so this orchestrator looks in context first
        # (populated on a prior turn) and escalates if it's not there yet.


        gene_list = input.context.get("gene_list")
        if not gene_list:
            return TraitDiscoveryOutput(
                status=AgentStatus.NEEDS_AGENT,
                target_agent="Genome Agent",
                prompt_to_target_agent=(
                    f"Resolve candidate genes for trait '{input.trait_name}' "
                    f"in species '{input.species_name}'."
                ),
            )

        # Step 1 — Gene Mapper (sequential: downstream steps need its GO annotations)


        gene_mapper_out = await self.gene_mapper(GeneMapperInput(
            trait_name=input.trait_name,
            gene_list=gene_list,
            species_name=input.species_name,
            instruction=input.instruction,
            context=input.context,
        ))

        if gene_mapper_out.status == AgentStatus.FAILED:
            return TraitDiscoveryOutput(status=AgentStatus.FAILED)

        # Step 2 — Functional Evidence (sub-orchestrator) + Literature Support, in parallel


        functional_evidence_in = FunctionalEvidenceInput(
            gene_list=gene_list,
            go_annotations=gene_mapper_out.go_annotations,
            instruction=input.instruction,
            context=input.context,
        )
        literature_in = LiteratureSupportInput(
            trait_name=input.trait_name,
            gene_list=gene_list,
            instruction=input.instruction,
            context=input.context,
        )

        functional_evidence_out, literature_out = await asyncio.gather(
            self.functional_evidence_orchestrator.run(functional_evidence_in),
            self.literature_support(literature_in),
        )

        explanation = self._build_explanation(
            input.trait_name, gene_mapper_out, functional_evidence_out, literature_out
        )


        # Step 3 — bubble Literature Support's own escalation upward, untouched.
        # This agent doesn't resolve it — it forwards to whoever called it, per
        # the card: "status=NEEDS_AGENT with target_agent='Literature Agent'
        # when evidence is thin."


        if literature_out.status == AgentStatus.NEEDS_AGENT:
            return TraitDiscoveryOutput(
                status=AgentStatus.NEEDS_AGENT,
                go_annotations=gene_mapper_out.go_annotations,
                pathway_data=functional_evidence_out.pathway_data,
                protein_data=functional_evidence_out.protein_data,
                evidence=literature_out.evidence,
                explanation=explanation,
                target_agent=literature_out.target_agent,
                prompt_to_target_agent=literature_out.prompt_to_target_agent,
            )

        # Step 4 — degrade gracefully rather than hard-failing on a single partial gap
        if functional_evidence_out.status == AgentStatus.FAILED and literature_out.status == AgentStatus.FAILED:
            return TraitDiscoveryOutput(status=AgentStatus.FAILED)

        return TraitDiscoveryOutput(
            status=AgentStatus.COMPLETED,
            go_annotations=gene_mapper_out.go_annotations,
            pathway_data=functional_evidence_out.pathway_data,
            protein_data=functional_evidence_out.protein_data,
            evidence=literature_out.evidence,
            explanation=explanation,
        )

    @staticmethod
    def _build_explanation(trait_name, gene_mapper_out, functional_evidence_out, literature_out) -> str:
        genes = ", ".join(a.gene_symbol for a in gene_mapper_out.go_annotations) or "no genes matched"
        pathways = ", ".join(p.pathway_name for p in functional_evidence_out.pathway_data) or "no pathways found"
        return (
            f"Trait '{trait_name}' is associated with genes: {genes}. "
            f"Relevant pathways: {pathways}. "
            f"{len(literature_out.evidence)} supporting literature reference(s) found."
        )