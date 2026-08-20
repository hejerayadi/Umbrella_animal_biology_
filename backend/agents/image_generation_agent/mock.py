from .schema import AgentRequest, AgentResult, AgentStatus


class ImageGenerationMock:

    def run(self, request: AgentRequest) -> AgentResult:

        species = request.context.get("species")

        if species is None:

            return AgentResult(
                status=AgentStatus.FAILED,
                output="No species provided."
            )

        if "traits" not in request.context:

            return AgentResult(
                status=AgentStatus.NEEDS_AGENT,
                target_agent="Trait",
                prompt_to_target_agent="""
Discover the genes associated with the observable physical traits of this
species - body shape, size, fur and colour - so its appearance can be drawn.
"""
            )

        if "biodiversity_report" not in request.context:

            return AgentResult(
                status=AgentStatus.NEEDS_AGENT,
                target_agent="Biodiversity",
                prompt_to_target_agent="""
Provide the habitat and conservation information of this species, to set the
scene of the image.
"""
            )

        return AgentResult(
            status=AgentStatus.COMPLETED,
            output={
                "generated_image": f"Generated image of {species}"
            }
        )
