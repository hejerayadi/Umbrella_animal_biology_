from .schema import AgentRequest, AgentResult, AgentStatus

# "Gene retrieval" is already one of this agent's advertised capabilities in
# card.json, but only the genome string was ever returned - so an agent that
# asked for genes (the Trait Discovery Agent does) got a genome back, escalated
# again, and looped. Returning a gene list closes that.
#
# Fixed rather than derived, and returned on every call: the orchestrator sends
# the user's original question plus the shared context, never the requesting
# agent's prompt, so this agent cannot tell who is asking or what for. Until it
# can, answering the same way every time is the honest behaviour.
#
# These three symbols are not arbitrary. The Trait Discovery Agent's Gene Mapper
# fails the whole step if ANY gene is unmatched, and its mock GO database knows
# only FGF5, KRT71, HR, TRPV3 and UCP1. Changing this list without checking that
# one will break the trait pipeline downstream.
_MOCK_GENE_LIST = ["FGF5", "KRT71", "HR"]


class GenomeMock:

    def run(self, request: AgentRequest) -> AgentResult:

        species = request.context.get("species")

        if species is None:
            return AgentResult(
                status=AgentStatus.FAILED,
                output="No species provided."
            )

        return AgentResult(
            status=AgentStatus.COMPLETED,
            output={
                "genome": f"Genome sequence of {species}",
                # Copied, not shared: this dict is merged into the orchestrator's
                # shared context, and handing out the module-level list would let
                # any later agent mutate it for every subsequent request.
                "gene_list": list(_MOCK_GENE_LIST),
            }
        )