from schemas.inputs import LiteratureSupportInput
from schemas.outputs import LiteratureSupportOutput, LiteratureRecord
from schemas.common import AgentStatus

_MOCK_LITERATURE_DB = {
    "fur growth": [
        LiteratureRecord(pmid="18239092", title="FGF5 and hair cycle regulation", year=2008,
                          short_summary="Links FGF5 mutation to hair length in mammals."),
        LiteratureRecord(pmid="30112233", title="Follicle regulatory network in rodents", year=2019,
                          short_summary="Broader gene network context around FGF5 signaling."),
    ],
    "cold adaptation": [
        LiteratureRecord(pmid="26123456", title="UCP1 and non-shivering thermogenesis", year=2016,
                          short_summary="UCP1 role in brown fat heat production."),
    ],
}

# 1 or fewer records is considered thin and triggers an escalation for deeper evidence
_THIN_EVIDENCE_THRESHOLD = 1

# The real Literature Agent can search any trait; this mock holds two. Without a
# fallback, every trait outside those two dead-ends in FAILED - which means a
# genuine user question ("tusk size", "body mass") can never reach the end of the
# pipeline, and the orchestration path stays untestable for anything but the two
# magic strings. Standing in with a known-good record set keeps the control flow
# exercisable. Delete this the moment real literature retrieval lands.
_FALLBACK_TRAIT = "fur growth"


async def mock_literature_support(input: LiteratureSupportInput) -> LiteratureSupportOutput:
    evidence = _MOCK_LITERATURE_DB.get(input.trait_name.lower())

    if evidence is None:
        evidence = _MOCK_LITERATURE_DB[_FALLBACK_TRAIT]

    if not evidence:
        return LiteratureSupportOutput(status=AgentStatus.FAILED, evidence=[])

    # An escalation has to be answerable, or it is not an escalation - it is a
    # loop. When this node asks for a deeper literature search, the orchestrator
    # runs the Literature Agent and merges its papers into the shared context,
    # then resumes this workflow from the top. Without this check the same thin
    # lookup runs again, escalates again, and the two agents ping-pong until the
    # graph's recursion limit trips.
    #
    # `papers` is the Literature Agent's declared output key (see its card.json),
    # so its presence means the request this node would raise has already been
    # satisfied upstream.
    already_answered = bool((input.context or {}).get("papers"))

    if len(evidence) <= _THIN_EVIDENCE_THRESHOLD and not already_answered:
        return LiteratureSupportOutput(
            status=AgentStatus.NEEDS_AGENT,
            evidence=evidence,
            target_agent="Literature Agent",
            prompt_to_target_agent=(
                f"Find additional peer-reviewed evidence for trait '{input.trait_name}' "
                f"and genes {input.gene_list}. Existing evidence is thin "
                f"({len(evidence)} record(s))."
            ),
        )

    return LiteratureSupportOutput(status=AgentStatus.COMPLETED, evidence=evidence)