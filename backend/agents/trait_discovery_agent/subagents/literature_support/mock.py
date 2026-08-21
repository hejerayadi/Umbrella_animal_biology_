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


async def mock_literature_support(input: LiteratureSupportInput) -> LiteratureSupportOutput:
    evidence = _MOCK_LITERATURE_DB.get(input.trait_name.lower(), [])

    if not evidence:
        return LiteratureSupportOutput(status=AgentStatus.FAILED, evidence=[])

    if len(evidence) <= _THIN_EVIDENCE_THRESHOLD:
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
