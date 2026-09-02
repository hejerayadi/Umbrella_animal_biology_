from schemas.inputs import LiteratureSupportInput
from schemas.outputs import LiteratureSupportOutput, LiteratureRecord
from schemas.common import AgentStatus

# Keyed by (trait_name.lower(), gene_symbol.upper()) -- NOT trait_name alone.
#
# Keying on trait_name only meant every gene sharing a trait_name silently
# returned the *same* records regardless of which gene was actually asked
# about: KRT71 (trait "fur growth") got back FGF5's summaries verbatim, and
# PRDM16 (trait "cold adaptation") got back UCP1's. Since
# rag_evaluators._claim_as_passage prepends the *queried* gene as the
# sentence subject (e.g. "PRDM16: UCP1 role in brown fat heat production is
# reported."), the cross-encoder was correctly scoring that as low relevance
# -- the passage body is genuinely about the wrong gene. This wasn't a
# scoring bug, it was a data bug: one array shared across every gene under a
# trait instead of one entry per gene.
#
# Covers all 12 (gene, trait_name) pairs in evaluation/rag_test_cases.yaml.
# TRPV3 and ASIP are deliberately left with NO entries: both are tagged
# `zero_literature` in that yaml specifically to test that this returns
# status=FAILED (zero evidence, no fabricated citation) rather than
# escalating or inventing something -- see the notes on those two entries.
_MOCK_LITERATURE_DB: dict[tuple[str, str], list[LiteratureRecord]] = {
    ("fur growth", "FGF5"): [
        LiteratureRecord(
            pmid="18239092", title="FGF5 and hair cycle regulation", year=2008,
            short_summary=(
                "FGF5 loss-of-function mutations prolong the anagen (active "
                "growth) phase of the hair follicle cycle, producing a "
                "visibly longer-fur phenotype in mice, dogs, and other mammals."
            ),
        ),
        LiteratureRecord(
            pmid="30112233", title="Follicle regulatory network in rodents", year=2019,
            short_summary=(
                "FGF5 signals through the MAPK pathway to trigger catagen "
                "(the regression phase), acting as the molecular brake that "
                "normally limits how long fur keeps growing."
            ),
        ),
    ],
    ("fur growth", "KRT71"): [
        LiteratureRecord(
            pmid="21234567", title="KRT71 mutations and the curly-coat phenotype", year=2009,
            short_summary=(
                "KRT71 mutations disrupt inner root sheath keratinization "
                "during hair shaft formation, altering fur texture and "
                "growth in mouse and canine coat-pattern models."
            ),
        ),
    ],
    ("cold adaptation", "UCP1"): [
        LiteratureRecord(
            pmid="26123456", title="UCP1 and non-shivering thermogenesis", year=2016,
            short_summary=(
                "UCP1 uncouples mitochondrial respiration in brown adipose "
                "tissue to generate heat directly, driving non-shivering "
                "cold adaptation in small mammals."
            ),
        ),
    ],
    ("cold adaptation", "PRDM16"): [
        LiteratureRecord(
            pmid="19556183", title="PRDM16 controls brown fat cell fate", year=2009,
            short_summary=(
                "PRDM16 is the transcriptional switch that commits precursor "
                "cells to a brown-fat identity, upstream of UCP1, enabling "
                "the cold-adaptive thermogenic program."
            ),
        ),
    ],
    ("tumor suppression", "TP53"): [
        LiteratureRecord(
            pmid="10723097", title="p53 as guardian of the genome", year=2000,
            short_summary=(
                "TP53 triggers cell-cycle arrest or apoptosis in response to "
                "DNA damage, making it the central tumor suppressor "
                "safeguarding genome integrity."
            ),
        ),
        LiteratureRecord(
            pmid="17554376", title="p53 mutation spectrum in human cancer", year=2007,
            short_summary=(
                "Loss-of-function TP53 mutations are among the most common "
                "events that disable tumor suppression across human cancer types."
            ),
        ),
    ],
    ("dna damage response", "BRCA1"): [
        LiteratureRecord(
            pmid="10521280", title="BRCA1 in homologous recombination repair", year=1999,
            short_summary=(
                "BRCA1 promotes homologous recombination repair of DNA "
                "double-strand breaks, a core arm of the DNA damage response."
            ),
        ),
    ],
    ("hair cycle regulation", "HR"): [
        LiteratureRecord(
            pmid="9843205", title="Hairless gene and catagen progression", year=1998,
            short_summary=(
                "The HR (hairless) transcriptional corepressor is required "
                "for the follicle to progress through catagen, the "
                "regression stage of the hair cycle."
            ),
        ),
    ],
    ("coat color", "MC1R"): [
        LiteratureRecord(
            pmid="8290042", title="MC1R variants and pigment-type switching", year=1995,
            short_summary=(
                "MC1R activation by alpha-MSH shifts melanocytes toward "
                "eumelanin production, and MC1R loss-of-function variants "
                "underlie red/yellow coat color."
            ),
        ),
    ],
    ("immune regulation", "MARCH1"): [
        LiteratureRecord(
            pmid="16697113", title="MARCH1 ubiquitinates MHC class II", year=2006,
            short_summary=(
                "MARCH1 is an E3 ubiquitin ligase that downregulates MHC "
                "class II surface expression on antigen-presenting cells, a "
                "key immune regulation checkpoint."
            ),
        ),
    ],
    ("cell growth regulation", "HRAS"): [
        LiteratureRecord(
            pmid="6093169", title="HRAS oncogene and constitutive proliferation", year=1982,
            short_summary=(
                "Activating HRAS mutations lock Ras-MAPK signaling in the "
                "'on' state, driving unchecked cell growth regulation escape "
                "in tumors."
            ),
        ),
    ],
    # Deliberately no entry for ("thermal sensation", "TRPV3") or
    # ("coat color agouti pattern", "ASIP") -- see module docstring.
}

# 1 or fewer records is considered thin and triggers an escalation for deeper evidence
_THIN_EVIDENCE_THRESHOLD = 1


async def mock_literature_support(input: LiteratureSupportInput) -> LiteratureSupportOutput:
    trait_key = input.trait_name.lower()

    # Aggregate per-gene, deduping by pmid, rather than looking up one shared
    # array for the whole trait -- each gene in gene_list now only pulls in
    # evidence keyed to *that* gene (see module docstring for why).
    seen_pmids: set[str] = set()
    evidence: list[LiteratureRecord] = []
    for gene in input.gene_list:
        for record in _MOCK_LITERATURE_DB.get((trait_key, gene.upper()), []):
            if record.pmid in seen_pmids:
                continue
            seen_pmids.add(record.pmid)
            evidence.append(record)

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
