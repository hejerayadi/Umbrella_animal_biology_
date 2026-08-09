from schemas.inputs import GeneMapperInput
from schemas.outputs import GeneMapperOutput, GOAnnotation
from schemas.common import AgentStatus

_MOCK_GO_DB = {
    "FGF5": GOAnnotation(gene_symbol="FGF5", go_id="GO:0031069", go_name="hair follicle development"),
    "KRT71": GOAnnotation(gene_symbol="KRT71", go_id="GO:0031069", go_name="hair follicle development"),
    "HR": GOAnnotation(gene_symbol="HR", go_id="GO:0042633", go_name="hair cycle"),
    "TRPV3": GOAnnotation(gene_symbol="TRPV3", go_id="GO:0050977", go_name="sensory perception of touch"),
    "UCP1": GOAnnotation(gene_symbol="UCP1", go_id="GO:0009408", go_name="response to heat"),
}

# Gene symbols are cased by species convention, not by identity: the same gene
# is UCP1 in human and Ucp1 in mouse. The keys above are human-style, and the
# Genome Agent returns whatever NCBI holds for the species asked about - so a
# live mouse query sends "Ucp1" and an exact-match lookup misses every time.
# Matching case-insensitively is what makes the two agents interoperate; the
# canonical casing from the table above is what gets reported back.
_GO_DB_BY_UPPER = {symbol.upper(): annotation for symbol, annotation in _MOCK_GO_DB.items()}


async def mock_gene_mapper(input: GeneMapperInput) -> GeneMapperOutput:
    annotations, unmatched = [], []
    for gene in input.gene_list:
        annotation = _GO_DB_BY_UPPER.get(str(gene).strip().upper())
        if annotation is not None:
            annotations.append(annotation)
        else:
            unmatched.append(gene)  # flagged, never silently dropped

    # Fails only when NOTHING resolved. An unmatched gene alongside matched
    # ones is reported in `unmatched_genes`, not treated as a failure.
    #
    # This used to be `if not annotations or unmatched`, which failed the whole
    # step on a single unknown symbol. That was survivable while the Genome
    # Agent returned a fixed three-gene list chosen to match the mock GO
    # database below. It stopped being survivable once that agent started
    # returning real NCBI genes: a live query for mouse comes back with fifty
    # symbols, none of which are in this five-entry mock, so every real request
    # failed outright.
    #
    # Partial tolerance also matches the rest of this workflow - the pathway
    # and protein steps already report their misses and carry on. Gene Mapper
    # was the only strict one.
    if not annotations:
        status = AgentStatus.FAILED
    else:
        status = AgentStatus.COMPLETED

    return GeneMapperOutput(status=status, go_annotations=annotations, unmatched_genes=unmatched)