
from __future__ import annotations

from schemas.outputs import GOAnnotation, PathwayEntry, ProteinEntry


class CallLog:
    """Records every call made to a faked external client so a test can assert
    the real network/API boundary was (or was not) hit."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __len__(self) -> int:
        return len(self.calls)

    def record(self, *args) -> None:
        self.calls.append(args)


# ---------------------------------------------------------------------------
# Canned "server" data, keyed the same way the real clients would be keyed.
# ---------------------------------------------------------------------------
GO_DB: dict[str, GOAnnotation] = {
    "FGF5": GOAnnotation(gene_symbol="FGF5", go_id="GO:0031069", go_name="hair follicle development"),
    "UCP1": GOAnnotation(gene_symbol="UCP1", go_id="GO:0009408", go_name="response to heat"),
}

KEGG_DB: dict[str, PathwayEntry] = {
    "hsa:FGF5": PathwayEntry(pathway_id="hsa04010", pathway_name="MAPK signaling pathway"),
    "hsa:UCP1": PathwayEntry(pathway_id="hsa00071", pathway_name="Fatty acid degradation"),
}

UNIPROT_DB: dict[tuple[str, int], ProteinEntry] = {
    ("FGF5", 9606): ProteinEntry(
        gene_symbol="FGF5", protein_name="Fibroblast growth factor 5",
        function_summary="Regulates hair follicle growth cycle.", source_accession="P12034",
    ),
    ("UCP1", 9606): ProteinEntry(
        gene_symbol="UCP1", protein_name="Uncoupling protein 1",
        function_summary="Mitochondrial proton channel, generates heat.", source_accession="P25874",
    ),
}


def make_fake_go_client(log: CallLog):
    async def fake_fetch_go_annotation(gene_symbol: str, uniprot_accession: str):
        log.record(gene_symbol, uniprot_accession)
        return GO_DB.get(gene_symbol)
    return fake_fetch_go_annotation


def make_fake_kegg_client(log: CallLog):
    async def fake_fetch_pathway(kegg_gene_id: str):
        log.record(kegg_gene_id)
        return KEGG_DB.get(kegg_gene_id)
    return fake_fetch_pathway


def make_fake_uniprot_client(log: CallLog):
    async def fake_fetch_uniprot(gene_symbol: str, tax_id: int):
        log.record(gene_symbol, tax_id)
        return UNIPROT_DB.get((gene_symbol, tax_id))
    return fake_fetch_uniprot


def make_fake_uniprot_multi_client(log: CallLog):
    """Candidate-list fake that always returns TWO reviewed hits for a gene —
    one genuinely trait-relevant, one a plausible-looking distractor — so
    live-LLM tests can force the §8 multi-hit branch deterministically
    without depending on a real gene that happens to have two reviewed
    UniProt entries today. The trait-relevant entry is placed SECOND so a
    passing test proves the model reasoned about function_summary text
    rather than defaulting to array order.
    """
    async def fake_list_uniprot_candidates(gene_symbol: str, tax_id: int):
        log.record(gene_symbol, tax_id)
        return [
            {
                "source_accession": "Q99999",
                "protein_name": f"{gene_symbol}-related pseudogene product",
                "function_summary": (
                    "Isoform lacking the canonical targeting sequence; catalytically "
                    "inactive, function undetermined, no established role in the "
                    "trait under investigation."
                ),
            },
            {
                "source_accession": "P25874",
                "protein_name": "Uncoupling protein 1",
                "function_summary": (
                    "Mitochondrial inner-membrane proton channel expressed in brown "
                    "adipose tissue; uncouples oxidative phosphorylation from ATP "
                    "synthesis to generate heat via non-shivering thermogenesis, the "
                    "core mechanism of cold adaptation."
                ),
            },
        ]
    return fake_list_uniprot_candidates


def make_fake_uniprot_list_client(log: CallLog):
    """Candidate-list fake for the primary list_uniprot_candidates path
    (§8) — always a single reviewed hit per gene, so the agent takes the
    'no decision needed' branch and never calls the LLM."""
    async def fake_list_uniprot_candidates(gene_symbol: str, tax_id: int):
        log.record(gene_symbol, tax_id)
        entry = UNIPROT_DB.get((gene_symbol, tax_id))
        if entry is None:
            return []
        return [{
            "source_accession": entry.source_accession,
            "protein_name": entry.protein_name,
            "function_summary": entry.function_summary,
        }]
    return fake_list_uniprot_candidates