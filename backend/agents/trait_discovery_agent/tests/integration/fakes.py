
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