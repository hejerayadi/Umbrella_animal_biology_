"""Shared protein-sequence retrieval for both Evolution sub-agents.

Molecular Comparison and Phylogenetic Reconstruction must analyse the SAME
sequences for a ``full_analysis`` answer to be coherent — a similarity
network built from full-length cytochrome b and a tree built from some
other fragment are two unrelated findings presented as one result.

It lives in ``tools`` rather than in either worker so that neither
sub-agent has to import the other: they stay independently testable and
independently replaceable, which is the property
``test_sub_agents_never_import_or_call_each_other`` protects.

No alignment happens here. MAFFT is exclusive to the phylogenetic branch —
gap characters would corrupt ESM-2 embeddings — so both workers receive
the same raw, unaligned sequences.
"""

from __future__ import annotations

import requests
from langsmith import traceable

# Mitochondrial genes with broad, reviewed coverage across vertebrates:
# good defaults when the caller names no target gene.
DEFAULT_GENE_CANDIDATES = ["CYTB", "COX1"]


@traceable(name="UniProt sequence fetch", run_type="tool")
def fetch_uniprot_sequence(species: str, gene_candidates: list[str]) -> str:
    """Fetch a reviewed protein sequence for `species` by organism name.

    Tries each gene symbol in order until one resolves. Raises ValueError
    if none do -- callers turn that into a FAILED AgentResult, they never
    let it propagate as an unhandled exception.
    """
    for gene in gene_candidates:
        params = {
            "query": f'organism_name:"{species}" AND gene:{gene} AND reviewed:true',
            "format": "fasta",
            "size": 1,
        }
        resp = requests.get(
            "https://rest.uniprot.org/uniprotkb/search", params=params, timeout=30
        )
        resp.raise_for_status()
        fasta = resp.text.strip()
        if fasta:
            lines = fasta.split("\n")
            return "".join(l for l in lines if not l.startswith(">"))
    raise ValueError(
        f"No reviewed UniProt sequence for species='{species}', "
        f"tried genes={gene_candidates}"
    )


def parse_fasta_inputs(
    protein_inputs: list[str], species_list: list[str]
) -> dict[str, str]:
    """Match pre-supplied FASTA strings to species by header substring."""
    resolved: dict[str, str] = {}
    for fasta in protein_inputs:
        lines = [l for l in fasta.strip().split("\n") if l]
        if not lines or not lines[0].startswith(">"):
            continue
        header = lines[0][1:].lower()
        seq = "".join(l for l in lines[1:] if not l.startswith(">"))
        for sp in species_list:
            if sp.lower().replace(" ", "_") in header or sp.lower() in header:
                resolved[sp] = seq
                break
    return resolved
