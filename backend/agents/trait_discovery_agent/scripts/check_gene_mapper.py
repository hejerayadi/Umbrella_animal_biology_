import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schemas.inputs import GeneMapperInput
from subagents.gene_mapper import mock_gene_mapper, gene_mapper_agent

logging.basicConfig(level=logging.INFO)


async def main():
    # --- Mock test (no LLM, no QuickGO) -----------------------------------
    print("=" * 60)
    print("MOCK GENE MAPPER TEST")
    print("=" * 60)
    mock_input = GeneMapperInput(
        trait_name="fur growth",
        gene_list=["FGF5", "KRT71", "HR", "UNKNOWN"],
        species_name="Mus musculus",
        instruction="Map genes to GO biological process",
        context={},
    )
    mock_out = await mock_gene_mapper(mock_input)
    print(f"Status: {mock_out.status}")
    for a in mock_out.go_annotations:
        print(f"  {a.gene_symbol} | {a.go_id} | {a.go_name}")
    print(f"Unmatched: {mock_out.unmatched_genes}")

    # --- Live test (requires QuickGO + optional NVIDIA_NIM_API_KEY) -------
    print("\n" + "=" * 60)
    print("LIVE GENE MAPPER TEST")
    print("=" * 60)
    live_input = GeneMapperInput(
        trait_name="hair follicle development",
        gene_list=["FGF5", "HR"],
        species_name="Homo sapiens",
        instruction="Map genes to GO biological process",
        context={
            "uniprot_accessions": {
                "FGF5": "P12034",
                "HR": "O43593",
            }
        },
    )
    try:
        live_out = await gene_mapper_agent(live_input)
        print(f"Status: {live_out.status}")
        for a in live_out.go_annotations:
            print(f"  {a.gene_symbol} | {a.go_id} | {a.go_name}")
        print(f"Unmatched: {live_out.unmatched_genes}")
    except Exception as exc:
        print(f"Live test failed (expected if no network or NIM key missing): {exc}")


if __name__ == "__main__":
    asyncio.run(main())