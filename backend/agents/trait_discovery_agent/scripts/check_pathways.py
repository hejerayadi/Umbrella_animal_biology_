import asyncio
import logging

from schemas.inputs import PathwaysInput
from subagents.pathways import mock_pathways_agent, pathways_agent

logging.basicConfig(level=logging.INFO)

async def main():
    # --- Mock test (no LLM, no KEGG) ---------------------------------------
    print("=" * 60)
    print("MOCK PATHWAYS TEST")
    print("=" * 60)
    mock_input = PathwaysInput(
        gene_list=["UCP1", "PRDM16", "UNKNOWN"],
        trait_name="cold adaptation",
        instruction="Find KEGG pathways",
        context={"kegg_gene_ids": {"UCP1": "hsa:7350", "PRDM16": "hsa:63976", "UNKNOWN": "hsa:99999"}},
    )
    mock_out = await mock_pathways_agent(mock_input)
    print(f"Status: {mock_out.status}")
    for p in mock_out.pathways:
        print(f"  {p.pathway_id} | {p.pathway_name} | {p.reasoning}")
    print(f"Malformed: {mock_out.malformed_ids}")

    # --- Live test (requires NVIDIA_NIM_API_KEY + network) ------------------
    print("\n" + "=" * 60)
    print("LIVE PATHWAYS TEST")
    print("=" * 60)
    live_input = PathwaysInput(
        gene_list=["UCP1"],
        trait_name="thermogenesis",
        instruction="Find KEGG pathways",
        context={"kegg_gene_ids": {"UCP1": "hsa:7350"}},
    )
    try:
        live_out = await pathways_agent(live_input)
        print(f"Status: {live_out.status}")
        for p in live_out.pathways:
            print(f"  {p.pathway_id} | {p.pathway_name} | {p.reasoning}")
        print(f"Malformed: {live_out.malformed_ids}")
    except Exception as exc:
        print(f"Live test failed (expected if NIM key missing): {exc}")

if __name__ == "__main__":
    asyncio.run(main())