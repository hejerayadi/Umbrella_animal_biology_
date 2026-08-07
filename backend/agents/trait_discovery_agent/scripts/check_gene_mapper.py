import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schemas.inputs import GeneMapperInput
from subagents.gene_mapper import gene_mapper_agent

async def main():
    out = await gene_mapper_agent(GeneMapperInput(
        trait_name="fur growth", gene_list=["FGF5"], species_name="Mus musculus",
        instruction="test", context={"uniprot_accessions": {"FGF5": "P48145"}}
    ))
    print(out)

asyncio.run(main())