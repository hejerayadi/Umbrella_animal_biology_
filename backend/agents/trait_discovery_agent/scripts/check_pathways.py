import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schemas.inputs import PathwaysInput
from subagents.pathways import pathways_agent

async def main():
    out = await pathways_agent(PathwaysInput(
        gene_list=["FGF5"], instruction="test",
        context={"kegg_gene_ids": {"FGF5": "hsa:2249"}},
    ))
    print(out)

asyncio.run(main())