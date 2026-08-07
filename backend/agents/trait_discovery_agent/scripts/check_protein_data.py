import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schemas.inputs import ProteinDataInput
from subagents.protein_data import protein_data_agent

async def main():
    out = await protein_data_agent(ProteinDataInput(
        gene_list=["FGF5"], instruction="test", context={"tax_id": 9606}
    ))
    print(out)

asyncio.run(main())