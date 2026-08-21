import pytest
from schemas.inputs import ProteinDataInput
from subagents.protein_data import protein_data_agent
from schemas.common import AgentStatus

@pytest.mark.asyncio
async def test_protein_data_real_uniprot_and_cache():
    input_ = ProteinDataInput(
        gene_list=["FGF5"], trait_name="hair growth", instruction="test", context={"tax_id": 9606},
    )
    first = await protein_data_agent(input_)
    assert first.status == AgentStatus.COMPLETED
    assert first.proteins[0].gene_symbol == "FGF5"

    second = await protein_data_agent(input_)  # should hit Qdrant cache, not UniProt
    assert second.proteins[0].function_summary == first.proteins[0].function_summary