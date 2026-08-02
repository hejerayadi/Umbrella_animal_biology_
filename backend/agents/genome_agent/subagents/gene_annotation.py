"""
Gene Annotation — mock subagent (Task 1)
Retrieves gene/feature annotation data for a resolved assembly.
Currently returns hardcoded fake data — no real NCBI calls yet.
"""

# Fake lookup table simulating what NCBI Gene would return
_FAKE_ANNOTATION_DB = {
    "GCF_000001635.27": {
        "gene_table": [
            {"gene_name": "Trp53", "location": "chr11:69580309-69591923", "function": "Tumor suppressor"},
            {"gene_name": "Fgf5", "location": "chr5:98211000-98221000", "function": "Hair growth regulation"},
        ],
        "gene_list": ["Trp53", "Fgf5"],
    },
    "GCF_000464555.1": {
        "gene_table": [
            {"gene_name": "Mc1r", "location": "chr15:12000000-12003000", "function": "Coat color regulation"},
        ],
        "gene_list": ["Mc1r"],
    },
    "GCA_024166365.1": {
        "gene_table": [],
        "gene_list": [],
    },
}


async def get_gene_annotation(assembly_id: str) -> dict:
    """
    Mock version of Gene Annotation.
    Input: assembly_id (str)
    Output: dict matching GeneAnnotationOutput shape
    """
    if assembly_id in _FAKE_ANNOTATION_DB:
        return _FAKE_ANNOTATION_DB[assembly_id]

    # No match found
    return {
        "gene_table": [],
        "gene_list": [],
    }


if __name__ == "__main__":
    import asyncio

    async def _quick_test():
        result = await get_gene_annotation("GCF_000001635.27")
        print("Mouse:", result)
        assert result["gene_list"] == ["Trp53", "Fgf5"]

        result = await get_gene_annotation("GCF_000464555.1")
        print("Tiger:", result)
        assert len(result["gene_table"]) == 1

        result = await get_gene_annotation("unknown_id")
        print("Unknown:", result)
        assert result["gene_list"] == []

        print("All tests passed ✅")

    asyncio.run(_quick_test())
