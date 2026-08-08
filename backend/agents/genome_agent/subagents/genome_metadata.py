"""
Genome Metadata — mock subagent (Task 1)
Retrieves genome size, chromosome count, and karyotype for a resolved assembly.
Currently returns hardcoded fake data — no real NCBI calls yet.
"""

# Fake lookup table simulating what NCBI Assembly would return
_FAKE_METADATA_DB = {
    "GCF_000001635.27": {
        "genome_size_bp": 2728222451,
        "chromosome_count": 40,
        "karyotype": "20 pairs",
        "assembly_level": "Chromosome",
    },
    "GCF_000464555.1": {
        "genome_size_bp": 2489000000,
        "chromosome_count": 38,
        "karyotype": "19 pairs",
        "assembly_level": "Chromosome",
    },
    "GCA_024166365.1": {
        "genome_size_bp": 3200000000,
        "chromosome_count": 56,
        "karyotype": "28 pairs",
        "assembly_level": "Scaffold",
    },
    "GCF_008795835.1": {
        "genome_size_bp": 2461000000,
        "chromosome_count": 38,
        "karyotype": "19 pairs",
        "assembly_level": "Chromosome",
    },
    "GCF_001857705.1": {
        "genome_size_bp": 2525000000,
        "chromosome_count": 38,
        "karyotype": "19 pairs",
        "assembly_level": "Chromosome",
    },
    "GCF_000181335.3": {
        "genome_size_bp": 2418000000,
        "chromosome_count": 38,
        "karyotype": "19 pairs",
        "assembly_level": "Chromosome",
    },
}


async def get_genome_metadata(assembly_id: str) -> dict:
    """
    Mock version of Genome Metadata.
    Input: assembly_id (str)
    Output: dict matching GenomeMetadataOutput shape
    """
    if assembly_id in _FAKE_METADATA_DB:
        return _FAKE_METADATA_DB[assembly_id]

    # No match found
    return {
        "genome_size_bp": None,
        "chromosome_count": None,
        "karyotype": None,
        "assembly_level": None,
    }


def get_all_genome_metadata() -> dict[str, dict]:
    """
    Return every assembly_id -> metadata pair currently known to this mock.
    Used by the Visualization subagent to build real cross-species
    comparisons instead of a single-species placeholder chart.
    """
    return dict(_FAKE_METADATA_DB)


if __name__ == "__main__":
    import asyncio

    async def _quick_test():
        result = await get_genome_metadata("GCF_000001635.27")
        print("Mouse:", result)
        assert result["genome_size_bp"] == 2728222451

        result = await get_genome_metadata("GCF_000464555.1")
        print("Tiger:", result)
        assert result["chromosome_count"] == 38

        result = await get_genome_metadata("unknown_id")
        print("Unknown:", result)
        assert result["genome_size_bp"] is None

        print("All tests passed ✅")

    asyncio.run(_quick_test())