"""
Species Resolver — mock subagent (Task 1)
Resolves a species name to its genome assembly ID.
Currently returns hardcoded fake data — no real NCBI calls yet.
"""

# Fake lookup table simulating what NCBI would return.
# taxonomic_group lets the Visualization subagent build a *relevant*
# size_comparison (e.g. "compare the tiger to other cats" should pull in
# other Felidae, not an elephant) instead of comparing against every
# species in the mock DB regardless of relatedness.
_FAKE_SPECIES_DB = {
    "house mouse": {
        "assembly_id": "GCF_000001635.27",
        "scientific_name": "Mus musculus",
        "common_name": "House mouse",
        "confidence": 1.0,
        "taxonomic_group": "Rodentia",
    },
    "tiger": {
        "assembly_id": "GCF_000464555.1",
        "scientific_name": "Panthera tigris",
        "common_name": "Tiger",
        "confidence": 0.95,
        "taxonomic_group": "Felidae",
    },
    "asian elephant": {
        "assembly_id": "GCA_024166365.1",
        "scientific_name": "Elephas maximus",
        "common_name": "Asian elephant",
        "confidence": 0.9,
        "taxonomic_group": "Proboscidea",
    },
    "lion": {
        "assembly_id": "GCF_008795835.1",
        "scientific_name": "Panthera leo",
        "common_name": "Lion",
        "confidence": 0.95,
        "taxonomic_group": "Felidae",
    },
    "leopard": {
        "assembly_id": "GCF_001857705.1",
        "scientific_name": "Panthera pardus",
        "common_name": "Leopard",
        "confidence": 0.93,
        "taxonomic_group": "Felidae",
    },
    "domestic cat": {
        "assembly_id": "GCF_000181335.3",
        "scientific_name": "Felis catus",
        "common_name": "Domestic cat",
        "confidence": 0.97,
        "taxonomic_group": "Felidae",
    },
}


async def resolve_species(species_name: str) -> dict:
    """
    Mock version of Species Resolver.
    Input: species_name (str)
    Output: dict matching SpeciesResolverOutput shape
    """
    key = species_name.strip().lower()

    if key in _FAKE_SPECIES_DB:
        return _FAKE_SPECIES_DB[key]

    # No match found — mimic the real "species not found" case
    return {
        "assembly_id": None,
        "scientific_name": None,
        "common_name": None,
        "confidence": 0.0,
        "taxonomic_group": None,
    }


def get_all_species() -> list[dict]:
    """
    Return every species record currently known to this mock, keyed by
    nothing in particular (list, not dict) since assembly_id is the real
    identity field. Used by the Visualization subagent to label
    cross-species comparison charts with real common/scientific names
    instead of bare assembly IDs.
    """
    return [dict(info) for info in _FAKE_SPECIES_DB.values()]


if __name__ == "__main__":
    import asyncio

    async def _quick_test():
        result = await resolve_species("tiger")
        print("Tiger:", result)
        assert result["assembly_id"] == "GCF_000464555.1"

        result = await resolve_species("  House Mouse  ")
        print("House mouse:", result)
        assert result["assembly_id"] == "GCF_000001635.27"

        result = await resolve_species("dragon")
        print("Dragon:", result)
        assert result["assembly_id"] is None

        print("All tests passed ✅")

    asyncio.run(_quick_test())