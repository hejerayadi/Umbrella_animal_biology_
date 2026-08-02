"""
Species Resolver — mock subagent (Task 1)
Resolves a species name to its genome assembly ID.
Currently returns hardcoded fake data — no real NCBI calls yet.
"""

# Fake lookup table simulating what NCBI would return
_FAKE_SPECIES_DB = {
    "house mouse": {
        "assembly_id": "GCF_000001635.27",
        "scientific_name": "Mus musculus",
        "common_name": "House mouse",
        "confidence": 1.0,
    },
    "tiger": {
        "assembly_id": "GCF_000464555.1",
        "scientific_name": "Panthera tigris",
        "common_name": "Tiger",
        "confidence": 0.95,
    },
    "asian elephant": {
        "assembly_id": "GCA_024166365.1",
        "scientific_name": "Elephas maximus",
        "common_name": "Asian elephant",
        "confidence": 0.9,
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
    }


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
