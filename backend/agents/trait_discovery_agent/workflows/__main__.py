import asyncio

from workflows.trait_discovery_graph import build_trait_discovery_graph
from workflows.state import TraitDiscoveryState


async def main():
    app = build_trait_discovery_graph()
    result = await app.ainvoke(TraitDiscoveryState(
        trait_name="fur growth",
        species_name="mouse",
        instruction="Which genes cause fur growth?",
        context={"gene_list": ["FGF5", "KRT71", "HR"]},
    ))
    print(result)


if __name__ == "__main__":
    asyncio.run(main())