import argparse
import asyncio
import logging

from workflows.reporting import WorkflowReportFormatter
from workflows.trait_discovery_graph import build_trait_discovery_graph
from workflows.state import TraitDiscoveryState

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


async def main(verbose: bool = False) -> None:
    if not verbose:
        logging.getLogger().setLevel(logging.WARNING)

    app = build_trait_discovery_graph()
    state = TraitDiscoveryState(
        trait_name="fur growth",
        species_name="mouse",
        instruction="Which genes cause fur growth?",
        context={"gene_list": ["FGF5", "KRT71", "HR"]},
    )
    result = await app.ainvoke(state)

    formatter = WorkflowReportFormatter()
    print(formatter.format(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Trait Discovery workflow")
    parser.add_argument("--verbose", action="store_true", help="Show developer INFO logs and node traces")
    args = parser.parse_args()
    asyncio.run(main(verbose=args.verbose))