import argparse
import asyncio
import logging
import os

from workflows.reporting import WorkflowReportFormatter
from workflows.scenario_runner import SCENARIOS, run_scenario
from workflows.trait_discovery_graph import build_trait_discovery_graph
from workflows.state import TraitDiscoveryState

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


async def main(verbose: bool = False) -> None:
    """Original single happy-path example. Unchanged — kept as the default
    when no --scenario is given. For a live, step-by-step trace of this same
    input (and three other control-flow paths), use --scenario instead."""
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


def _warn_if_no_api_key() -> None:
    if not (os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NVIDIA_NIM_API_KEY")):
        print(
            "NOTE: NVIDIA_NIM_API_KEY is not set. Every path below calls the real NIM "
            "model and will raise once it reaches an LLM node. Copy .env.example to "
            ".env and add your key.\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Trait Discovery workflow")
    parser.add_argument("--verbose", action="store_true", help="Show developer INFO logs and node traces")
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        default=None,
        help=(
            "Run a predefined scenario against the real graph and the real NVIDIA NIM "
            "model, with a live step-by-step execution trace and summary. Omit this "
            "flag to run the original single happy-path example instead."
        ),
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.WARNING)  # scenario trace speaks for itself
    _warn_if_no_api_key()

    if args.scenario:
        asyncio.run(run_scenario(args.scenario))
    else:
        asyncio.run(main(verbose=args.verbose))
