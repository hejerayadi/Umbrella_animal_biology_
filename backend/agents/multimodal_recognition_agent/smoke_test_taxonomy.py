"""
Live smoke test for Phase 4's real GBIF and NCBI providers.

Makes REAL network calls to the public GBIF and NCBI APIs. Not a pytest test on
purpose - this is an opt-in manual check, run directly, the same way
smoke_test_azure.py checks live Azure connectivity.

Run from the agent's own venv, with .env populated (NCBI_TOOL, NCBI_EMAIL):

    python smoke_test_taxonomy.py
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

AGENT_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(AGENT_ENV_PATH)

import os  # noqa: E402 - after load_dotenv on purpose
import sys  # noqa: E402

# adapters/taxonomy.py uses a relative import (..domain.errors), so it needs
# to be imported as part of the multimodal_recognition_agent package, not as
# a bare top-level module. Put backend/ on sys.path and import the full path.
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from agents.multimodal_recognition_agent.adapters.taxonomy import (  # noqa: E402
    RealGBIFProvider,
    RealNCBIProvider,
)


def main() -> None:
    tool = os.getenv("NCBI_TOOL")
    email = os.getenv("NCBI_EMAIL")
    if not tool or not email:
        print("NCBI_TOOL and NCBI_EMAIL must both be set in .env - aborting.")
        return

    gbif = RealGBIFProvider()
    ncbi = RealNCBIProvider(tool=tool, email=email)

    test_species = [
        ("panthera_leo", "Panthera leo"),
        ("panthera_tigris", "Panthera tigris"),
        ("totally_fake_species", "Xyzabc nonexistus"),
    ]

    for species_id, scientific_name in test_species:
        print(f"\n--- {scientific_name} ---")

        gbif_result = gbif.lookup(species_id, scientific_name)
        print(f"GBIF: available={gbif_result.available} matched={gbif_result.matched} "
              f"identifier={gbif_result.identifier} accepted_name={gbif_result.accepted_name} "
              f"rank={gbif_result.rank}")

        ncbi_result = ncbi.lookup(species_id, scientific_name)
        print(f"NCBI: available={ncbi_result.available} matched={ncbi_result.matched} "
              f"identifier={ncbi_result.identifier}")


if __name__ == "__main__":
    main()
