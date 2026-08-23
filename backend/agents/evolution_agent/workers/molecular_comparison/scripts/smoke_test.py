"""Manual smoke test for MolecularComparisonAgent's real pipeline.

Runs the real (non-injected) fetch_fn/embed_fn -- hits UniProt over the
network and loads ESM-2 -- so it's a script, not a pytest test. Re-run
any time to sanity-check the live pipeline end-to-end.

Usage (evolution_agent venv activated, run from anywhere):
    python backend/agents/evolution_agent/workers/molecular_comparison/scripts/smoke_test.py
"""

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[6]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.agents.evolution_agent.schema import AgentRequest, AgentStatus
from backend.agents.evolution_agent.workers.molecular_comparison.logic import (
    MolecularComparisonAgent,
)

SPECIES = ["homo sapiens", "pan troglodytes", "mus musculus"]


def main() -> None:
    agent = MolecularComparisonAgent()  # real fetch_fn + embed_fn, no injection
    req = AgentRequest(
        instruction="Compare cytochrome b across primates and rodents",
        species_list=SPECIES,
    )

    t0 = time.time()
    result = agent.run(req)
    elapsed = time.time() - t0

    print(f"\n--- status: {result.status} (elapsed {elapsed:.1f}s) ---")
    if result.status == AgentStatus.FAILED:
        print("output:", result.output)
        return

    mc = result.output
    print("species_list:", mc.species_list)
    print("\nsimilarity_scores:")
    for e in mc.similarity_scores:
        print(f"  {e.species_a} <-> {e.species_b}: {e.score}")
    print("\nspecies_groups:")
    for g in mc.species_groups:
        print(f"  group {g.group_id}: {g.species} (mean_score={g.mean_score})")
    print("\nsimilarity_network:")
    print("  nodes:", [n["id"] for n in mc.similarity_network["nodes"]])
    print("  edges:", mc.similarity_network["edges"])
    print("\nconfidence:", result.confidence)


if __name__ == "__main__":
    main()
