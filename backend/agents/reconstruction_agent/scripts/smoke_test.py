"""End-to-end check that the agent runs, without touching any external service.

Answers "is this deployment wired correctly?" - settings load, the registry
builds, the graph compiles, and a request produces a well-formed result.

    uv run python scripts/smoke_test.py

Exits non-zero on failure, so it works as a deployment gate.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Runnable straight from a checkout, before `uv sync` has installed the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reconstruction_agent.application.reconstruction_service import (  # noqa: E402
    ReconstructionService,
)
from reconstruction_agent.configuration.logging import configure_logging  # noqa: E402
from reconstruction_agent.configuration.settings import Settings  # noqa: E402
from reconstruction_agent.contracts.input import ReconstructionRequest  # noqa: E402
from reconstruction_agent.observability.events import CollectingEmitter  # noqa: E402
from reconstruction_agent.tools.registry import ToolRegistry  # noqa: E402

# A sequence with one 12-base gap and generous flanks on both sides.
SAMPLE = "ACGTTGCA" * 30 + "N" * 12 + "TTACGGCA" * 30


async def main() -> int:
    configure_logging()

    # An empty registry: no tool can be selected, so nothing leaves the process.
    # The run should still complete and report the gap as unresolved.
    settings = Settings(max_iterations=2)
    events = CollectingEmitter()
    service = ReconstructionService(settings, ToolRegistry(), events=events)

    request = ReconstructionRequest.from_agent_request(
        "Reconstruct the unresolved region.",
        {"sequence": {"identifier": "smoke_seq", "residues": SAMPLE},
         "organism": "Testus organismus"},
    )

    result = await service.reconstruct(request)

    print(f"\nSequence      : {result.sequence_id} ({result.original_length} bases)")
    print(f"Gaps examined : {len(result.gaps)}")
    print(f"Reconstructed : {result.reconstructed_count}")
    print(f"Iterations    : {result.iterations}")
    print(f"Events        : {len(events.events)}")
    print(f"\nSummary: {result.summary}\n")

    problems: list[str] = []
    if len(result.gaps) != 1:
        problems.append(f"Expected 1 gap, found {len(result.gaps)}.")
    if not result.summary:
        problems.append("No summary was produced.")
    if not events.events:
        problems.append("No events were emitted.")

    if problems:
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        return 1

    print("Smoke test passed: the agent runs end to end with no external calls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
