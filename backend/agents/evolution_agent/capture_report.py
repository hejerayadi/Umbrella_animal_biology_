"""Capture live full-stack responses (Azure LLM + pipeline) to a JSON file.

Run from the repo root:

    $env:PYTHONIOENCODING="utf-8"
    python -m backend.agents.evolution_agent.capture_report [output.json]

Each instruction is classified by the real Azure LLM, then the sequential
MC -> Phylo pipeline runs.  The intent, raw result, and an HTTP-style
response body are captured for review.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

from backend.agents.evolution_agent.orchestrator_adapter import (
    OrchestratorEvolutionAgent,
)
from backend.agents.evolution_agent.schema import AgentRequest

OUT_DEFAULT = "evolution_capture.json"

INSTRUCTIONS: list[tuple[str, str]] = [
    (
        "molcomp-2species",
        "How similar are human and chimpanzee proteins?",
    ),
    (
        "molcomp-3species",
        "Compare the protein sequences of human, mouse and zebrafish.",
    ),
    (
        "phylo-4species",
        "Show me the evolutionary tree for human, mouse, chicken and zebrafish.",
    ),
    (
        "full-3species",
        "Compare homo sapiens, pan troglodytes and mus musculus evolutionarily.",
    ),
    (
        "full-5species",
        "Compare the evolution of human, chimp, mouse, chicken and zebrafish.",
    ),
    (
        "rejected-nonbio",
        "What is the capital of France?",
    ),
    (
        "common-names",
        "How are humans, chimpanzees and mice related evolutionarily?",
    ),
]


def _status_value(status) -> str:
    return getattr(status, "value", str(status))


async def run_one(agent: OrchestratorEvolutionAgent, label: str, instruction: str) -> dict:
    request = AgentRequest(instruction=instruction, context={})

    try:
        result = await agent.run(request)
    except Exception as exc:  # noqa: BLE001
        return {
            "label": label,
            "instruction": instruction,
            "error": f"{type(exc).__name__}: {exc}",
        }

    output = result.output

    return {
        "label": label,
        "instruction": instruction,
        "status": _status_value(result.status),
        "confidence": result.confidence,
        "newick_tree": result.newick_tree,
        "tree_url": result.tree_url,
        "similarity_scores": result.similarity_scores,
        "alignment_url": result.alignment_url,
        "source_agents": list(result.source_agents),
        "output": output,
    }


async def main() -> None:
    out_path = Path(sys.argv[1] if len(sys.argv) > 1 else OUT_DEFAULT)
    agent = OrchestratorEvolutionAgent()

    print(f"Capturing {len(INSTRUCTIONS)} live requests via Azure LLM...")
    captures: list[dict] = []
    for label, instruction in INSTRUCTIONS:
        print(f"  - {label}: {instruction!r}")
        captures.append(await run_one(agent, label, instruction))

    out_path.write_text(
        json.dumps({"captured_at": None, "cases": captures}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nWrote {out_path} ({len(captures)} cases)")


if __name__ == "__main__":
    asyncio.run(main())
