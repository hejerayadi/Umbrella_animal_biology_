"""Runnable demo of the LangGraph-based scientific multi-agent orchestrator.

Run from the repository root with:

    python -m backend.main

This exercises the full Planner -> Worker -> Capability Resolver -> Worker
loop against the mocked worker agents, using real Azure OpenAI calls for the
Planner and Capability Resolver.
"""
from __future__ import annotations

from .orchestrator.lang_graph import GlobalOrchestrator
from .orchestrator.state import WorkflowState


def _print_run(user_query: str, state: WorkflowState) -> None:
    print(f"\nQuery: {user_query}")
    print("-" * 72)
    for entry in state.execution_history:
        print(f"  {entry}")
    print("-" * 72)
    print(f"Final context: {state.context}")


def main() -> None:
    orchestrator = GlobalOrchestrator()

    demo_runs: list[tuple[str, dict[str, str]]] = [
        ("Show me the genome of the woolly mammoth.", {"species": "woolly mammoth"}),
        (
            "Predict the 3D protein structure linked to the tusk trait in the woolly mammoth.",
            {"species": "woolly mammoth"},
        ),
        (
            "Reconstruct the incomplete genome of the woolly mammoth using related species.",
            {"species": "woolly mammoth"},
        ),
    ]

    for user_query, initial_context in demo_runs:
        final_state = orchestrator.run(user_query, initial_context=initial_context)
        _print_run(user_query, final_state)


if __name__ == "__main__":
    main()
