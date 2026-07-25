from __future__ import annotations

from dataclasses import dataclass

from .state import WorkflowState
from .llm import ask_llm

@dataclass(frozen=True)
class ExecutionPlan:
    """The initial routing decision produced from a user's query."""

    initial_agent: str
    # The original user request is preserved so downstream agents keep the
    # full context instead of only seeing the keyword that triggered routing.
    objective: str


class Planner:
    """Decide where the workflow should start and how it should continue.

    The planner performs two related jobs:
    1. `plan()` chooses the first agent from the user's query using a simple
       keyword-based heuristic.
    2. `next_agent()` determines the default hand-off chain when an agent
       finishes and the workflow should continue.

    This is intentionally lightweight and rule-driven rather than model-driven
    so the orchestration behavior is easy to reason about and debug.
    """

    def plan(self, user_query: str) -> ExecutionPlan:
        # Normalize once so keyword matching is case-insensitive.
        query = user_query.lower()

        # Protein-focused requests are routed first because they usually ask
        # about structures, 3D forms, or sequence-to-structure interpretation.
        if any(keyword in query for keyword in [
            "protein", "proteins", "structure", "3d"
        ]):
            return ExecutionPlan(
                initial_agent="Protein",
                objective=user_query
            )

        # Genomic requests are detected by common sequence and gene terms.
        if any(keyword in query for keyword in [
            "gene", "genes", "dna", "genome",
            "chromosome", "sequence"
        ]):
            return ExecutionPlan(
                initial_agent="Genome",
                objective=user_query
            )

        # Trait discovery is used when the query is asking what causes or is
        # associated with a biological trait.
        if any(keyword in query for keyword in [
            "trait", "tusk", "tusks", "horn",
            "adaptation", "responsible for",
            "associated with", "linked to"
        ]):
            return ExecutionPlan(
                initial_agent="Trait",
                objective=user_query
            )

        # Evolutionary comparisons move the workflow toward species relation
        # analysis and comparative biology.
        if any(keyword in query for keyword in [
            "evolution", "evolve",
            "compare species",
            "related species"
        ]):
            return ExecutionPlan(
                initial_agent="Evolution",
                objective=user_query
            )

        # Reconstruction is chosen for incomplete biological data that needs
        # to be inferred or rebuilt.
        if any(keyword in query for keyword in [
            "reconstruct", "reconstruction",
            "missing sequence",
            "incomplete genome"
        ]):
            return ExecutionPlan(
                initial_agent="Reconstruction",
                objective=user_query
            )

        # Biodiversity questions are usually about habitat, distribution, and
        # conservation context.
        if any(keyword in query for keyword in [
            "habitat", "distribution",
            "biodiversity",
            "conservation",
            "hotspot"
        ]):
            return ExecutionPlan(
                initial_agent="Biodiversity",
                objective=user_query
            )

        # Image-based requests need multimodal recognition rather than text-only
        # analysis.
        if any(keyword in query for keyword in [
            "image", "photo", "picture"
        ]):
            return ExecutionPlan(
                initial_agent="Multimodal",
                objective=user_query
            )

        # Literature lookup acts as the default route for paper and reference
        # oriented questions, and also serves as the fallback if no other rule
        # matches more specifically.
        if any(keyword in query for keyword in [
            "literature", "paper", "papers",
            "study", "pubmed", "reference"
        ]):
            return ExecutionPlan(
                initial_agent="Literature",
                objective=user_query
            )

        # Fallback route: when the query is ambiguous, start with literature
        # review to gather context before deciding on a narrower analysis path.
        return ExecutionPlan(
            initial_agent="Literature",
            objective=user_query
        )

    def next_agent(self, state: WorkflowState) -> str | None:
        """Return the next agent in the hand-off chain, if any.

        The state object carries the current agent name, and this method uses a
        small fixed progression to keep the mock workflow deterministic.
        Returning `None` means the workflow should stop after the current step.
        """

        current = state.current_agent

        # The mock workflow models a few simple multi-agent paths without
        # attempting to infer dynamic dependencies between every possible agent.

        if current == "Genome":
            return "Evolution"

        if current == "Evolution":
            return "Biodiversity"

        # Trait questions may naturally hand off to protein-level analysis.
        if current == "Trait":
            return "Protein"

        # The remaining agents are terminal in this mock flow.
        if current == "Protein":
            return None

        if current == "Reconstruction":
            return None

        if current == "Biodiversity":
            return None

        if current == "Multimodal":
            return None

        if current == "Literature":
            return None

        # Any unknown agent name is treated as a terminal state.
        return None