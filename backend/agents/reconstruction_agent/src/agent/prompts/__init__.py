"""Prompt construction. The prompt text itself lives in `src/prompts/*.md`.

This package holds only the code that loads those files and fills in their
placeholders - no prompt wording. See `loader.py` for why.
"""
from __future__ import annotations

import json
from typing import Any

from agent.prompts.loader import (
    PromptNotFoundError,
    PromptRenderError,
    available,
    clear_cache,
    load,
    render,
)

__all__ = [
    "PromptNotFoundError",
    "PromptRenderError",
    "available",
    "clear_cache",
    "critic_system",
    "critic_user",
    "deterministic_explanation",
    "explanation_system",
    "explanation_user",
    "load",
    "planner_system",
    "planner_user",
    "render",
]


def _json(value: Any) -> str:
    """Pretty JSON for embedding in a prompt, with non-serialisable values coerced."""
    return json.dumps(value, indent=2, default=str)


# --- Planner ---------------------------------------------------------------


def planner_system() -> str:
    return load("planner.system")


def planner_user(
    *,
    instruction: str,
    organism: str | None,
    gaps: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    prior_critiques: list[str],
) -> str:
    """The planning request for one iteration.

    Prior critiques are included so the planner can correct course: without
    them each iteration would replan from identical information and produce
    the identical failing plan.
    """
    critiques_section = ""
    if prior_critiques:
        problems = "\n".join(f"- {critique}" for critique in prior_critiques)
        critiques_section = (
            "\nProblems found with the previous attempt - your new plan must "
            f"address these:\n{problems}\n"
        )

    return render(
        "planner.user",
        instruction=instruction,
        organism=organism or "unspecified",
        gaps=_json(gaps),
        tools=_json(tools),
        critiques_section=critiques_section,
    )


# --- Critic ----------------------------------------------------------------


def critic_system() -> str:
    return load("critic.system")


def critic_user(
    *, gap: dict[str, Any], candidate: dict[str, Any], evidence: list[dict[str, Any]]
) -> str:
    return render(
        "critic.user", gap=_json(gap), candidate=_json(candidate), evidence=_json(evidence)
    )


# --- Explanation -----------------------------------------------------------


def explanation_system() -> str:
    return load("explanation.system")


def explanation_user(
    *, gap: dict[str, Any], candidate: dict[str, Any], evidence: list[dict[str, Any]]
) -> str:
    return render(
        "explanation.user", gap=_json(gap), candidate=_json(candidate), evidence=_json(evidence)
    )


def deterministic_explanation(
    *,
    gap_id: str,
    length: int,
    confidence: float,
    references: list[str],
    mean_identity: float | None,
) -> str:
    """The explanation used when no LLM is configured.

    Stays in code rather than in a prompt file: it is not a prompt. It says
    the same things `explanation.system.md` asks a model to say, assembled
    from numbers already computed, so a run without an LLM still produces a
    reviewable account rather than an empty field.
    """
    if not references:
        return (
            f"No reference sequence aligned across {gap_id}, so no reconstruction could be "
            "supported by evidence."
        )

    shown = ", ".join(references[:3])
    more = f" and {len(references) - 3} other reference(s)" if len(references) > 3 else ""
    identity = (
        f" Mean identity across the flanking context was {mean_identity:.0%}."
        if mean_identity is not None
        else ""
    )

    return (
        f"The {length}-base reconstruction of {gap_id} is a consensus over {shown}{more}, "
        f"aligned across the sequence flanking the gap.{identity} Confidence is "
        f"{confidence:.0%}; this is an inference from homologous sequence, not an "
        "observation, and should be verified before being relied on."
    )
