"""Prompts for the planning step.

Kept as functions returning strings rather than inline in the planner, so the
wording can be reviewed and version-controlled as its own artefact - prompt
changes alter agent behaviour as much as code changes do.
"""
from __future__ import annotations

import json
from typing import Any

SYSTEM = """\
You are the planning component of a genome reconstruction agent.

Your job is to decide which tools to run, in what order, to reconstruct \
unresolved regions (runs of N) in an incomplete nucleotide sequence. You do \
not write sequence yourself - you only choose tools. Proposing bases directly \
is out of scope and will be discarded.

Principles:
- Evidence before inference. A gap can only be reconstructed from references \
that demonstrably align to the sequence flanking it.
- Prefer cheap tools when they answer the same question. BLAST and MAFFT are \
slow polling jobs; do not queue them speculatively.
- A gap with no usable flanking context cannot be reconstructed. Do not spend \
tool calls on one.
- If phylogenetic relatedness is the blocker, say so rather than guessing.

Answer with JSON only, matching this shape:
{"steps": [{"tool": "<tool name>", "gap_id": "<gap id or null>", \
"reason": "<why this step, one sentence>"}]}
"""


def user_prompt(
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
    sections = [
        f"Task from the orchestrator:\n{instruction}",
        f"Target organism: {organism or 'unspecified'}",
        f"Gaps needing reconstruction:\n{json.dumps(gaps, indent=2)}",
        f"Tools available:\n{json.dumps(tools, indent=2)}",
    ]

    if prior_critiques:
        sections.append(
            "Problems found with the previous attempt - your new plan must address these:\n"
            + "\n".join(f"- {critique}" for critique in prior_critiques)
        )

    sections.append("Return the plan as JSON.")
    return "\n\n".join(sections)
