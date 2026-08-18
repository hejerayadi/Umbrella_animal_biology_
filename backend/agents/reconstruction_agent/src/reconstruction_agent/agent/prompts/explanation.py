"""Prompts for turning evidence into an explanation a biologist can check."""
from __future__ import annotations

import json
from typing import Any

SYSTEM = """\
You explain genome reconstruction results to a working biologist.

Write the account of how a reconstruction was arrived at: which references \
supported it, how well they aligned, and what would make it more or less \
trustworthy. The reader wants to judge the result, not be reassured about it.

Rules:
- State the evidence and its limits. Never overstate confidence.
- Name the specific accessions and organisms involved.
- If the reconstruction rests on thin evidence, lead with that.
- Plain prose, two to four sentences. No headings, no bullet points.
- Never present a reconstruction as an observed sequence. It is an inference.
"""


def user_prompt(
    *, gap: dict[str, Any], candidate: dict[str, Any], evidence: list[dict[str, Any]]
) -> str:
    return "\n\n".join(
        [
            f"Gap:\n{json.dumps(gap, indent=2)}",
            f"Reconstruction:\n{json.dumps(candidate, indent=2)}",
            f"Evidence:\n{json.dumps(evidence, indent=2)}",
            "Explain how this reconstruction was reached and how far it should be trusted.",
        ]
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

    Says exactly the same things the prompt above asks for, from the numbers
    already computed - so a run without an LLM still produces a reviewable
    account rather than an empty field.
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
