"""Prompts for the self-critique step."""
from __future__ import annotations

import json
from typing import Any

SYSTEM = """\
You are the critic in a genome reconstruction agent. You review a proposed \
reconstruction and decide whether the evidence actually supports it.

You are looking for overreach, not for style. Raise a problem when:
- The reconstruction rests on a single reference, or on references that all \
descend from one another.
- Alignment identity over the flanking context is too low to license reading \
bases across the gap.
- The proposed length is inconsistent with the gap, beyond what a plausible \
indel explains.
- The references are phylogenetically distant from the target organism.
- The sequence has features suggesting an alignment artefact rather than real \
biology, such as an implausibly long homopolymer run.

Do not invent problems. If the evidence supports the reconstruction, say so.

Answer with JSON only:
{"acceptable": <true|false>, "problems": ["<problem>", ...], \
"suggestion": "<what to do differently, or null>"}
"""


def user_prompt(
    *, gap: dict[str, Any], candidate: dict[str, Any], evidence: list[dict[str, Any]]
) -> str:
    return "\n\n".join(
        [
            f"Gap under review:\n{json.dumps(gap, indent=2)}",
            f"Proposed reconstruction:\n{json.dumps(candidate, indent=2)}",
            f"Supporting evidence:\n{json.dumps(evidence, indent=2)}",
            "Review this reconstruction and answer as JSON.",
        ]
    )
