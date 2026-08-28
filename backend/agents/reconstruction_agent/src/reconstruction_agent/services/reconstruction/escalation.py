"""Asking the Genome Agent for a target this agent cannot choose itself.

`card.json` says it plainly: this agent works on a specific sequence and does
not pick one. A request naming only a species has no accession to fetch, no
flanks to search with and no coordinates to trust - so the honest answer is to
name what is missing and say which agent can supply it.

Two rules from the orchestrator's worker node shape everything here.

**Never call the other agent.** Escalation is a status, not a request. The
orchestrator routes it; this agent never imports or HTTP-calls a sibling.

**The escalation must add something to the shared context.** `worker_node.py`
snapshots `sorted(state.context)` each time an agent escalates and force-fails
a second escalation whose signature is unchanged - that guard is what stopped
one unmet dependency from becoming 36 steps of NCBI calls. So the payload below
introduces a key that was not there before, carrying what this agent worked out
about the request. Escalating with an empty hand looks identical to looping.
"""

from __future__ import annotations

from typing import Any

#: The agent that resolves a species to a specific gapped scaffold. Named as a
#: routing hint, not as an address: nothing here opens a connection to it.
GENOME_AGENT = "Genome Agent"

#: The context key this escalation contributes. Deliberately not one the Genome
#: Agent already sets, so the signature the orchestrator compares actually
#: changes and the escalation is not read as a loop.
FINDINGS_KEY = "reconstruction_request"

#: Keys that would mean a target is present. Checked against the raw context
#: rather than the parsed request, because parsing is what failed.
_TARGET_KEYS = (
    "sequence_accession",
    "accession",
    "nucleotide_accession",
    "refseq_accession",
    "sequence",
    "residues",
    "nucleotide_sequence",
)

#: A species name is what makes the Genome Agent able to help. Without one,
#: escalating would ask it to resolve nothing.
_ORGANISM_KEYS = ("scientific_name", "species", "organism", "species_name")


def should_escalate(context: dict[str, Any] | None) -> bool:
    """Whether the Genome Agent could supply what this request is missing.

    True only when a species is named and no sequence is. A request naming
    neither cannot be helped by anyone, and escalating it would spend a step to
    be told the same thing.
    """
    data = context or {}
    if any(_present(data, key) for key in _TARGET_KEYS):
        return False
    return any(_present(data, key) for key in _ORGANISM_KEYS)


def build_escalation(
    instruction: str, context: dict[str, Any] | None, reason: str
) -> tuple[str, dict[str, Any]]:
    """The prompt for the Genome Agent, and the findings to carry with it."""
    data = context or {}
    organism = next(
        (str(data[key]).strip() for key in _ORGANISM_KEYS if _present(data, key)),
        "the named species",
    )

    prompt = (
        f"Identify a gapped genome assembly for {organism} and return its nucleotide "
        "accession, assembly level and the coordinates of its unresolved regions. "
        "The Reconstruction Agent needs a specific sequence and cannot select one."
    )

    findings: dict[str, Any] = {
        FINDINGS_KEY: {
            "status": "awaiting_target",
            "needs": ["sequence_accession", "target_gaps"],
            "organism": organism,
            "reason": reason,
            "original_instruction": instruction,
            # What was already usable, so the Genome Agent does not re-derive it.
            "known": {
                key: data[key]
                for key in ("assembly_id", "assembly", "assembly_level")
                if _present(data, key)
            },
        }
    }
    return prompt, findings


def _present(data: dict[str, Any], key: str) -> bool:
    value = data.get(key)
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)
