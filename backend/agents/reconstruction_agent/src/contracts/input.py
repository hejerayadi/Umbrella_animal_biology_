"""What the orchestrator sends this agent.

This mirrors the repo-wide `AgentRequest` (instruction + context) but adds the
typed reading of `context` that this agent specifically expects. The
orchestrator is free to send a bare instruction; everything here is optional
and falls back to parsing the instruction text.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SequenceInput(BaseModel):
    """A nucleotide sequence handed to us directly, rather than by accession.

    `residues` may contain IUPAC ambiguity codes; runs of `N` are what the gap
    detector looks for.
    """

    model_config = ConfigDict(frozen=True)

    identifier: str = Field(description="Caller's name for this sequence, used in output.")
    residues: str = Field(description="Nucleotide string, IUPAC codes permitted.")
    organism: str | None = Field(
        default=None, description="Scientific name, e.g. 'Mammuthus primigenius'."
    )
    description: str | None = None


def _organism_from(context: dict[str, Any]) -> str | None:
    """The target organism, under whichever key the caller used.

    `organism` is this agent's own name for it, but nothing upstream writes
    that key. The orchestrator's extractor seeds `species` from the user's
    question, and the Genome agent publishes a `species_record`; both name the
    same thing, and reading only `organism` meant the organism was silently
    dropped on every orchestrated run. It is not cosmetic - reference ranking
    and the Evolution Agent escalation are both written in terms of it.
    """
    # `species_record.scientific_name` is checked first, ahead of the looser
    # keys. The orchestrator sends both: a real payload carries
    # `species: "polar bear"` alongside
    # `species_record: {"scientific_name": "Ursus maritimus"}`, and reading
    # `species` first took the common name. Nothing downstream can use it -
    # `organism_affinity` compares binomials by design, so every reference then
    # scored nothing on relatedness - and the escalation prompt named the animal
    # in a form no sequence database indexes. An explicit scientific name wins.
    record = context.get("species_record")
    if isinstance(record, dict):
        value = record.get("scientific_name")
        if isinstance(value, str) and value.strip():
            return value.strip()

    for key in ("organism", "species", "species_name"):
        value = context.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    # Last resort: whatever else the record calls the animal, common name
    # included. Better than nothing for the prompt, and NCBI's taxonomy resolves
    # common names even though the binomial affinity heuristic does not.
    if isinstance(record, dict):
        for key in ("organism", "name", "common_name"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return None


class ReconstructionRequest(BaseModel):
    """The typed form of one reconstruction job.

    Built by `from_agent_request`, which is the only place that knows how the
    orchestrator's loose `context` dict maps onto these fields.
    """

    model_config = ConfigDict(frozen=True)

    instruction: str = Field(description="Natural-language task from the orchestrator.")

    # Two ways to point at the sequence to repair. At least one must be present
    # by the time the service runs; validation lives in the service so that a
    # missing sequence becomes a FAILED AgentResult rather than a 422.
    sequence: SequenceInput | None = None
    accession: str | None = Field(
        default=None, description="NCBI accession to fetch the target sequence from."
    )

    organism: str | None = None
    # Restricts reference search to these organisms when the caller already
    # knows which relatives are informative.
    reference_organisms: list[str] = Field(default_factory=list)

    max_gap_length: int | None = Field(
        default=None, description="Per-request override of the configured cap."
    )
    min_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Per-request override of the configured floor."
    )

    raw_context: dict[str, Any] = Field(
        default_factory=dict, description="The orchestrator's context dict, unmodified."
    )

    @classmethod
    def from_agent_request(cls, instruction: str, context: dict[str, Any]) -> ReconstructionRequest:
        """Read an orchestrator request into this shape.

        Deliberately permissive: unknown keys are kept in `raw_context` and
        ignored rather than rejected, because the orchestrator's context is a
        shared scratchpad that other agents also write to.
        """
        organism = _organism_from(context)

        sequence: SequenceInput | None = None
        raw_sequence = context.get("sequence")
        if isinstance(raw_sequence, dict) and raw_sequence.get("residues"):
            sequence = SequenceInput(
                identifier=str(raw_sequence.get("identifier") or "input_sequence"),
                residues=str(raw_sequence["residues"]),
                organism=raw_sequence.get("organism") or organism,
                description=raw_sequence.get("description"),
            )
        elif isinstance(raw_sequence, str) and raw_sequence.strip():
            # A bare string is accepted as the residues themselves.
            sequence = SequenceInput(
                identifier="input_sequence",
                residues=raw_sequence,
                organism=organism,
            )

        references = context.get("reference_organisms") or []
        if isinstance(references, str):
            references = [references]

        return cls(
            instruction=instruction,
            sequence=sequence,
            accession=context.get("accession") or context.get("sequence_accession"),
            organism=organism,
            reference_organisms=[str(item) for item in references],
            max_gap_length=context.get("max_gap_length"),
            min_confidence=context.get("min_confidence"),
            raw_context=context,
        )
