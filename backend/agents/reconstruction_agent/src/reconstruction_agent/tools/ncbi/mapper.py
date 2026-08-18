"""Translate NCBI payloads into domain objects.

Kept apart from the client so that parsing is testable against saved fixtures
with no network involved - which is what the `tests/tools` suite does.
"""
from __future__ import annotations

import re
from typing import Any

from ...domain.models import Reference

# NCBI FASTA headers look like:
#   >NC_007596.2 Mammuthus primigenius mitochondrion, complete genome
# The organism is the first two words of the description often enough to be a
# useful default, and `esummary` is the reliable source when it matters.
_ORGANISM_PATTERN = re.compile(r"^\S+\s+([A-Z][a-z]+ [a-z]+)")


def parse_fasta(raw: str) -> list[tuple[str, str, str]]:
    """Split multi-FASTA into (accession, description, residues) triples.

    Hand-rolled rather than via Biopython's parser because the input is a
    string already in memory and this avoids a file-handle round trip.
    """
    records: list[tuple[str, str, str]] = []
    accession = ""
    description = ""
    residues: list[str] = []

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith(">"):
            if accession:
                records.append((accession, description, "".join(residues)))
            header = line[1:].strip()
            accession, _, description = header.partition(" ")
            residues = []
        elif accession:
            residues.append(line.upper())

    if accession:
        records.append((accession, description, "".join(residues)))

    return records


def to_references(raw_fasta: str, *, source: str = "ncbi") -> list[Reference]:
    """Domain `Reference` objects for every record in a multi-FASTA block."""
    references: list[Reference] = []

    for accession, description, residues in parse_fasta(raw_fasta):
        match = _ORGANISM_PATTERN.match(f"{accession} {description}")
        references.append(
            Reference(
                accession=accession,
                organism=match.group(1) if match else None,
                description=description or None,
                residues=residues or None,
                source=source,
            )
        )

    return references


def summaries_to_references(
    summary: dict[str, Any], *, source: str = "ncbi"
) -> list[Reference]:
    """References from an `esummary` payload, without residues.

    The payload's `uids` list is the authority on which keys are records; the
    dict also carries bookkeeping entries that are not.
    """
    references: list[Reference] = []

    for uid in summary.get("uids", []):
        record = summary.get(uid)
        if not isinstance(record, dict):
            continue
        references.append(
            Reference(
                accession=record.get("accessionversion") or str(uid),
                organism=record.get("organism"),
                description=record.get("title"),
                source=source,
            )
        )

    return references
