"""Safe FASTA identifiers for taxa whose names contain spaces.

MAFFT and IQ-TREE both truncate a FASTA identifier at the first whitespace.
Feeding them ``>Homo sapiens`` yields a tree labelled ``Homo``, which silently
destroys species identity — ``Homo sapiens`` and ``Homo erectus`` collapse
onto the same label.

The fix is a bidirectional mapping applied around the external tools:

    Homo sapiens  ->  taxon_0001  ->  (MAFFT, IQ-TREE)  ->  'Homo sapiens'

Identifiers are allocated sequentially, so two distinct scientific names can
never collide regardless of how similar they are.
"""

from __future__ import annotations

import re

_SAFE_ID = re.compile(r"\btaxon_\d{4}\b")

# Newick reserved characters: a label containing any of them must be quoted.
_NEEDS_QUOTING = set(" \t\n()[]':;,")


def make_mapping(names: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Allocate one safe id per name, preserving input order.

    Returns ``(name_to_id, id_to_name)``. A repeated name reuses its id.
    """
    name_to_id: dict[str, str] = {}
    id_to_name: dict[str, str] = {}
    for name in names:
        if name in name_to_id:
            continue
        safe = f"taxon_{len(name_to_id) + 1:04d}"
        name_to_id[name] = safe
        id_to_name[safe] = name
    return name_to_id, id_to_name


def to_safe_sequences(
    sequences: dict[str, str], name_to_id: dict[str, str]
) -> dict[str, str]:
    """Re-key ``sequences`` by safe id, ready for FASTA output."""
    return {name_to_id[name]: seq for name, seq in sequences.items()}


def restore_fasta(fasta: str, id_to_name: dict[str, str]) -> str:
    """Put the full scientific names back into a FASTA header block."""
    out: list[str] = []
    for line in fasta.splitlines():
        if line.startswith(">"):
            token = line[1:].strip().split()[0] if line[1:].strip() else ""
            out.append(f">{id_to_name.get(token, token)}")
        else:
            out.append(line)
    return "\n".join(out) + ("\n" if fasta.endswith("\n") else "")


def quote_newick_label(name: str) -> str:
    """Quote a Newick label when the name needs it.

    Newick single-quoting doubles any embedded single quote.
    """
    if any(ch in _NEEDS_QUOTING for ch in name):
        return "'" + name.replace("'", "''") + "'"
    return name


def restore_newick(newick: str, id_to_name: dict[str, str]) -> str:
    """Replace every safe id in a Newick string by its quoted full name."""

    def _swap(match: re.Match) -> str:
        token = match.group(0)
        name = id_to_name.get(token)
        return quote_newick_label(name) if name else token

    return _SAFE_ID.sub(_swap, newick)


def parse_fasta(fasta: str) -> dict[str, str]:
    """Minimal FASTA parser: identifier is the first whitespace-free token."""
    result: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    for line in fasta.splitlines():
        line = line.strip()
        if line.startswith(">"):
            if name is not None:
                result[name] = "".join(chunks)
            header = line[1:].strip()
            name = header.split()[0] if header else ""
            chunks = []
        elif line:
            chunks.append(line)
    if name is not None:
        result[name] = "".join(chunks)
    return result
