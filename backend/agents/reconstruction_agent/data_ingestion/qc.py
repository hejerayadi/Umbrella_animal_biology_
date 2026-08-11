from __future__ import annotations

from pathlib import Path

from . import CleanedScaffold


def clean_scaffolds(raw_fasta_path: Path, accession: str, species_id: str) -> list[CleanedScaffold]:
    """Clean and validate FASTA scaffolds before ingestion."""
    if not raw_fasta_path.exists():
        return []

    text = raw_fasta_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    records: list[CleanedScaffold] = []
    header: str | None = None
    sequence_parts: list[str] = []

    for line in lines:
        if line.startswith(">"):
            if header is None:
                header = line[1:].split()[0]
                continue
            scaffold = _build_scaffold(header, sequence_parts, accession, species_id)
            if scaffold is not None:
                records.append(scaffold)
            header = line[1:].split()[0]
            sequence_parts = []
            continue
        sequence_parts.append(line.upper())

    if header is not None:
        scaffold = _build_scaffold(header, sequence_parts, accession, species_id)
        if scaffold is not None:
            records.append(scaffold)

    return records


def _build_scaffold(scaffold_id: str, parts: list[str], accession: str, species_id: str) -> CleanedScaffold | None:
    sequence = "".join(parts).replace(" ", "")
    if not sequence or len(sequence) < 2000:
        return None
    if any(base not in {"A", "C", "G", "T", "N"} for base in sequence):
        return None
    if sequence.count("N") / len(sequence) > 0.50:
        return None
    if sequence.startswith(">"):
        return None
    header = f">{accession}|{species_id}|{scaffold_id}"
    return CleanedScaffold(accession=accession, species_id=species_id, scaffold_id=scaffold_id, sequence=sequence, header=header)
