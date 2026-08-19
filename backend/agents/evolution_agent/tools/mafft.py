"""MAFFT multiple sequence alignment wrapper.

Calls the MAFFT CLI (mafft.bat on Windows) via subprocess.
Input: raw sequences as FASTA string or dict of {species: sequence}
Output: aligned sequences in FASTA format

Pipeline position:
    Sequences → MAFFT → aligned sequences → IQ-TREE
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

# Default MAFFT path (Windows install)
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_MAFFT_DEFAULT = str(_PROJECT_ROOT / "mafft" / "mafft-win" / "mafft.bat")


class MAFFTError(Exception):
    """Raised when MAFFT alignment fails."""


def align(
    sequences: dict[str, str],
    mafft_path: str | None = None,
    method: str = "--auto",
    timeout: int = 300,
) -> str:
    """Align sequences using MAFFT.

    Parameters
    ----------
    sequences : dict[str, str]
        Mapping of species name → protein/nucleotide sequence.
    mafft_path : str, optional
        Path to mafft.bat. Defaults to bundled install.
    method : str
        MAFFT method (--auto, --localpair, --globalpair, etc.)
    timeout : int
        Max seconds to wait for MAFFT.

    Returns
    -------
    str
        Aligned sequences in FASTA format.

    Raises
    ------
    MAFFTError
        If MAFFT fails or times out.
    """
    if not sequences:
        raise MAFFTError("No sequences provided for alignment.")

    mafft = mafft_path or _MAFFT_DEFAULT
    if not os.path.exists(mafft):
        raise MAFFTError(f"MAFFT not found at: {mafft}")

    # Build FASTA input
    fasta_input = _dict_to_fasta(sequences)

    # Write to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".fasta", delete=False, encoding="utf-8"
    ) as f:
        f.write(fasta_input)
        input_path = f.name

    try:
        # Call MAFFT
        cmd = [mafft, method, "--quiet", input_path]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode != 0:
            raise MAFFTError(
                f"MAFFT failed (exit {result.returncode}): {result.stderr[:500]}"
            )

        output = result.stdout.strip()
        if not output:
            raise MAFFTError("MAFFT returned empty output.")

        return output

    except subprocess.TimeoutExpired:
        raise MAFFTError(f"MAFFT timed out after {timeout}s.")
    finally:
        os.unlink(input_path)


def parse_aligned(fasta_str: str) -> dict[str, str]:
    """Parse aligned FASTA string into {species: aligned_sequence}."""
    result = {}
    current_name = None
    current_seq = []

    for line in fasta_str.splitlines():
        line = line.strip()
        if line.startswith(">"):
            if current_name is not None:
                result[current_name] = "".join(current_seq)
            current_name = line[1:].strip()
            current_seq = []
        elif line:
            current_seq.append(line)

    if current_name is not None:
        result[current_name] = "".join(current_seq)

    return result


def _dict_to_fasta(sequences: dict[str, str]) -> str:
    """Convert dict to FASTA format."""
    lines = []
    for name, seq in sequences.items():
        lines.append(f">{name}")
        # Wrap at 60 chars
        for i in range(0, len(seq), 60):
            lines.append(seq[i : i + 60])
    return "\n".join(lines) + "\n"
