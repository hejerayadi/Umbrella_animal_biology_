"""MAFFT multiple sequence alignment.

Primary: EBI REST API (https://www.ebi.ac.uk/Tools/services/rest/mafft).
Fallback: a locally installed MAFFT binary.

The local binary is located by ``binaries.resolve_binary``:
``$MAFFT_BINARY`` first, then ``mafft.bat`` / ``mafft`` on PATH, then the
legacy bundled path kept below for backward compatibility.

Callers must pass FASTA identifiers without whitespace — MAFFT truncates a
header at the first space. Use ``tools.taxon_ids`` to map scientific names
to safe ids and back.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path

from .binaries import (
    MAFFT_CANDIDATES,
    MAFFT_ENV_VAR,
    describe_search,
    resolve_binary,
)

_logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

# Legacy bundled location — last resort, kept so existing setups keep working.
_LOCAL_MAFFT = str(_PROJECT_ROOT / "mafft" / "mafft-win" / "mafft.bat")


def resolve_mafft() -> str | None:
    """Path to a usable MAFFT executable, or ``None`` if none is installed."""
    return resolve_binary(MAFFT_ENV_VAR, MAFFT_CANDIDATES, _LOCAL_MAFFT)

EBI_RUN_URL = "https://www.ebi.ac.uk/Tools/services/rest/mafft/run/"
EBI_STATUS_URL = "https://www.ebi.ac.uk/Tools/services/rest/mafft/status/"
EBI_RESULT_URL = "https://www.ebi.ac.uk/Tools/services/rest/mafft/result/"


class MAFFTError(Exception):
    """Raised when MAFFT alignment fails."""


def align(
    sequences: dict[str, str],
    method: str = "auto",
    timeout: int = 120,
) -> str:
    """Align sequences using MAFFT.

    Tries EBI REST API first, falls back to local binary.

    Parameters
    ----------
    sequences : dict[str, str]
        Mapping of species name → protein/nucleotide sequence.
    method : str
        MAFFT method (auto, FFT-NS-2, L-INS-i, etc.)
    timeout : int
        Max seconds to wait for alignment.

    Returns
    -------
    str
        Aligned sequences in FASTA format.
    """
    if not sequences:
        raise MAFFTError("No sequences provided for alignment.")

    fasta_input = _dict_to_fasta(sequences)

    # Primary: EBI REST API
    try:
        return _align_ebi(fasta_input, timeout)
    except MAFFTError as exc:
        _logger.warning("[MAFFT] EBI API failed (%s), trying local binary...", exc)

    # Fallback: local binary
    binary = resolve_mafft()
    if binary:
        try:
            return _align_local(fasta_input, method, timeout, binary=binary)
        except MAFFTError as exc:
            _logger.warning("[MAFFT] local binary failed (%s)", exc)
            raise

    raise MAFFTError(
        "EBI API failed and no local MAFFT executable was found. Searched: "
        + describe_search(MAFFT_ENV_VAR, MAFFT_CANDIDATES, _LOCAL_MAFFT)
    )


def _align_ebi(fasta_input: str, timeout: int) -> str:
    """Align using EBI REST API."""
    try:
        import httpx
        from urllib.parse import urlencode

        email = os.environ.get("EBI_MAFFT_EMAIL", "evolution-agent@umbrella.local")

        params = {
            "email": email,
            "sequence": fasta_input,
            "stype": "protein",
            "format": "fasta",
            "order": "aligned",
            "maxiterate": "2",
            "nbtree": "2",
        }

        with httpx.Client(timeout=60) as client:
            response = client.post(
                EBI_RUN_URL,
                content=urlencode(params).encode("utf-8"),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            job_id = response.text.strip()

        if not job_id:
            raise MAFFTError("No job ID returned from EBI")

        _logger.info("[MAFFT] EBI job submitted: %s", job_id)

        # Poll for completion
        start = time.time()
        with httpx.Client(timeout=30) as client:
            while time.time() - start < timeout:
                try:
                    r = client.get(f"{EBI_STATUS_URL}{job_id}")
                    r.raise_for_status()
                    status = r.text.strip()
                    if status == "FINISHED":
                        break
                    if status in ("ERROR", "FAILED"):
                        raise MAFFTError(f"EBI job failed with status: {status}")
                    time.sleep(3)
                except httpx.HTTPStatusError:
                    time.sleep(3)

        # Fetch result
        with httpx.Client(timeout=60) as client:
            r = client.get(f"{EBI_RESULT_URL}{job_id}/aln-fasta")
            r.raise_for_status()
            result = r.text.strip()
            if not result:
                raise MAFFTError("EBI returned empty alignment")
            _logger.info("[MAFFT] EBI alignment complete (%d chars)", len(result))
            return result

    except ImportError:
        raise MAFFTError("httpx not installed")
    except MAFFTError:
        raise
    except Exception as exc:
        raise MAFFTError(f"EBI API error: {exc}") from exc


def _align_local(
    fasta_input: str, method: str, timeout: int, binary: str | None = None
) -> str:
    """Run MAFFT using the local binary."""
    import subprocess

    executable = binary or resolve_mafft()
    if not executable:
        raise MAFFTError(
            "No local MAFFT executable was found. Searched: "
            + describe_search(MAFFT_ENV_VAR, MAFFT_CANDIDATES, _LOCAL_MAFFT)
        )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".fasta", delete=False, encoding="utf-8"
    ) as inp:
        inp.write(fasta_input)
        inp_path = inp.name

    try:
        cmd = [executable, "--auto", inp_path]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise MAFFTError(f"MAFFT failed (exit {result.returncode}): {result.stderr[:500]}")

        output = result.stdout.strip()
        if not output:
            raise MAFFTError("MAFFT returned empty alignment")

        _logger.info("[MAFFT] local alignment complete (%d chars)", len(output))
        return output

    except subprocess.TimeoutExpired:
        raise MAFFTError(f"MAFFT timed out after {timeout}s")
    finally:
        os.unlink(inp_path)


def _dict_to_fasta(sequences: dict[str, str]) -> str:
    """Convert dict to FASTA format."""
    lines = []
    for name, seq in sequences.items():
        lines.append(f">{name}")
        for i in range(0, len(seq), 60):
            lines.append(seq[i : i + 60])
    return "\n".join(lines) + "\n"
