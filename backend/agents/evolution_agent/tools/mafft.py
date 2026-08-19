"""MAFFT multiple sequence alignment via EBI REST API.

Uses the EBI Job Dispatcher REST API for MAFFT alignment.
No API key needed — just an email address.

API flow:
    1. POST sequences to EBI MAFFT /run endpoint
    2. Poll /status until FINISHED
    3. Get /result/{jobId}/aln-fasta

API endpoint: https://www.ebi.ac.uk/Tools/services/rest/mafft
"""

from __future__ import annotations

import os
import time
import logging
from typing import Any
from urllib.parse import urlencode

import httpx
from dotenv import load_dotenv

_logger = logging.getLogger(__name__)

# Load .env from project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"), override=False)
load_dotenv(os.path.join(_PROJECT_ROOT, "backend", "agents", "evolution_agent", ".env"), override=False)

_BASE_URL = "https://www.ebi.ac.uk/Tools/services/rest/mafft"


class MAFFTError(Exception):
    """Raised when MAFFT alignment fails."""


def align(
    sequences: dict[str, str],
    email: str | None = None,
    method: str = "auto",
    timeout: int = 300,
    poll_interval: int = 3,
) -> str:
    """Align sequences using EBI MAFFT REST API.

    Parameters
    ----------
    sequences : dict[str, str]
        Mapping of species name → protein/nucleotide sequence.
    email : str, optional
        Email for EBI API. Defaults to EBI_MAFFT_EMAIL env var.
    method : str
        MAFFT method (auto, FFT-NS-2, L-INS-i, etc.)
    timeout : int
        Max seconds to wait for alignment.
    poll_interval : int
        Seconds between status polls.

    Returns
    -------
    str
        Aligned sequences in FASTA format.

    Raises
    ------
    MAFFTError
        If API call fails or times out.
    """
    if not sequences:
        raise MAFFTError("No sequences provided for alignment.")

    email = email or os.environ.get("EBI_MAFFT_EMAIL", "evolution-agent@umbrella.local")
    fasta_input = _dict_to_fasta(sequences)

    try:
        # Submit job
        job_id = _submit_job(fasta_input, email, method)
        _logger.info("[MAFFT] job submitted: %s", job_id)

        # Poll for completion
        _poll_job(job_id, timeout, poll_interval)
        _logger.info("[MAFFT] job completed: %s", job_id)

        # Get aligned result
        result = _get_result(job_id)
        return result

    except httpx.HTTPError as exc:
        raise MAFFTError(f"EBI API error: {exc}") from exc


def _submit_job(fasta: str, email: str, method: str) -> str:
    """Submit alignment job to EBI MAFFT API."""
    url = f"{_BASE_URL}/run/"

    params = {
        "email": email,
        "sequence": fasta,
        "stype": "protein",
        "format": "fasta",
        "order": "aligned",
    }

    # Set method
    if method == "auto":
        params["maxiterate"] = "2"
        params["nbtree"] = "2"
    else:
        params["maxiterate"] = "1000"
        params["nbtree"] = "1"

    with httpx.Client(timeout=60) as client:
        response = client.post(
            url,
            content=urlencode(params).encode("utf-8"),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()

        job_id = response.text.strip()
        if not job_id:
            raise MAFFTError("No job ID returned from EBI")

        return job_id


def _poll_job(job_id: str, timeout: int, poll_interval: int) -> None:
    """Poll job status until completion."""
    start = time.time()
    url = f"{_BASE_URL}/status/{job_id}"

    with httpx.Client(timeout=30) as client:
        while time.time() - start < timeout:
            try:
                response = client.get(url)
                response.raise_for_status()

                status = response.text.strip()

                if status == "FINISHED":
                    return

                if status in ("ERROR", "FAILED"):
                    raise MAFFTError(f"EBI job failed with status: {status}")

                time.sleep(poll_interval)

            except httpx.HTTPError:
                time.sleep(poll_interval)

    raise MAFFTError(f"EBI job timed out after {timeout}s")


def _get_result(job_id: str) -> str:
    """Get alignment result from EBI."""
    url = f"{_BASE_URL}/result/{job_id}/aln-fasta"

    with httpx.Client(timeout=60) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.text


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
        for i in range(0, len(seq), 60):
            lines.append(seq[i : i + 60])
    return "\n".join(lines) + "\n"
