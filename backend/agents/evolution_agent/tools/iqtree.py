"""IQ-TREE phylogenetic tree builder via CIPRES REST API.

Uses the CIPRES Science Gateway REST API for tree building.
Requires registration and API key.

API flow:
    1. Upload alignment to CIPRES
    2. Submit IQ-TREE job
    3. Poll for completion
    4. Download tree file

API endpoint: https://cipresrest.sdsc.edu/cipresrest/v1
Tool ID: IQTREE3_REST (IQ-TREE 3)
"""

from __future__ import annotations

import os
import re
import time
import logging
import xml.etree.ElementTree as ET
from typing import Any

import httpx
from dotenv import load_dotenv

_logger = logging.getLogger(__name__)

# Load .env from project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"), override=False)
load_dotenv(os.path.join(_PROJECT_ROOT, "backend", "agents", "evolution_agent", ".env"), override=False)


class IQTreeError(Exception):
    """Raised when IQ-TREE fails."""


def build_tree(
    alignment: str | dict[str, str],
    model: str = "MFP",
    bootstrap: int = 1000,
    timeout: int = 1800,
    poll_interval: int = 10,
) -> PhyloResult:
    """Build a phylogenetic tree using CIPRES IQ-TREE API.

    Parameters
    ----------
    alignment : str or dict[str, str]
        Either a FASTA string (aligned) or a dict of {species: aligned_sequence}.
    model : str
        Substitution model or "MFP" for ModelFinder.
    bootstrap : int
        Number of UFBoot replicates (0 to disable).
    timeout : int
        Max seconds to wait for tree building.
    poll_interval : int
        Seconds between status polls.

    Returns
    -------
    PhyloResult
        Newick tree, model, bootstrap support, confidence values.

    Raises
    ------
    IQTreeError
        If CIPRES API fails or times out.
    """
    # Get CIPRES credentials from .env
    username = os.environ.get("CIPRES_USERNAME")
    password = os.environ.get("CIPRES_PASSWORD")
    api_key = os.environ.get("CIPRES_API_KEY")

    if not all([username, password, api_key]):
        raise IQTreeError(
            "CIPRES credentials not configured. "
            "Set CIPRES_USERNAME, CIPRES_PASSWORD, and CIPRES_API_KEY in .env"
        )

    # Prepare alignment
    if isinstance(alignment, dict):
        alignment_str = _dict_to_fasta(alignment)
    else:
        alignment_str = alignment

    # Adjust bootstrap for small datasets
    n_sequences = alignment_str.count(">")
    if n_sequences < 4:
        bootstrap = 0

    try:
        # Submit job
        job_url = _submit_job(alignment_str, model, bootstrap, username, password, api_key)
        _logger.info("[IQ-TREE] job submitted: %s", job_url)

        # Poll for completion
        result_files = _poll_job(job_url, username, password, api_key, timeout, poll_interval)
        _logger.info("[IQ-TREE] job completed, %d result files", len(result_files))

        # Download and parse results
        return _parse_results(result_files, username, password, api_key)

    except httpx.HTTPError as exc:
        raise IQTreeError(f"CIPRES API error: {exc}") from exc


def _submit_job(
    alignment: str,
    model: str,
    bootstrap: int,
    username: str,
    password: str,
    api_key: str,
) -> str:
    """Submit IQ-TREE job to CIPRES API."""
    url = f"https://cipresrest.sdsc.edu/cipresrest/v1/job/{username}"

    # Upload alignment file
    files = {
        "input.infile_": ("alignment.fasta", alignment.encode("utf-8"), "text/plain")
    }

    # IQ-TREE parameters
    data = {
        "tool": "IQTREE3_REST",
        "input.infile_": alignment,
        "metadata.statusEmail": "false",
    }

    # Add model parameters
    if model == "MFP":
        data["input.model_"] = "MFP"  # ModelFinder
    else:
        data["input.model_"] = model

    # Add bootstrap
    if bootstrap > 0:
        data["input.n bootstrap_"] = str(bootstrap)
        data["input.bb_"] = str(bootstrap)  # UFBoot

    headers = {
        "cipres-appkey": api_key,
    }

    with httpx.Client(timeout=60) as client:
        response = client.post(
            url,
            data=data,
            files=files,
            headers=headers,
            auth=(username, password),
        )
        response.raise_for_status()

        # Extract job URL from response
        root = ET.fromstring(response.text)
        job_url = _find_element(root, "selfUri")
        if not job_url:
            raise IQTreeError("No job URL returned from CIPRES")

        return job_url


def _poll_job(
    job_url: str,
    username: str,
    password: str,
    api_key: str,
    timeout: int,
    poll_interval: int,
) -> list[str]:
    """Poll job status until completion, return result file URLs."""
    start = time.time()
    headers = {"cipres-appkey": api_key}

    with httpx.Client(timeout=30) as client:
        while time.time() - start < timeout:
            try:
                response = client.get(
                    job_url,
                    headers=headers,
                    auth=(username, password),
                )
                response.raise_for_status()

                root = ET.fromstring(response.text)

                # Check terminal stage
                terminal = _find_element(root, "terminalStage")
                if terminal and terminal.lower() == "true":
                    # Get results URL
                    results_url = _find_element(root, "resultsUri")
                    if results_url:
                        return _get_result_files(results_url, username, password, api_key)

                # Check for errors
                stage = _find_element(root, "currentStage")
                if stage and "error" in stage.lower():
                    raise IQTreeError(f"CIPRES job failed at stage: {stage}")

                time.sleep(poll_interval)

            except ET.ParseError:
                time.sleep(poll_interval)

    raise IQTreeError(f"CIPRES job timed out after {timeout}s")


def _get_result_files(
    results_url: str,
    username: str,
    password: str,
    api_key: str,
) -> list[str]:
    """Get list of result file URLs."""
    headers = {"cipres-appkey": api_key}
    file_urls = []

    with httpx.Client(timeout=30) as client:
        response = client.get(
            results_url,
            headers=headers,
            auth=(username, password),
        )
        response.raise_for_status()

        root = ET.fromstring(response.text)

        # Find all jobfile elements
        for jobfile in root.iter("jobfile"):
            download_url = _find_element(jobfile, "downloadUri")
            if download_url:
                file_urls.append(download_url)

    return file_urls


def _parse_results(
    file_urls: list[str],
    username: str,
    password: str,
    api_key: str,
) -> PhyloResult:
    """Download and parse IQ-TREE result files."""
    headers = {"cipres-appkey": api_key}
    newick_tree = ""
    model_name = "unknown"
    bootstrap_support = {}

    with httpx.Client(timeout=60) as client:
        for url in file_urls:
            response = client.get(
                url,
                headers=headers,
                auth=(username, password),
            )
            response.raise_for_status()

            # Identify file type by URL or content
            if ".treefile" in url or ".nwk" in url:
                newick_tree = response.text.strip()
            elif ".log" in url:
                log_content = response.text
                model_name = _parse_model(log_content)
                bootstrap_support = _parse_bootstrap(newick_tree)

    if not newick_tree:
        raise IQTreeError("No tree file found in CIPRES results")

    # Compute confidence values
    confidence_values = {node: value / 100.0 for node, value in bootstrap_support.items()}
    overall_confidence = (
        sum(confidence_values.values()) / len(confidence_values)
        if confidence_values
        else 0.0
    )

    return PhyloResult(
        newick_tree=newick_tree,
        model=model_name,
        bootstrap_support=bootstrap_support,
        confidence_values=confidence_values,
        overall_confidence=round(overall_confidence, 4),
    )


def _parse_model(log_content: str) -> str:
    """Extract the best model from IQ-TREE log."""
    match = re.search(r"Best-fit model:\s*(\S+)", log_content)
    if match:
        return match.group(1)
    match = re.search(r"Model found:\s*(\S+)", log_content)
    if match:
        return match.group(1)
    return "unknown"


def _parse_bootstrap(newick: str) -> dict[str, int]:
    """Parse UFBoot support values from Newick string."""
    support = {}
    # IQ-TREE 2 format: )number:
    matches_v2 = re.findall(r"\)(\d+)(?=:)", newick)
    for i, value in enumerate(matches_v2):
        val = int(value)
        if 0 <= val <= 100:
            support[f"node_{i+1}"] = val

    # Legacy format: {number}
    if not support:
        matches_legacy = re.findall(r"\{(\d+)\}", newick)
        for i, value in enumerate(matches_legacy):
            support[f"node_{i+1}"] = int(value)

    return support


def _find_element(root: ET.Element, tag: str) -> str | None:
    """Find text content of XML element."""
    for elem in root.iter(tag):
        if elem.text:
            return elem.text.strip()
    return None


def _dict_to_fasta(sequences: dict[str, str]) -> str:
    """Convert dict to FASTA format."""
    lines = []
    for name, seq in sequences.items():
        lines.append(f">{name}")
        for i in range(0, len(seq), 60):
            lines.append(seq[i : i + 60])
    return "\n".join(lines) + "\n"


class PhyloResult:
    """Result from IQ-TREE."""
    def __init__(
        self,
        newick_tree: str,
        model: str,
        bootstrap_support: dict[str, int],
        confidence_values: dict[str, float],
        overall_confidence: float,
    ):
        self.newick_tree = newick_tree
        self.model = model
        self.bootstrap_support = bootstrap_support
        self.confidence_values = confidence_values
        self.overall_confidence = overall_confidence
